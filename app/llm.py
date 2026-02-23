"""
LLM Layer -- Groq API integration for RAG-based question answering.

Provider : Groq (https://console.groq.com)
Model    : llama-3.3-70b-versatile (free tier, very fast)
SDK      : openai-compatible via groq package

Free tier limits (as of 2025)
------------------------------
* 14,400 requests / day
* 6,000 requests / minute  
* 6,000 tokens / minute
* 500,000 tokens / day

Improvements over Gemini version
----------------------------------
* Retry with exponential backoff on 429 rate-limit errors.
* Context trimming per chunk to stay within token limits.
* Clean user-facing messages for all error types.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq, RateLimitError as GroqRateLimitError
from .utils import _CITED_PAGE_RE

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"

LOW_CONFIDENCE_THRESHOLD = 0.3

SHORT_QUERY_WORD_LIMIT = 5

# Trim each retrieved chunk to this many characters before sending to Groq.
# Keeps token usage low and well within free-tier limits.
MAX_CHARS_PER_CHUNK = 800

# Retry settings for 429 rate-limit errors
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [5, 15, 30]

SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable call center assistant for an ACUVUE contact lens support line. "
    "Your job is to help customers with their questions in a warm, natural, conversational tone — "
    "as if you are speaking to them directly on a call. "
    "\n\n"
    "Guidelines:\n"
    "- Greet the answer naturally, as a real agent would (e.g. 'Great question!' or 'Sure, I can help with that!')\n"
    "- Break your answer into clear, easy-to-follow steps or points when the answer has multiple parts\n"
    "- Use simple, everyday language — avoid robotic or overly formal phrasing\n"
    "- Always include the page reference naturally in the flow of your answer "
    "(e.g. \'as mentioned in our guide on Page 27\' — NOT as a raw citation at the end)\n"
    "- Answer ONLY using the provided context — do not add information not present in the context\n"
    "- If the answer is not in the context, say: \'I\'m sorry, I don\'t have that information on hand "
    "— I\'d recommend speaking with your eye care professional for guidance.\'\n"
    "- End with a short, helpful closing line (e.g. \'Let me know if you have any other questions!\')\n"
)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_client() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY is not set. /ask will return a config error.")
        return None
    client = Groq(api_key=api_key)
    logger.info("Groq client initialised (model=%s)", GROQ_MODEL)
    return client


_client: Groq | None = _init_client()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Raised when all retries are exhausted due to 429 responses."""


def _call_groq_with_retry(prompt: str) -> str:
    """
    Call Groq and retry up to MAX_RETRIES times on 429 errors.
    Raises RateLimitError if all retries are exhausted.
    Raises the original exception for any other error type.
    """
    if _client is None:
        raise RuntimeError("Groq client is not initialised (missing GROQ_API_KEY).")

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except GroqRateLimitError as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS[attempt]
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                logger.error("All %d retries exhausted due to rate limiting.", MAX_RETRIES)
                raise RateLimitError(str(exc)) from exc

        except Exception:
            raise   # non-rate-limit errors bubble up immediately


def _trim_chunk(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    """Trim a chunk to max_chars, cutting at the last sentence boundary."""
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    last_period = max(trimmed.rfind(". "), trimmed.rfind(".\n"))
    if last_period > max_chars // 2:
        return trimmed[: last_period + 1]
    return trimmed + "..."


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(question: str) -> str:
    """
    Rewrite short/vague questions into richer retrieval queries.
    Returns unchanged if already detailed (>= SHORT_QUERY_WORD_LIMIT words).
    """
    words = question.strip().split()
    if len(words) >= SHORT_QUERY_WORD_LIMIT:
        return question

    base  = question.rstrip("?!. ")
    lower = base.lower()

    if lower.startswith("what is ") or lower.startswith("what are "):
        topic    = base.split(" ", 2)[-1]
        expanded = f"{question} definition and explanation of {topic}"
    elif lower.startswith("how do") or lower.startswith("how can") or lower.startswith("how to"):
        topic    = base.split(" ", 2)[-1]
        expanded = f"{question} steps to {topic}"
    elif lower.startswith("can i") or lower.startswith("should i"):
        expanded = f"{question} guidelines and recommendations"
    else:
        expanded = f"{question} details and explanation"

    logger.debug("Query expanded: %r -> %r", question, expanded)
    return expanded


def get_expanded_query(question: str) -> str:
    """Public helper for main.py to get the expanded query for FAISS search."""
    return expand_query(question)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    answer: str
    sources: list[str]          # e.g. ["Page 23", "Page 28"]
    confidence: float
    expanded_query: str | None


def answer_with_groq(
    question: str,
    hits: list[tuple],          # list of (ChunkMeta, score) from engine.search()
) -> RAGAnswer:
    """
    Run the full RAG pipeline for one question using Groq + Llama 3.3 70B.

    Parameters
    ----------
    question : original user question (used in the prompt as-is)
    hits     : ranked (ChunkMeta, score) tuples — retrieve with expanded query
    """
    expanded = expand_query(question)

    if not hits:
        return RAGAnswer(
            answer="I could not find any relevant information in the document.",
            sources=[],
            confidence=0.0,
            expanded_query=expanded,
        )

    top_score = float(hits[0][1])

    # --- Confidence gate ------------------------------------------------
    if top_score < LOW_CONFIDENCE_THRESHOLD:
        logger.info("Top similarity %.4f below threshold -- skipping LLM.", top_score)
        return RAGAnswer(
            answer=(
                "I could not find a clear answer in the document. "
                "Could you rephrase your question or provide more detail?"
            ),
            sources=[],
            confidence=round(top_score, 4),
            expanded_query=expanded,
        )

    # --- Build trimmed context ------------------------------------------
    # Source citation rules:
    # 1. Only cite a page if the chunk contains an explicit inline "Page XX"
    #    citation in the text — chunks that fell back to the physical PDF page
    #    number (no inline citation found) are excluded from sources.
    # 2. Additionally the chunk score must be within SOURCE_SCORE_MARGIN of
    #    the top score so low-relevance hits don't pollute the sources list.
    SOURCE_SCORE_MARGIN = 0.15

    context_blocks: list[str] = []
    source_labels:  list[str] = []
    seen_pages:     set[int]  = set()

    for rank, (meta, score) in enumerate(hits, start=1):
        page_label   = f"Page {meta.page_number}"
        trimmed_text = _trim_chunk(meta.text)
        # Always pass chunk to LLM as context regardless of citation status
        context_blocks.append(f"[Passage {rank} | {page_label}]\n{trimmed_text}")
        # Only surface as a cited source if BOTH conditions are met:
        # a) chunk has an explicit inline page citation (not a physical fallback)
        # b) chunk score is close to the top score
        has_inline_citation = bool(_CITED_PAGE_RE.search(meta.text))
        score_is_close      = (top_score - float(score)) <= SOURCE_SCORE_MARGIN
        if has_inline_citation and score_is_close:
            if meta.page_number not in seen_pages:
                source_labels.append(page_label)
                seen_pages.add(meta.page_number)

    context = "\n\n".join(context_blocks)
    prompt  = f"Context:\n{context}\n\nQuestion:\n{question}"

    # --- LLM not configured ---------------------------------------------
    if _client is None:
        return RAGAnswer(
            answer=(
                "The LLM is not configured. "
                "Please set the GROQ_API_KEY environment variable and restart the server."
            ),
            sources=source_labels,
            confidence=round(top_score, 4),
            expanded_query=expanded,
        )

    # --- Call Groq with retry -------------------------------------------
    try:
        logger.debug("Sending prompt to Groq (%d chars).", len(prompt))
        answer_text = _call_groq_with_retry(prompt)

    except RateLimitError:
        return RAGAnswer(
            answer=(
                "The API rate limit has been reached. "
                "Please wait a moment and try again."
            ),
            sources=source_labels,
            confidence=round(top_score, 4),
            expanded_query=expanded,
        )
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        return RAGAnswer(
            answer=f"The language model returned an error: {exc}",
            sources=source_labels,
            confidence=round(top_score, 4),
            expanded_query=expanded,
        )

    logger.info("Groq responded successfully (top_score=%.4f).", top_score)
    return RAGAnswer(
        answer=answer_text,
        sources=source_labels,
        confidence=round(top_score, 4),
        expanded_query=expanded,
    )