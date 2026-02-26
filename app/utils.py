"""
Utility helpers for PDF text extraction and text chunking.

Page number strategy
--------------------
This PDF embeds cited page numbers inline in the content, e.g. "Page 27" at
the end of each answer block. We prefer those cited page numbers over the
physical PDF page number (which just tells us which sheet of paper the text
is on). extract_cited_page() detects these inline citations.

Functions
---------
* extract_pages_from_pdf()      -> list[(physical_page, text)]
* extract_text_from_pdf()       -> plain str (backward-compat)
* chunk_text()                  -> list[str] (backward-compat)
* chunk_text_with_pages()       -> list[(chunk, cited_page)]
* extract_and_chunk_with_pages()-> end-to-end pipeline
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH      = 50

# Matches inline page citations like:
#   "Page 27", "Page 27 ", "page 23", "Page 27\n"
# at the END of a chunk (last ~20 chars) or anywhere in the text.
_CITED_PAGE_RE = re.compile(r"\bpage\s+(\d+)", re.IGNORECASE)



def extract_cited_page(text: str, fallback: int = 1) -> int:
    """
    Return the LAST inline page citation found in *text* (e.g. "Page 27").

    This PDF format appends "Page XX" at the end of each answer block, so
    taking the last match gives us the most relevant cited page for that chunk.

    Falls back to *fallback* (the physical PDF page number) when no citation
    is found.

    Examples
    --------
    "...feel uncomfortable. Page 27"  -> 27
    "No citation here"                -> fallback
    """
    matches = _CITED_PAGE_RE.findall(text)
    if matches:
        return int(matches[-1])   # last citation in the chunk
    return fallback


def extract_pages_from_pdf(filepath: str | Path) -> list[tuple[int, str]]:
    """
    Extract text per page, preserving 1-based PHYSICAL page numbers.

    Returns list of (physical_page_number, page_text).
    Skips pages with no extractable text.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((page_num, text))
                else:
                    logger.debug("Page %d yielded no text (possibly image-based).", page_num)
    except Exception as exc:
        raise RuntimeError(f"pdfplumber failed on '{path}': {exc}") from exc

    logger.info("Extracted %d page(s) from %s", len(pages), path.name)
    return pages


def extract_text_from_pdf(filepath: str | Path) -> str:
    """Backward-compatible: returns all page text joined by form-feeds."""
    pages = extract_pages_from_pdf(filepath)
    return "\f".join(text for _, text in pages)



def clean_text(text: str) -> str:
    """Normalise whitespace without destroying sentence/paragraph structure."""
    text = re.sub(r"\f",    "\n\n", text)
    text = re.sub(r"[ \t]+", " ",   text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    splitter = re.compile(r"(?<=[.!?])\s+")
    return [s.strip() for s in splitter.split(text) if s.strip()]



def chunk_text(
    text: str,
    chunk_size:       int = DEFAULT_CHUNK_SIZE,
    chunk_overlap:    int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_length: int = MIN_CHUNK_LENGTH,
) -> list[str]:
    """Backward-compatible chunker — returns plain chunk strings."""
    text = clean_text(text)
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for s in sentences:
        slen = len(s)
        if cur_len + slen > chunk_size and cur:
            chunk = " ".join(cur).strip()
            if len(chunk) >= min_chunk_length:
                chunks.append(chunk)
            overlap, olen = [], 0
            for prev in reversed(cur):
                if olen + len(prev) <= chunk_overlap:
                    overlap.insert(0, prev)
                    olen += len(prev)
                else:
                    break
            cur, cur_len = overlap, olen
        cur.append(s)
        cur_len += slen + 1

    if cur:
        chunk = " ".join(cur).strip()
        if len(chunk) >= min_chunk_length:
            chunks.append(chunk)

    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    logger.info("chunk_text: %d chunks from %d chars", len(unique), len(text))
    return unique


def chunk_text_with_pages(
    pages:            list[tuple[int, str]],
    chunk_size:       int = DEFAULT_CHUNK_SIZE,
    chunk_overlap:    int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_length: int = MIN_CHUNK_LENGTH,
) -> list[tuple[str, int]]:
    """
    Chunk (physical_page, text) pairs and return (chunk_text, cited_page).

    Page number resolution order
    ----------------------------
    1. If the chunk contains an inline citation like "Page 27", use that.
    2. Otherwise fall back to the physical PDF page where the chunk starts.

    This means answers in this QA-format PDF will always cite the correct
    document page (e.g. 27) rather than which sheet of paper they appear on.
    """
    # Flatten pages into (sentence, physical_page) pairs
    sentence_page_pairs: list[tuple[str, int]] = []
    for phys_page, page_text in pages:
        cleaned = clean_text(page_text)
        for s in _split_sentences(cleaned):
            sentence_page_pairs.append((s, phys_page))

    if not sentence_page_pairs:
        return []

    result:       list[tuple[str, int]] = []
    cur_sentences: list[str]            = []
    cur_phys:      list[int]            = []
    cur_len = 0

    for sentence, phys_page in sentence_page_pairs:
        slen = len(sentence)

        if cur_len + slen > chunk_size and cur_sentences:
            chunk = " ".join(cur_sentences).strip()
            if len(chunk) >= min_chunk_length:
                # Prefer inline cited page; fall back to physical page of
                # the first sentence in this chunk.
                cited = extract_cited_page(chunk, fallback=cur_phys[0])
                result.append((chunk, cited))

            # Build overlap window
            ov_s: list[str] = []
            ov_p: list[int] = []
            ov_len = 0
            for s, p in zip(reversed(cur_sentences), reversed(cur_phys)):
                if ov_len + len(s) <= chunk_overlap:
                    ov_s.insert(0, s)
                    ov_p.insert(0, p)
                    ov_len += len(s)
                else:
                    break
            cur_sentences, cur_phys, cur_len = ov_s, ov_p, ov_len

        cur_sentences.append(sentence)
        cur_phys.append(phys_page)
        cur_len += slen + 1

    # Flush final chunk
    if cur_sentences:
        chunk = " ".join(cur_sentences).strip()
        if len(chunk) >= min_chunk_length:
            cited = extract_cited_page(chunk, fallback=cur_phys[0])
            result.append((chunk, cited))

    # De-duplicate by text
    seen:   set[str]             = set()
    unique: list[tuple[str, int]] = []
    for chunk, page in result:
        if chunk not in seen:
            seen.add(chunk)
            unique.append((chunk, page))

    logger.info(
        "chunk_text_with_pages: %d chunks, page refs resolved via inline citations",
        len(unique),
    )
    return unique


def extract_and_chunk(
    filepath:      str | Path,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Backward-compatible: PDF -> plain chunk strings."""
    text = extract_text_from_pdf(filepath)
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def extract_and_chunk_with_pages(
    filepath:      str | Path,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, int]]:
    """Full pipeline: PDF -> (chunk_text, cited_page_number) tuples."""
    pages = extract_pages_from_pdf(filepath)
    return chunk_text_with_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# Patterns that strongly suggest a line is a heading in this QA-format PDF:
# 1. Lines that are a question (start with What/How/Can/Should/Why/Is/Do/Are)
# 2. Lines in ALL CAPS (section titles)
# 3. Short lines (< 80 chars) that don't end with common sentence punctuation
_QUESTION_START = re.compile(
    r"^(what|how|can|should|why|is|do|are|when|where|which|who)\s",
    re.IGNORECASE,
)


def extract_headings(filepath: str | Path, max_headings: int = 12) -> list[str]:
    """
    Extract meaningful topic headings from a PDF for navigation display.

    Rules (stricter than before to avoid over-extraction)
    -----------------------------------------------------
    1. Question lines that are SHORT (< 80 chars) and end with "?"
       — covers the QA-format PDFs where every topic is phrased as a question
    2. ALL CAPS lines with 3-6 words — section titles in formal documents
       (capped at 6 words to skip long ALL-CAPS sentences)

    Rule 3 (short non-punctuated lines) was removed because it matched
    too many mid-sentence fragments in this PDF format.

    Deduplicates and caps at *max_headings* results.

    Returns
    -------
    list[str] — cleaned heading strings, ready for display.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    headings: list[str] = []
    seen:     set[str]  = set()

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                for raw_line in text.splitlines():
                    line = raw_line.strip()

                    # Skip empty or very short lines
                    if len(line) < 10:
                        continue

                    # Skip pure page citations e.g. "Page 27"
                    if _CITED_PAGE_RE.fullmatch(line):
                        continue

                    is_heading = False

                    # Rule 1: question that ends with "?" and is short enough
                    # to be a heading, not a paragraph mid-sentence question
                    if (
                        _QUESTION_START.match(line)
                        and line.endswith("?")
                        and len(line) < 80
                    ):
                        is_heading = True

                    # Rule 2: ALL CAPS line, 3–6 words (avoids long cap sentences)
                    elif (
                        line.isupper()
                        and 3 <= len(line.split()) <= 6
                        and len(line) < 60
                    ):
                        is_heading = True

                    if is_heading:
                        cleaned = " ".join(line.split())
                        key     = cleaned.lower()
                        if key not in seen:
                            seen.add(key)
                            headings.append(cleaned)

                    if len(headings) >= max_headings:
                        break

    except Exception as exc:
        raise RuntimeError(f"Heading extraction failed on '{path}': {exc}") from exc

    logger.info("Extracted %d heading(s) from %s", len(headings), path.name)
    return headings[:max_headings]