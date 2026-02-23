"""
QA Engine — FAISS vector store + sentence-transformer embeddings.

Design
──────
• One *global* FAISS index (IndexFlatIP — inner-product / cosine after L2-norm).
• Each chunk is stored as a row; a parallel metadata list mirrors the index.
• Documents can be added or removed at runtime.  Deletion rebuilds the index
  from the surviving metadata (FAISS doesn't support in-place removal).
• ChunkMeta carries page_number so callers (e.g. the LLM layer) can cite pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from textwrap import shorten
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # Fast & accurate; 384-dim embeddings
EMBEDDING_DIM = 384


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class ChunkMeta:
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    page_number: int = 1        # ← NEW: 1-based page number from the source PDF


@dataclass
class QAEngine:
    """Thin wrapper around a FAISS flat index with document management."""

    model_name: str = MODEL_NAME
    _model: SentenceTransformer = field(init=False, repr=False)
    _index: faiss.IndexFlatIP = field(init=False, repr=False)
    _meta: list[ChunkMeta] = field(default_factory=list)
    # doc_id → {filename, num_chunks}
    _docs: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        logger.info("Loading sentence-transformer model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        logger.info("FAISS IndexFlatIP initialised (dim=%d)", EMBEDDING_DIM)

    # ── Embedding helpers ─────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings (shape: N × dim, dtype float32)."""
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine via inner product
        ).astype("float32")
        return vecs

    # ── Indexing ──────────────────────────────────────────────────────────────
    def index_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        page_numbers: list[int] | None = None,   # ← NEW: parallel list of page numbers
    ) -> None:
        """
        Embed and index all chunks for a document.

        Parameters
        ----------
        chunks       : list of text chunks
        page_numbers : optional parallel list mapping each chunk to its PDF page.
                       Falls back to 1 for every chunk when omitted.
        """
        if not chunks:
            raise ValueError("chunks list is empty — nothing to index.")

        if page_numbers is None:
            page_numbers = [1] * len(chunks)

        if len(page_numbers) != len(chunks):
            raise ValueError("page_numbers must have the same length as chunks.")

        logger.info(
            "Indexing doc_id=%s (%s) — %d chunks", doc_id, filename, len(chunks)
        )
        vecs = self._embed(chunks)

        # Build metadata
        new_meta = [
            ChunkMeta(
                doc_id=doc_id,
                filename=filename,
                chunk_index=i,
                text=chunk,
                page_number=page_numbers[i],
            )
            for i, chunk in enumerate(chunks)
        ]

        self._index.add(vecs)
        self._meta.extend(new_meta)
        self._docs[doc_id] = {"filename": filename, "num_chunks": len(chunks)}
        logger.info("Total vectors in index: %d", self._index.ntotal)

    # ── Search ────────────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[tuple[ChunkMeta, float]]:
        """
        Return up to *top_k* (ChunkMeta, score) pairs, sorted by relevance.
        When *doc_id* is given, only chunks from that document are returned.
        """
        if self._index.ntotal == 0:
            return []

        q_vec = self._embed([query])

        # Over-fetch when filtering so we still get top_k after the filter
        fetch_k = self._index.ntotal if doc_id else top_k
        scores, indices = self._index.search(q_vec, k=min(fetch_k, self._index.ntotal))

        results: list[tuple[ChunkMeta, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._meta[idx]
            if doc_id and meta.doc_id != doc_id:
                continue
            results.append((meta, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ── QA ────────────────────────────────────────────────────────────────────
    def answer_question(
        self,
        question: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> tuple[str, list[dict]]:
        """
        Retrieve the most relevant chunks and synthesise a plain-text answer.

        Returns
        ───────
        answer  : str   — synthesised response
        sources : list  — metadata for each chunk used
        """
        if doc_id and doc_id not in self._docs:
            raise ValueError(f"No document with doc_id='{doc_id}' found in the index.")

        hits = self.search(question, doc_id=doc_id, top_k=top_k)
        if not hits:
            return (
                "I could not find any relevant information in the indexed documents.",
                [],
            )

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        for rank, (meta, score) in enumerate(hits, start=1):
            context_parts.append(f"[{rank}] {meta.text}")
            sources.append(
                {
                    "rank": rank,
                    "doc_id": meta.doc_id,
                    "filename": meta.filename,
                    "chunk_index": meta.chunk_index,
                    "page_number": meta.page_number,
                    "score": round(score, 4),
                    "preview": shorten(meta.text, width=200, placeholder="…"),
                }
            )

        context = "\n\n".join(context_parts)

        # Simple extractive answer: most relevant chunk + citation hint
        top_text = hits[0][0].text
        answer = (
            f"{top_text}\n\n"
            f"(Based on {len(hits)} passage(s) from "
            f"{'document ' + doc_id if doc_id else 'all indexed documents'}.)"
        )

        return answer, sources

    # ── Document management ───────────────────────────────────────────────────
    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for *doc_id* and rebuild the FAISS index."""
        if doc_id not in self._docs:
            raise ValueError(f"Document '{doc_id}' not found.")

        # Filter surviving metadata
        surviving = [m for m in self._meta if m.doc_id != doc_id]

        # Rebuild FAISS from surviving chunks
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if surviving:
            texts = [m.text for m in surviving]
            vecs = self._embed(texts)
            new_index.add(vecs)

        self._index = new_index
        self._meta = surviving
        del self._docs[doc_id]
        logger.info(
            "Deleted doc_id=%s. Vectors remaining: %d", doc_id, self._index.ntotal
        )

    def list_documents(self) -> list[dict]:
        return [
            {"doc_id": did, "filename": info["filename"], "num_chunks": info["num_chunks"]}
            for did, info in self._docs.items()
        ]

    def total_chunks(self) -> int:
        return self._index.ntotal

    def get_stats(self) -> dict:
        return {
            "total_documents": len(self._docs),
            "total_chunks": self._index.ntotal,
            "embedding_model": self.model_name,
            "embedding_dim": EMBEDDING_DIM,
        }