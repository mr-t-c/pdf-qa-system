"""
PDF Question-Answering System -- FastAPI Application (RAG edition)
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import QAEngine
from .llm import answer_with_groq, get_expanded_query
from .utils import extract_and_chunk_with_pages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

RAG_TOP_K = 3   # number of chunks to retrieve for RAG

# ---------------------------------------------------------------------------
# App & Engine
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF Question-Answering System",
    description=(
        "Upload PDFs and ask natural-language questions. "
        "Powered by FAISS semantic search + Google Gemini."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = QAEngine()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None   # None -> search all indexed documents
    top_k: int = RAG_TOP_K


class RAGAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]          # e.g. ["Page 23", "Page 28"]
    confidence: float
    doc_id: Optional[str]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


# ---------------------------------------------------------------------------
# Routes -- Health
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "PDF QA System (RAG) is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", **engine.get_stats()}


# ---------------------------------------------------------------------------
# Routes -- Documents
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    The file is parsed page-by-page, chunked (with page numbers preserved),
    and indexed into FAISS. Returns a doc_id to scope future questions.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    # Save to disk
    try:
        contents = await file.read()
        save_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Extract and chunk with page numbers
    try:
        chunk_page_pairs = extract_and_chunk_with_pages(str(save_path))
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {exc}")

    if not chunk_page_pairs:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. The PDF may be scanned/image-based.",
        )

    chunks = [c for c, _ in chunk_page_pairs]
    page_numbers = [p for _, p in chunk_page_pairs]

    # Index into FAISS
    try:
        engine.index_document(
            doc_id=doc_id,
            filename=file.filename,
            chunks=chunks,
            page_numbers=page_numbers,
        )
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        message="PDF uploaded and indexed successfully.",
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
def list_documents():
    """List all indexed documents."""
    return engine.list_documents()


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """Remove a document and all its chunks from the index."""
    try:
        engine.delete_document(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    for path in UPLOAD_DIR.glob(f"{doc_id}_*"):
        path.unlink(missing_ok=True)

    return {"message": f"Document {doc_id} deleted successfully."}


# ---------------------------------------------------------------------------
# Routes -- QA (RAG)
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=RAGAnswerResponse, tags=["QA"])
def ask_question(request: QuestionRequest):
    """
    RAG-based question answering.

    Flow
    ----
    1. Expand the query (short questions get enriched for better retrieval).
    2. Embed the expanded query and retrieve top-k chunks via FAISS.
    3. If top similarity < 0.3 -> return clarification response (no LLM call).
    4. Otherwise -> build grounded prompt and call Gemini 1.5 Flash.
    5. Return structured JSON with answer, page sources, and confidence score.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if engine.total_chunks() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Upload a PDF first.",
        )

    if request.doc_id and request.doc_id not in {
        d["doc_id"] for d in engine.list_documents()
    }:
        raise HTTPException(
            status_code=404,
            detail=f"doc_id '{request.doc_id}' not found.",
        )

    # Expand query for better FAISS retrieval on short questions
    expanded_query = get_expanded_query(request.question)

    # Retrieve using the expanded query
    hits = engine.search(
        query=expanded_query,
        doc_id=request.doc_id,
        top_k=request.top_k,
    )

    # Generate via Gemini (passes original question to keep the answer natural)
    rag_result = answer_with_groq(question=request.question, hits=hits)

    return RAGAnswerResponse(
        question=request.question,
        answer=rag_result.answer,
        sources=rag_result.sources,
        confidence=rag_result.confidence,
        doc_id=request.doc_id,
    )