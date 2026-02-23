# PDF Question-Answering System

A **FastAPI** service that lets you upload PDF documents and ask natural-language questions about them. Semantic search is powered by **FAISS** (vector index) and **sentence-transformers** (embeddings).

---

## Architecture

```
pdf-qa-system/
├── app/
│   ├── main.py      ← FastAPI routes & app lifecycle
│   ├── engine.py    ← FAISS index + sentence-transformer embeddings + QA logic
│   ├── utils.py     ← PDF text extraction (pdfplumber) & sliding-window chunking
│   └── uploads/     ← Uploaded PDFs are stored here (auto-created)
├── requirements.txt
└── README.md
```

### How it works

```
PDF upload
   │
   ▼
pdfplumber extracts raw text
   │
   ▼
Sliding-window chunker splits text into ~500-char overlapping chunks
   │
   ▼
sentence-transformers (all-MiniLM-L6-v2) encodes each chunk → 384-dim vector
   │
   ▼
FAISS IndexFlatIP stores L2-normalised vectors (cosine similarity via inner product)
   │
   ▼
Query arrives → encoded → nearest-neighbour search → top-k chunks returned
   │
   ▼
Synthesised answer + source metadata returned to caller
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU users** — swap `faiss-cpu` for `faiss-gpu` in `requirements.txt` before installing.

### 2. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Visit the interactive API docs at **http://localhost:8000/docs**.

---

## API Reference

### `GET /health`
Returns server status and index statistics.

```json
{
  "status": "ok",
  "total_documents": 2,
  "total_chunks": 148,
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dim": 384
}
```

---

### `POST /upload`
Upload a PDF. The file is parsed, chunked, and indexed.

**Request** — multipart/form-data
| Field | Type | Description |
|-------|------|-------------|
| `file` | PDF | The PDF file to upload |

**Response**
```json
{
  "doc_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "my-document.pdf",
  "num_chunks": 74,
  "message": "PDF uploaded and indexed successfully."
}
```

---

### `POST /ask`
Ask a question against the indexed documents.

**Request body**
```json
{
  "question": "What are the main findings of the study?",
  "doc_id": null,
  "top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | — | Your question |
| `doc_id` | string \| null | `null` | Scope to one document; `null` searches all |
| `top_k` | int | `5` | Number of chunks to retrieve |

**Response**
```json
{
  "question": "What are the main findings of the study?",
  "answer": "The study found that ... (Based on 5 passage(s) from all indexed documents.)",
  "sources": [
    {
      "rank": 1,
      "doc_id": "3fa85f64-...",
      "filename": "my-document.pdf",
      "chunk_index": 12,
      "score": 0.8732,
      "preview": "The study found that …"
    }
  ],
  "doc_id": null
}
```

---

### `GET /documents`
List all indexed documents.

```json
[
  {
    "doc_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "filename": "my-document.pdf",
    "num_chunks": 74
  }
]
```

---

### `DELETE /documents/{doc_id}`
Remove a document and all its chunks from the index.

```json
{ "message": "Document 3fa85f64-... deleted successfully." }
```

---

## Configuration

Key constants you can tune without touching the API surface:

| File | Constant | Default | Effect |
|------|----------|---------|--------|
| `engine.py` | `MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model |
| `utils.py` | `DEFAULT_CHUNK_SIZE` | `500` | Target chars per chunk |
| `utils.py` | `DEFAULT_CHUNK_OVERLAP` | `50` | Overlap chars between chunks |
| `utils.py` | `MIN_CHUNK_LENGTH` | `50` | Discard chunks shorter than this |

### Alternative embedding models

| Model | Dim | Notes |
|-------|-----|-------|
| `all-MiniLM-L6-v2` | 384 | Default — fast, good quality |
| `all-mpnet-base-v2` | 768 | Higher quality, slower |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual support |

> If you change the model, update `EMBEDDING_DIM` in `engine.py` to match.

---

## Example — cURL

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@research-paper.pdf"

# Ask a question (all documents)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology was used?", "top_k": 3}'

# Ask scoped to one document
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology was used?", "doc_id": "<your-doc-id>", "top_k": 3}'

# List documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/<doc-id>
```

---

## Extending the System

**Add LLM-generated answers** — pass the retrieved chunks as context to an LLM (OpenAI, Anthropic, local Ollama) inside `engine.answer_question()`.

**Persist the index** — call `faiss.write_index()` / `faiss.read_index()` and pickle `_meta` + `_docs` to disk on startup/shutdown.

**Scale** — swap `IndexFlatIP` for `IndexIVFFlat` or `IndexHNSWFlat` for million-scale corpora.

**OCR support** — pre-process scanned PDFs with `pytesseract` or `easyocr` before calling `extract_text_from_pdf`.

---

## License

MIT