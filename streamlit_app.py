"""
ACUVUE Lens FAQ System — Streamlit Interface
============================================
UI matches the HTML frontend:
  • Dark theme: #0d0f12 bg, #3d8ef0 accent, DM Sans + DM Mono fonts
  • Two-column layout: sidebar (upload, docs, topics) + main (question, answer)
  • Card-based sections with matching border/surface colours
  • Confidence bar colour-coded green/amber/red
  • Source page pills in monospace accent style
  • Clickable topic chips that pre-fill the question input
  • Error banners and loading states
"""

import uuid
import tempfile
import os

import streamlit as st
from dotenv import load_dotenv

from app.engine import QAEngine
from app.utils import extract_and_chunk_with_pages, extract_headings
from app.llm import answer_with_groq, get_expanded_query

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACUVUE Lens FAQ System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Global CSS — mirrors the HTML frontend design system exactly
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0d0f12 !important;
    color: #e8eaf0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

.app-header {
    display: flex;
    align-items: center;
    padding: 0 0 20px 0;
    border-bottom: 1px solid #252931;
    margin-bottom: 24px;
}
.app-logo { display: flex; align-items: center; gap: 10px; }
.app-logo-icon { font-size: 1.2rem; color: #3d8ef0; }
.app-logo-text {
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem; font-weight: 500;
    letter-spacing: 0.04em; color: #e8eaf0;
}
.st-card {
    background: #13161b;
    border: 1px solid #252931;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}
.st-card-accent { border-color: #1e3a5f; }
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.card-icon {
    width: 26px; height: 26px;
    background: #1e3a5f; color: #3d8ef0;
    border-radius: 6px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 600; flex-shrink: 0;
}
.card-title { font-size: 0.9rem; font-weight: 600; letter-spacing: 0.02em; color: #e8eaf0; margin: 0; }
.answer-body {
    font-size: 0.9rem; line-height: 1.75; color: #e8eaf0;
    white-space: pre-wrap;
    border-left: 2px solid #1e3a5f;
    padding-left: 16px; margin: 12px 0;
}
.conf-wrap { display: flex; align-items: center; gap: 10px; }
.conf-label { font-size: 0.75rem; color: #7c8494; white-space: nowrap; font-family: 'DM Mono', monospace; }
.conf-bar-bg { flex:1; height: 6px; background: #1a1e25; border-radius: 3px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; }
.conf-pct { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #e8eaf0; min-width: 36px; }
.sources-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.sources-label { font-size: 0.72rem; color: #7c8494; text-transform: uppercase; letter-spacing: 0.08em; }
.source-tag {
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem;
    color: #3d8ef0; background: #1e3a5f;
    border: 1px solid rgba(61,142,240,0.25);
    padding: 3px 10px; border-radius: 20px; display: inline-block;
}
.scope-label { font-size: 0.78rem; color: #7c8494; font-style: italic; margin-bottom: 6px; }
.error-banner {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 12px 16px;
    background: rgba(240,96,96,0.08);
    border: 1px solid rgba(240,96,96,0.3);
    border-radius: 6px; color: #f06060; font-size: 0.85rem; margin: 8px 0;
}
.stTextArea textarea {
    background: #1a1e25 !important; border: 1px solid #252931 !important;
    border-radius: 6px !important; color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
.stTextArea textarea:focus { border-color: #3d8ef0 !important; box-shadow: none !important; }
.stSelectbox > div > div {
    background: #1a1e25 !important; border: 1px solid #252931 !important;
    border-radius: 6px !important; color: #e8eaf0 !important;
}
.stFileUploader {
    background: #1a1e25 !important;
    border: 1.5px dashed #252931 !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: #3d8ef0 !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.875rem !important;
    padding: 9px 20px !important; transition: background 0.15s !important;
}
.stButton > button:hover { background: #5aa3ff !important; }
[data-testid="stSidebar"] {
    background: #13161b !important;
    border-right: 1px solid #252931 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: #7c8494 !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e8eaf0 !important; font-size: 0.85rem !important;
    font-weight: 600 !important; letter-spacing: 0.02em !important;
}
[data-testid="stExpander"] {
    background: #1a1e25 !important;
    border: 1px solid #252931 !important; border-radius: 6px !important;
}
[data-testid="stMetric"] {
    background: #1a1e25; border: 1px solid #252931;
    border-radius: 6px; padding: 10px 14px;
}
[data-testid="stMetricValue"] {
    color: #3d8ef0 !important; font-family: 'DM Mono', monospace !important;
}
.stSpinner > div { border-top-color: #3d8ef0 !important; }
hr { border-color: #252931 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "engine"            not in st.session_state: st.session_state.engine            = QAEngine()
if "uploaded_docs"     not in st.session_state: st.session_state.uploaded_docs     = {}
if "prefill_question"  not in st.session_state: st.session_state.prefill_question  = ""
if "last_result"       not in st.session_state: st.session_state.last_result       = None
if "question"          not in st.session_state: st.session_state.question          = ""

engine = st.session_state.engine

# ─────────────────────────────────────────────────────────────
# Custom header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">
    <span class="app-logo-icon">👁</span>
    <span class="app-logo-text">ACUVUE Lens FAQ</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Upload ──
    st.markdown("### ↑ Upload PDF")
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        already_indexed = any(
            info["filename"] == uploaded_file.name
            for info in st.session_state.uploaded_docs.values()
        )
        if not already_indexed:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    chunk_page_pairs = extract_and_chunk_with_pages(tmp_path)
                    if not chunk_page_pairs:
                        st.error("No extractable text found.")
                    else:
                        chunks       = [c for c, _ in chunk_page_pairs]
                        page_numbers = [p for _, p in chunk_page_pairs]
                        doc_id       = str(uuid.uuid4())
                        engine.index_document(
                            doc_id=doc_id, filename=uploaded_file.name,
                            chunks=chunks, page_numbers=page_numbers,
                        )
                        st.session_state.uploaded_docs[doc_id] = {
                            "filename": uploaded_file.name,
                            "save_path": tmp_path,
                            "num_chunks": len(chunks),
                        }
                        st.success(f"✓ **{uploaded_file.name}** — {len(chunks)} chunks indexed.")
                        st.session_state.last_result = None
                except Exception as e:
                    st.error(f"Failed: {e}")
                    if os.path.exists(tmp_path): os.unlink(tmp_path)
        else:
            st.info(f"**{uploaded_file.name}** already indexed.")

    # ── Documents ──
    st.markdown("---")
    st.markdown("### ≡ Documents")
    docs = engine.list_documents()

    if not docs:
        st.caption("No documents uploaded yet.")
        selected_doc_id = None
    else:
        doc_options = {"— Search all documents —": None}
        for d in docs: doc_options[d["filename"]] = d["doc_id"]

        selected_label  = st.selectbox("Scope", list(doc_options.keys()), label_visibility="collapsed")
        selected_doc_id = doc_options[selected_label]

        for d in docs:
            c1, c2 = st.columns([5, 1])
            c1.caption(f"▣ {d['filename']}  `{d['num_chunks']}c`")
            if c2.button("✕", key=f"del_{d['doc_id']}", help="Delete"):
                engine.delete_document(d["doc_id"])
                info = st.session_state.uploaded_docs.pop(d["doc_id"], {})
                if info.get("save_path") and os.path.exists(info["save_path"]):
                    os.unlink(info["save_path"])
                st.session_state.last_result = None
                st.rerun()

    # ── Topics ──
    if selected_doc_id and selected_doc_id in st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("### 👁 Topics")
        st.caption("click to expand · click topic to ask")
        doc_info  = st.session_state.uploaded_docs[selected_doc_id]
        save_path = doc_info.get("save_path", "")

        with st.expander("Browse topics", expanded=False):
            if save_path and os.path.exists(save_path):
                try:
                    headings = extract_headings(save_path)
                    if headings:
                        for heading in headings:
                            if st.button(heading, key=f"topic_{heading[:40]}", use_container_width=True):
                                st.session_state.question = (
                                    heading if heading.endswith("?") else f"{heading}?"
                                )
                                st.session_state.last_result = None
                                st.rerun()
                    else:
                        st.caption("No topics found.")
                except Exception as e:
                    st.caption(f"Could not extract topics: {e}")
            else:
                st.caption("PDF not available.")

    # ── Stats ──
    st.markdown("---")
    st.markdown("### ◎ Index Stats")
    stats = engine.get_stats()
    ca, cb = st.columns(2)
    ca.metric("Documents", stats["total_documents"])
    cb.metric("Chunks",    stats["total_chunks"])
    st.caption(f"Model: `{stats['embedding_model']}`")

# ─────────────────────────────────────────────────────────────
# Main — Question card
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="st-card">
  <div class="card-header">
    <div class="card-icon">?</div>
    <p class="card-title">Ask a Question</p>
  </div>
</div>
""", unsafe_allow_html=True)

scope_text = (
    f"Searching: {selected_label}"
    if docs and selected_doc_id
    else "Searching all documents"
)
st.markdown(f'<p class="scope-label">{scope_text}</p>', unsafe_allow_html=True)

# If a topic chip set a prefill, move it into question state
if st.session_state.prefill_question:
    st.session_state.question = st.session_state.prefill_question
    st.session_state.prefill_question = ""

# No key= here — we control the value entirely via session state.
# Streamlit keyed widgets ignore value= after first render, which breaks
# topic prefill. Managing state manually avoids that conflict.
question = st.text_area(
    "question",
    value=st.session_state.question,
    placeholder="e.g. What care regimen should be used with ACUVUE lenses?",
    height=100,
    label_visibility="collapsed",
)
# Always write back so typing is preserved across reruns
st.session_state.question = question

col_ask, _ = st.columns([1, 4])
with col_ask:
    ask_clicked = st.button("Ask →", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# RAG pipeline
# ─────────────────────────────────────────────────────────────
if ask_clicked:
    if not st.session_state.question.strip():
        st.markdown('<div class="error-banner">⚠ Please enter a question.</div>', unsafe_allow_html=True)
    elif engine.total_chunks() == 0:
        st.markdown('<div class="error-banner">⚠ No documents indexed yet. Upload a PDF first.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Searching and generating answer..."):
            expanded_query = get_expanded_query(st.session_state.question)
            hits = engine.search(
                query=expanded_query,
                doc_id=selected_doc_id if docs else None,
                top_k=3,
            )
            result = answer_with_groq(question=st.session_state.question, hits=hits)
            st.session_state.last_result = (st.session_state.question, result)

# ─────────────────────────────────────────────────────────────
# Answer card
# ─────────────────────────────────────────────────────────────
if st.session_state.last_result:
    _, result = st.session_state.last_result
    pct = int(result.confidence * 100)
    bar_colour = "#2dd4a0" if pct >= 60 else "#f0a030" if pct >= 35 else "#f06060"

    sources_html = ""
    if result.sources:
        tags = "".join(f'<span class="source-tag">{s}</span>' for s in result.sources)
        sources_html = f'<div class="sources-row"><span class="sources-label">Sources</span>{tags}</div>'

    st.markdown(f"""
<div class="st-card st-card-accent">
  <div class="card-header">
    <div class="card-icon">✦</div>
    <p class="card-title">Answer</p>
    <div style="margin-left:auto;" class="conf-wrap">
      <span class="conf-label">Confidence</span>
      <div class="conf-bar-bg" style="width:100px;">
        <div class="conf-bar-fill" style="width:{pct}%;background:{bar_colour};"></div>
      </div>
      <span class="conf-pct">{pct}%</span>
    </div>
  </div>
  <div class="answer-body">{result.answer}</div>
  {sources_html}
</div>
""", unsafe_allow_html=True)