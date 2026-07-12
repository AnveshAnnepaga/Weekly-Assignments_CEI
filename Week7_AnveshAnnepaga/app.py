"""
app.py
Streamlit UI for an end-to-end RAG app — "The Reading Room" theme.

- User pastes their own Groq API key (dynamic, never hardcoded)
- User uploads one or more documents (PDF / DOCX / TXT / MD) dynamically
- Documents are chunked, embedded locally, and stored in an in-memory FAISS index
- Questions are answered by Groq LLMs using retrieved chunks as context
"""

import streamlit as st
from vector_store import VectorStore
from rag_pipeline import ingest_files, ask

st.set_page_config(page_title="The Reading Room", page_icon="🗃️", layout="wide")

# ---------------------------------------------------------------------------
# Design system — "card catalog / reading room" theme
# Ink navy background, brass accent, warm paper panels, serif display type,
# monospace used for metadata (mimics catalog card typewriter labels).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #10131A;
        --ink-panel: #171B24;
        --paper: #F3ECDA;
        --paper-dim: #C9C2AE;
        --brass: #C0923F;
        --brass-dim: #8A6A32;
        --sage: #7FA383;
        --rule: #2B3140;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--ink);
        color: var(--paper);
    }

    /* ---- Headline ---- */
    .archive-header {
        border-bottom: 1px solid var(--rule);
        padding-bottom: 1.1rem;
        margin-bottom: 1.6rem;
    }
    .archive-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--brass);
        margin-bottom: 0.35rem;
    }
    .archive-title {
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        font-size: 2.4rem;
        color: var(--paper);
        line-height: 1.15;
        margin: 0;
    }
    .archive-sub {
        font-family: 'Source Serif 4', serif;
        font-style: italic;
        font-size: 1.02rem;
        color: var(--paper-dim);
        margin-top: 0.4rem;
    }

    /* ---- Sidebar restyle ---- */
    section[data-testid="stSidebar"] {
        background: var(--ink-panel);
        border-right: 1px solid var(--rule);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.85rem;
        color: var(--brass);
        border-bottom: 1px solid var(--rule);
        padding-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.04em;
        color: var(--paper-dim) !important;
        text-transform: uppercase;
    }

    /* ---- Catalog card (file list entries) ---- */
    .catalog-card {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--paper);
        background: var(--ink);
        border: 1px dashed var(--rule);
        border-left: 3px solid var(--brass);
        padding: 0.45rem 0.6rem;
        margin-bottom: 0.4rem;
        border-radius: 2px;
    }

    /* ---- Stat strip ---- */
    .stat-strip {
        display: flex;
        gap: 0.6rem;
        margin: 0.6rem 0 1rem 0;
    }
    .stat-box {
        flex: 1;
        background: var(--ink-panel);
        border: 1px solid var(--rule);
        border-top: 2px solid var(--brass);
        padding: 0.55rem 0.7rem;
        text-align: left;
    }
    .stat-num {
        font-family: 'Source Serif 4', serif;
        font-size: 1.5rem;
        color: var(--paper);
        font-weight: 700;
        line-height: 1;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--paper-dim);
        margin-top: 0.25rem;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: transparent;
        border: 1px solid var(--brass);
        color: var(--brass);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 2px;
        padding: 0.5rem 0.9rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--brass);
        color: var(--ink);
        border-color: var(--brass);
    }

    /* ---- Chat bubbles ---- */
    div[data-testid="stChatMessage"] {
        background: var(--ink-panel);
        border: 1px solid var(--rule);
        border-radius: 4px;
        padding: 0.4rem 0.6rem;
    }

    /* ---- Empty state ---- */
    .empty-slip {
        border: 1px dashed var(--rule);
        padding: 1.4rem;
        text-align: center;
        font-family: 'Source Serif 4', serif;
        font-style: italic;
        color: var(--paper-dim);
        margin-top: 1rem;
    }

    /* ---- Source expander tag ---- */
    .source-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--sage);
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ---------------------------------------------------------------------------
# Sidebar — "Catalog Desk"
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 Access Slip")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at console.groq.com/keys. Held only for this session.",
        placeholder="gsk_...",
        label_visibility="visible",
    )

    model = st.selectbox(
        "Reference Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )

    st.markdown("### 🗃️ File Intake")
    uploaded_files = st.file_uploader(
        "Deposit documents (PDF · DOCX · TXT · MD)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Slip size", min_value=200, max_value=4000, value=1000, step=100)
    with col2:
        chunk_overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=150, step=50)

    top_k = st.slider("Slips retrieved per query", min_value=1, max_value=10, value=4)

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files and st.button(f"File {len(new_files)} new document(s)", use_container_width=True):
            with st.spinner("Cataloging documents..."):
                files_payload = [(f.name, f.read()) for f in new_files]
                num_chunks = ingest_files(
                    st.session_state.vector_store,
                    files_payload,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for f in new_files:
                    st.session_state.processed_files.add(f.name)
            st.success(f"Indexed {num_chunks} slips from {len(new_files)} document(s).")

    st.markdown("### 📇 Catalog Index")
    st.markdown(
        f"""
        <div class="stat-strip">
            <div class="stat-box">
                <div class="stat-num">{st.session_state.vector_store.num_chunks}</div>
                <div class="stat-label">Slips indexed</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">{len(st.session_state.processed_files)}</div>
                <div class="stat-label">Documents filed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.processed_files:
        for name in sorted(st.session_state.processed_files):
            st.markdown(f'<div class="catalog-card">📄 {name}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear the shelf", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.processed_files = set()
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main — "Reading Room"
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="archive-header">
        <div class="archive-eyebrow">Groq · FAISS · Local Embeddings</div>
        <p class="archive-title">The Reading Room</p>
        <p class="archive-sub">File your documents at the catalog desk, then ask the room anything they contain.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vector_store.num_chunks == 0:
    st.markdown(
        '<div class="empty-slip">The shelf is empty. File a document at the Catalog Desk to begin.</div>',
        unsafe_allow_html=True,
    )

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Slips consulted"):
                for text, meta, score in msg["sources"]:
                    st.markdown(
                        f'<span class="source-tag">{meta.get("source")} · relevance {score:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))

question = st.chat_input("Ask the room about your documents...")

if question:
    if not api_key:
        st.error("⚠️ Present your Groq API key at the Catalog Desk first.")
    elif st.session_state.vector_store.num_chunks == 0:
        st.error("⚠️ File at least one document before asking the room.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Pulling slips from the shelf..."):
                try:
                    prior_turns = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                    ]
                    answer, sources = ask(
                        api_key=api_key,
                        vector_store=st.session_state.vector_store,
                        question=question,
                        model=model,
                        top_k=top_k,
                        chat_history=prior_turns,
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Slips consulted"):
                            for text, meta, score in sources:
                                st.markdown(
                                    f'<span class="source-tag">{meta.get("source")} · relevance {score:.2f}</span>',
                                    unsafe_allow_html=True,
                                )
                                st.text(text[:500] + ("..." if len(text) > 500 else ""))

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error calling Groq API: {e}")