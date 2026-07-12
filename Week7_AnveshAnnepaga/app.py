"""
app.py
Streamlit UI for an end-to-end RAG app:
- User pastes their own Groq API key (dynamic, never hardcoded)
- User uploads one or more documents (PDF / DOCX / TXT / MD) dynamically
- Documents are chunked, embedded locally, and stored in an in-memory FAISS index
- Questions are answered by Groq LLMs using retrieved chunks as context
"""

import streamlit as st
from vector_store import VectorStore
from rag_pipeline import ingest_files, ask

st.set_page_config(page_title="RAG Chat bot with Groq", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ---------------------------------------------------------------------------
# Sidebar: API key + document upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com/keys. Never stored on any server.",
        placeholder="gsk_...",
    )

    model = st.selectbox(
        "Groq model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )

    st.divider()
    st.subheader("📄 Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    with col2:
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=50)

    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files and st.button(f"➕ Ingest {len(new_files)} new file(s)", use_container_width=True):
            with st.spinner("Reading, chunking, and embedding documents..."):
                files_payload = [(f.name, f.read()) for f in new_files]
                num_chunks = ingest_files(
                    st.session_state.vector_store,
                    files_payload,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for f in new_files:
                    st.session_state.processed_files.add(f.name)
            st.success(f"Added {num_chunks} chunks from {len(new_files)} file(s).")

    st.divider()
    st.caption(f"📚 Indexed chunks: **{st.session_state.vector_store.num_chunks}**")
    st.caption(f"📁 Files ingested: **{len(st.session_state.processed_files)}**")
    if st.session_state.processed_files:
        for name in sorted(st.session_state.processed_files):
            st.caption(f"• {name}")

    if st.button("🗑️ Clear all documents & chat", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.processed_files = set()
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("📚 RAG Chat — Groq + FAISS")
st.caption("Upload your documents in the sidebar, add your Groq API key, then ask questions grounded in your files.")

if st.session_state.vector_store.num_chunks == 0:
    st.info("👈 Upload and ingest at least one document to get started.")

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources used"):
                for text, meta, score in msg["sources"]:
                    st.markdown(f"**{meta.get('source')}** (score: {score:.2f})")
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
    elif st.session_state.vector_store.num_chunks == 0:
        st.error("⚠️ Please upload and ingest at least one document first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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
                        with st.expander("📎 Sources used"):
                            for text, meta, score in sources:
                                st.markdown(f"**{meta.get('source')}** (score: {score:.2f})")
                                st.text(text[:500] + ("..." if len(text) > 500 else ""))

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error calling Groq API: {e}")
