"""
rag_pipeline.py
Ties together retrieval (VectorStore) and generation (Groq API)
into a single ask() call, plus the ingestion pipeline for uploaded files.
"""

from typing import List, Tuple
from groq import Groq

from document_loader import load_document
from text_splitter import split_text
from vector_store import VectorStore

SYSTEM_PROMPT = """You are a helpful assistant answering questions strictly using the \
provided context extracted from the user's uploaded documents.

Rules:
- Answer only using the context below. If the answer isn't in the context, say so honestly.
- Be concise and precise.
- Where useful, mention which document/source the info came from.
- Do not make up facts that are not supported by the context.
"""


def ingest_files(vector_store: VectorStore, files, chunk_size=1000, chunk_overlap=150):
    """
    files: list of (filename, bytes) tuples
    Returns number of chunks added.
    """
    all_chunks = []
    all_metas = []

    for filename, file_bytes in files:
        text = load_document(filename, file_bytes)
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append({"source": filename, "chunk_index": i})

    vector_store.add_texts(all_chunks, all_metas)
    return len(all_chunks)


def build_context(results: List[Tuple[str, dict, float]]) -> str:
    blocks = []
    for text, meta, score in results:
        source = meta.get("source", "unknown")
        blocks.append(f"[Source: {source} | relevance: {score:.2f}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def ask(
    api_key: str,
    vector_store: VectorStore,
    question: str,
    model: str = "llama-3.3-70b-versatile",
    top_k: int = 4,
    chat_history: list | None = None,
):
    """
    Runs retrieval + Groq generation for a single question.
    Returns (answer_text, retrieved_results).
    """
    client = Groq(api_key=api_key)

    results = vector_store.similarity_search(question, k=top_k)
    context = build_context(results) if results else "No relevant context found."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history[-6:])  # keep last few turns for continuity

    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    return answer, results
