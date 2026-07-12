"""
vector_store.py
Wraps sentence-transformers embeddings + a FAISS index so we can
add document chunks and run similarity search over them.
"""

from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    In-memory FAISS-backed vector store.
    Uses a local sentence-transformers model for embeddings
    (Groq does not currently serve an embeddings endpoint, so
    embeddings are computed locally / for free, and only the
    generation step calls the Groq API).
    """

    _EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self.index = None
        self.dim = None
        self.texts: List[str] = []
        self.metadatas: List[dict] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._EMBED_MODEL_NAME)
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        # normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        return vectors.astype("float32")

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        if not texts:
            return
        vectors = self._embed(texts)

        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, dict, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self._embed([query])
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.texts[idx], self.metadatas[idx], float(score)))
        return results

    @property
    def num_chunks(self) -> int:
        return len(self.texts)

    def clear(self):
        self.index = None
        self.texts = []
        self.metadatas = []
