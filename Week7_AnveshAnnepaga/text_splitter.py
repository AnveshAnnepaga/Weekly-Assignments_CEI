"""
text_splitter.py
Simple, dependency-light recursive character text splitter
(chunks text into overlapping pieces for embedding).
"""

from typing import List


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Split text into overlapping chunks, trying to break on paragraph/sentence
    boundaries where possible.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text, separators, chunk_size, chunk_overlap)


def _recursive_split(text, separators, chunk_size, chunk_overlap):
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    if not separators:
        # Hard split
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
        return [c for c in chunks if c]

    sep = separators[0]
    parts = text.split(sep)

    chunks = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(part) > chunk_size:
                # part itself too big, recurse with next separator
                sub_chunks = _recursive_split(
                    part, separators[1:], chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part
    if current:
        chunks.append(current.strip())

    # Add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        return overlapped

    return [c for c in chunks if c]
