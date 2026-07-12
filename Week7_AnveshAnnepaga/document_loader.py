"""
document_loader.py
Handles reading text out of uploaded files (PDF, DOCX, TXT, MD).
"""

import io
from pypdf import PdfReader
import docx


def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def read_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)


def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def load_document(filename: str, file_bytes: bytes) -> str:
    """Dispatch to the right parser based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return read_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return read_docx(file_bytes)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return read_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
