"""Chunking, embeddings and the FAISS vector database (build / save / load)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loader import load_documents

def get_embeddings() -> HuggingFaceEmbeddings:
    # Normalised vectors let us turn FAISS distances into cosine similarity later.
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def build_session_index(embeddings, files: list[tuple[str, bytes]]):
    """Build an in-memory FAISS index from uploaded files. Nothing is kept on disk."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for name, data in files:
            (Path(tmp) / Path(name).name).write_bytes(data)
        documents, errors = load_documents(Path(tmp))  # text is read into memory here
    if not documents:
        detail = f" Errors: {'; '.join(errors)}" if errors else ""
        raise ValueError("No readable text found in your files." + detail)

    chunks = [c for c in split_documents(documents) if c.page_content.strip()]
    db = FAISS.from_documents(chunks, embeddings)
    stats = {
        "files": sorted({d.metadata["file"] for d in documents}),
        "pages": len(documents),
        "chunks": len(chunks),
        "errors": errors,
    }
    return db, stats