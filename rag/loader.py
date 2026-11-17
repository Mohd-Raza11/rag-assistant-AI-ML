"""Load PDF / DOCX / PPTX / TXT files into LangChain Document objects."""
from __future__ import annotations

import logging
from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def clean_text(text: str) -> str:
    """Remove characters that crash the tokenizer (lone surrogates, null/control bytes)."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return _CONTROL_CHARS.sub(" ", text)

def list_documents(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _doc(text: str, path: Path, location: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": str(path), "file": path.name, "location": location},
    )


def _load_pdf(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    for d in docs:
        d.metadata["file"] = path.name
        d.metadata["location"] = f"Page {int(d.metadata.get('page', 0)) + 1}"
    return docs


def _load_txt(path: Path) -> list[Document]:
    return [_doc(clean_text(path.read_text(encoding="utf-8", errors="ignore")), path, "Whole file")]


def _load_docx(path: Path) -> list[Document]:
    from docx import Document as DocxDocument  # python-docx

    d = DocxDocument(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [_doc(clean_text("\n".join(parts)), path, "Whole document")]


def _load_pptx(path: Path) -> list[Document]:
    from pptx import Presentation  # python-pptx

    docs = []
    for i, slide in enumerate(Presentation(str(path)).slides, start=1):
        texts = [
            shape.text_frame.text.strip()
            for shape in slide.shapes
            if shape.has_text_frame and shape.text_frame.text.strip()
        ]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(notes)
        if texts:
            docs.append(_doc("\n".join(texts), path, f"Slide {i}"))
    return docs


_LOADERS = {".pdf": _load_pdf, ".txt": _load_txt, ".docx": _load_docx, ".pptx": _load_pptx}


def load_documents(folder: Path) -> tuple[list[Document], list[str]]:
    """Return (documents, error_messages). One bad file never stops the others."""
    documents: list[Document] = []
    errors: list[str] = []
    for path in list_documents(folder):
        try:
            loaded = _LOADERS[path.suffix.lower()](path)
            for d in loaded:
                d.page_content = clean_text(d.page_content)
                if d.page_content.strip():
                    documents.append(d)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.warning("Could not load %s: %s", path.name, exc)
            errors.append(f"{path.name}: {exc}")
    return documents, errors
