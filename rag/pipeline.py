"""The RAG pipeline: guardrails -> retrieve -> build prompt -> stream answer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from . import config
from .guardrails import check_query
from .llm import BaseLLM
from .prompts import NOT_FOUND, build_messages, format_context


@dataclass
class Source:
    file: str
    location: str
    score: float  # cosine similarity, 0..1
    text: str

    @property
    def snippet(self) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= 300 else t[:300] + "…"


@dataclass
class RAGResponse:
    sources: list[Source] = field(default_factory=list)
    text: str | None = None              # ready-made reply (guardrail / not found)
    stream: Iterator[str] | None = None  # streamed LLM answer


class RAGPipeline:
    def __init__(self, vector_db, llm: BaseLLM):
        self.vector_db = vector_db
        self.llm = llm

    def retrieve(self, query: str, k: int) -> list[Source]:
        results = self.vector_db.similarity_search_with_score(query, k=k)
        sources = []
        for doc, distance in results:
            # FAISS returns squared L2 distance; for unit vectors cosine = 1 - d/2.
            similarity = max(0.0, 1.0 - float(distance) / 2.0)
            if similarity < config.MIN_SIMILARITY:
                continue
            sources.append(
                Source(
                    file=doc.metadata.get("file", "Unknown"),
                    location=doc.metadata.get("location", "N/A"),
                    score=similarity,
                    text=doc.page_content,
                )
            )
        return sources

    def _safe_stream(self, messages, temperature) -> Iterator[str]:
        try:
            yield from self.llm.stream(messages, temperature, config.MAX_NEW_TOKENS)
        except Exception as exc:  # noqa: BLE001 - show the problem in the chat
            yield f"\n\n⚠️ The language model failed: {exc}"

    def ask(self, query: str, top_k: int = config.TOP_K, temperature: float = config.TEMPERATURE) -> RAGResponse:
        canned = check_query(query)
        if canned:
            return RAGResponse(text=canned)

        sources = self.retrieve(query, top_k)
        if not sources:
            return RAGResponse(text=NOT_FOUND)

        messages = build_messages(query, format_context(sources))
        return RAGResponse(sources=sources, stream=self._safe_stream(messages, temperature))
