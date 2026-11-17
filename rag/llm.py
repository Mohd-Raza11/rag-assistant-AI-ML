"""Free LLM through the Groq API (OpenAI-compatible). llm.stream(messages) yields text pieces."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Iterator

import requests

from . import config


class BaseLLM(ABC):
    name: str

    @abstractmethod
    def stream(self, messages: list[dict], temperature: float, max_new_tokens: int) -> Iterator[str]:
        """Yield the answer piece by piece."""

    @abstractmethod
    def check(self) -> tuple[bool, str]:
        """Return (is_ready, human_readable_status)."""


class GroqLLM(BaseLLM):
    """Calls Groq's free hosted Llama models. Nothing is downloaded to your computer."""

    def __init__(self, model: str, api_key: str, base_url: str = config.GROQ_BASE_URL):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.name = f"groq:{model}"

    def check(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "No API key found. Add GROQ_API_KEY to the .env file and restart the app."
        return True, f"Groq is ready ({self.model})."

    @staticmethod
    def _explain_error(r: requests.Response) -> str:
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except ValueError:
            detail = r.text
        hints = {
            401: "Invalid API key. Check GROQ_API_KEY in your .env file.",
            404: "Model not found. Check the model name at console.groq.com/docs/models.",
            429: "Free-tier rate limit reached. Wait a minute and try again.",
        }
        return f"{hints.get(r.status_code, 'Request failed.')} (HTTP {r.status_code}: {detail})"

    def stream(self, messages, temperature, max_new_tokens):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        with requests.post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload, stream=True, timeout=60
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(self._explain_error(r))
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                if not choices:
                    continue
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    yield piece


def get_llm(backend: str, model: str | None = None) -> BaseLLM:
    if backend == "groq":
        return GroqLLM(model or config.GROQ_MODEL, config.GROQ_API_KEY)
    raise ValueError(f"Unknown LLM backend: {backend}")