"""Unified LLM: Ollama (local CPU) or Gemini (cloud deploy)."""

from __future__ import annotations

from src.config import GEMINI_MODEL, OLLAMA_MODEL
from src.gemini_client import GeminiError, chat_gemini
from src.ollama_client import OllamaError, chat as chat_ollama

Backend = str  # "ollama" | "gemini"


def chat(
    messages: list[dict],
    backend: Backend,
    model: str | None = None,
    temperature: float = 0.35,
) -> str:
    if backend == "ollama":
        try:
            return chat_ollama(
                messages,
                model=model or OLLAMA_MODEL,
                temperature=temperature,
            )
        except OllamaError as exc:
            raise exc
    if backend == "gemini":
        try:
            return chat_gemini(
                messages,
                model=model or GEMINI_MODEL,
                temperature=temperature,
            )
        except GeminiError as exc:
            raise exc
    raise RuntimeError("कुनै AI backend उपलब्ध छैन।")
