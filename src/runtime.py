"""Detect local vs Streamlit Cloud and pick LLM backend."""

from __future__ import annotations

import os


def is_streamlit_cloud() -> bool:
    if os.getenv("STREAMLIT_RUNTIME_ENV"):
        return True
    host = (os.getenv("HOSTNAME") or "") + (os.getenv("STREAMLIT_SERVER_ADDRESS") or "")
    return "streamlit.app" in host.lower()


def get_gemini_api_key() -> str | None:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip() or None
    except Exception:
        pass
    return None


def resolve_backend(ollama_ok: bool, user_choice: str | None = None) -> str:
    """
    user_choice: 'ollama' | 'gemini' | None (auto)
    Local laptop: auto prefers Ollama when running (offline-first).
    """
    if user_choice == "ollama":
        return "ollama" if ollama_ok else "none"
    if user_choice == "gemini":
        return "gemini" if get_gemini_api_key() else "none"

    forced = (os.getenv("LLM_BACKEND") or "auto").lower().strip()
    if forced == "ollama":
        return "ollama" if ollama_ok else "none"
    if forced == "gemini":
        return "gemini" if get_gemini_api_key() else "none"

    # auto: offline-first on local machine
    if ollama_ok:
        return "ollama"
    if get_gemini_api_key():
        return "gemini"
    return "none"


def rag_enabled() -> bool:
    """RAG is opt-in only (avoids torch/langchain on every app start)."""
    return os.getenv("ENABLE_RAG", "").lower() in ("1", "true", "yes")
