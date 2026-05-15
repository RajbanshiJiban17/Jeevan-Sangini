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


def resolve_backend(ollama_ok: bool) -> str:
    forced = (os.getenv("LLM_BACKEND") or "auto").lower().strip()
    if forced == "ollama":
        return "ollama" if ollama_ok else "none"
    if forced == "gemini":
        return "gemini" if get_gemini_api_key() else "none"
    if ollama_ok:
        return "ollama"
    if get_gemini_api_key():
        return "gemini"
    return "none"


def rag_enabled() -> bool:
    if os.getenv("ENABLE_RAG", "").lower() in ("0", "false", "no"):
        return False
    if is_streamlit_cloud() and os.getenv("ENABLE_RAG", "").lower() not in ("1", "true", "yes"):
        return False
    return True
