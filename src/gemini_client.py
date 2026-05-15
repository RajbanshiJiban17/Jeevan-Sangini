"""Google Gemini for Streamlit Cloud (no Ollama on server)."""

from __future__ import annotations

import time

from src.runtime import get_gemini_api_key


class GeminiError(RuntimeError):
    pass


def chat_gemini(
    messages: list[dict],
    model: str = "gemini-1.5-flash",
    temperature: float = 0.35,
) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        raise GeminiError(
            "Gemini API key छैन। Streamlit → Settings → Secrets मा GOOGLE_API_KEY राख्नुहोस्।"
        )

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise GeminiError("pip install google-generativeai") from exc

    genai.configure(api_key=api_key)

    system_parts: list[str] = []
    user_parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)

    prompt = "\n\n".join(user_parts) or ""
    system = "\n".join(system_parts)
    if system:
        prompt = f"{system}\n\n{prompt}"

    models_to_try = [model, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro"]
    seen: set[str] = set()
    last_err: Exception | None = None

    for mid in models_to_try:
        if mid in seen:
            continue
        seen.add(mid)
        try:
            time.sleep(0.25)
            gmodel = genai.GenerativeModel(mid)
            resp = gmodel.generate_content(prompt)
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
        except Exception as exc:
            last_err = exc

    raise GeminiError(f"Gemini error: {last_err}")
