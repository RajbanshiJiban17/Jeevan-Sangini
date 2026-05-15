"""Thin Ollama HTTP client (offline local inference)."""

from __future__ import annotations

import json
import ssl
from typing import Generator, Optional
from urllib import error, request

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

# Ngrok को ब्राउजर वार्निङ हटाउन यो Headers अनिवार्य छ
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "JeevanSangini-Bot",
    "Content-Type": "application/json"
}

class OllamaError(RuntimeError):
    pass

def _post(path: str, payload: dict, timeout: int = OLLAMA_TIMEOUT) -> dict:
    # OLLAMA_BASE_URL (Ngrok URL) प्रयोग गर्ने
    url = f"{OLLAMA_BASE_URL.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    
    req = request.Request(
        url,
        data=data,
        headers=NGROK_HEADERS,
        method="POST",
    )
    try:
        # SSL Verification लाई बेवास्ता गर्न (Ngrok को लागि कहिलेकाहीँ चाहिन्छ)
        context = ssl._create_unverified_context()
        with request.urlopen(req, timeout=timeout, context=context) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except error.URLError as exc:
        raise OllamaError(
            f"Ollama सर्भर जोडिएन ({OLLAMA_BASE_URL}). "
            f"विवरण: {exc}"
        ) from exc

def is_ollama_running(base_url=None) -> bool:
    target_url = (base_url or OLLAMA_BASE_URL).rstrip('/')
    try:
        req = request.Request(f"{target_url}/api/tags", headers=NGROK_HEADERS, method="GET")
        context = ssl._create_unverified_context()
        with request.urlopen(req, timeout=10, context=context) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"DEBUG Check Failed: {e}")
        return False

def list_models(base_url=None) -> list[str]:
    target_url = (base_url or OLLAMA_BASE_URL).rstrip('/')
    try:
        req = request.Request(f"{target_url}/api/tags", headers=NGROK_HEADERS, method="GET")
        context = ssl._create_unverified_context()
        with request.urlopen(req, timeout=10, context=context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []

def model_available(model: str, base_url=None) -> bool:
    names = list_models(base_url=base_url)
    if not names:
        return False
    base = model.split(":")[0]
    return any(n == model or n.startswith(f"{base}:") or n.startswith(base) for n in names)

def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.4,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    model = model or OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature},
    }

    if not stream:
        out = _post("/api/chat", payload)
        msg = out.get("message") or {}
        text = (msg.get("content") or "").strip()
        if not text:
            raise OllamaError("मोडेलले खाली जवाफ दियो।")
        return text

    # Streaming को लागि
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    data = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers=NGROK_HEADERS,
        method="POST",
    )

    def _gen() -> Generator[str, None, None]:
        try:
            context = ssl._create_unverified_context()
            with request.urlopen(req, timeout=OLLAMA_TIMEOUT, context=context) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    part = (chunk.get("message") or {}).get("content") or ""
                    if part:
                        yield part
                    if chunk.get("done"):
                        break
        except Exception as exc:
            raise OllamaError(f"Ollama stream विफल: {exc}")

    return _gen()