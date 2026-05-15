"""Thin Ollama HTTP client (offline local inference)."""

from __future__ import annotations

import json
from typing import Generator, Optional
from urllib import error, request

from src.config import OLLAMA_BASE_URL, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = OLLAMA_TIMEOUT) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except error.URLError as exc:
        raise OllamaError(
            f"Ollama सर्भर जोडिएन ({OLLAMA_HOST}). "
            f"पहिले `ollama serve` चलाउनुहोस् र `ollama pull {OLLAMA_MODEL}` गर्नुहोस्। "
            f"विवरण: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama बाट अमान्य जवाफ आयो।") from exc


def is_ollama_running(base_url) -> bool:
    try:
        req = request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        resp = req.get(f"{base_url}/api/tags", headers={"ngrok-skip-browser-warning": "true"})
        with request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_models(base_url) -> list[str]:
    try:
        req = request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        headers = {"ngrok-skip-browser-warning": "true"}
        resp = req.get(f"{base_url}/api/tags", headers=headers)
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def model_available(model: str) -> bool:
    names = list_models()
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

    url = f"{OLLAMA_HOST}/api/chat"
    data = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _gen() -> Generator[str, None, None]:
        try:
            with request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
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
        except error.URLError as exc:
            raise OllamaError(f"Ollama stream विफल: {exc}") from exc

    return _gen()
