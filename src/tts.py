"""Text-to-speech: offline (pyttsx3) with optional online Nepali (gTTS)."""

from __future__ import annotations

import os
import tempfile


def text_to_speech(text: str, lang: str = "ne", prefer_offline: bool = True) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    text = text[:500]

    if prefer_offline:
        path = _offline_tts(text)
        if path:
            return path

    if lang.startswith("ne"):
        return _online_gtts(text, lang="ne")
    return _offline_tts(text) or _online_gtts(text, lang="en")


def _offline_tts(text: str) -> str | None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        out.close()
        engine.save_to_file(text, out.name)
        engine.runAndWait()
        if os.path.isfile(out.name) and os.path.getsize(out.name) > 0:
            return out.name
    except Exception as e:
        print("Offline TTS:", e)
    return None


def _online_gtts(text: str, lang: str = "ne") -> str | None:
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang=lang)
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(f.name)
        return f.name
    except Exception as e:
        print("gTTS:", e)
        return None
