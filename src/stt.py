"""Offline speech-to-text (optional faster-whisper)."""

from __future__ import annotations


def transcribe_audio(file_path: str) -> tuple[str | None, str | None]:
    """
    Returns (text, error_message).
    On success error_message is None.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, (
            "अफलाइन आवाजका लागि: pip install faster-whisper "
            "(पहिलो पटक मोडेल डाउनलोड हुन्छ, पछि अफलाइन चल्छ)।"
        )

    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(file_path, language=None)
        text = " ".join(s.text.strip() for s in segments if s.text).strip()
        if not text:
            return None, "आवाजबाट पाठ बनाउन सकिएन। फेरि रेकर्ड गर्नुहोस्।"
        return text, None
    except Exception as exc:
        return None, f"आवाज पहिचान विफल: {exc}"
