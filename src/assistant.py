from __future__ import annotations

import re

from src.config import GEMINI_MODEL, OLLAMA_MODEL
from src.emergency import assess_emergency
from src.gemini_client import GeminiError
from src.llm import chat
from src.ollama_client import OllamaError

SYSTEM_PROMPT = """You are Jeevan Sangini (जीवन सङ्गिनी), a maternal healthcare assistant for Nepal.
Support Nepali and English. Never diagnose with certainty. Encourage doctor visits when needed.
Keep answers short and kind. Use local foods (dal bhat, saag, eggs, milk).
State severity: Safe | Monitor Carefully | Urgent Medical Attention
State risk: Low Risk | Medium Risk | High Risk Emergency"""

REPORT_PROMPT = """Analyze maternal health report text simply:
1) Key values (Hb, BP, sugar)
2) Mild concerns in plain language
3) Next steps
4) Severity level
No definitive diagnosis."""


def _detect_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text or ""):
        return "नेपाली"
    return "English"


class HealthAssistant:
    def __init__(self, backend: str = "ollama", model: str | None = None):
        self.backend = backend
        self.model = model or (
            OLLAMA_MODEL if backend == "ollama" else GEMINI_MODEL
        )

    def ask(
        self,
        user_query: str,
        context: str = "",
        lang: str | None = None,
        pregnancy_week: int | None = None,
    ) -> str:
        lang = lang or _detect_language(user_query)
        emergency = assess_emergency(user_query)

        user_block = user_query.strip()
        if context:
            user_block = f"Guidelines:\n{context[:2000]}\n\nQuestion: {user_query}"
        if pregnancy_week:
            user_block = f"Pregnancy week: {pregnancy_week}\n\n{user_block}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Reply in {lang}. Short sentences.\n\n{user_block}",
            },
        ]

        try:
            answer = chat(messages, backend=self.backend, model=self.model)
        except (OllamaError, GeminiError, RuntimeError) as exc:
            return str(exc)

        if emergency["level"] == "high":
            p = emergency["message_ne"] if lang == "नेपाली" else emergency["message_en"]
            answer = f"{p}\n\n---\n\n{answer}"
        elif emergency["level"] == "medium":
            p = emergency["message_ne"] if lang == "नेपाली" else emergency["message_en"]
            answer = f"{p}\n\n{answer}"
        return answer

    def analyze_report(self, report_text: str, lang: str | None = None) -> str:
        lang = lang or _detect_language(report_text)
        text = (report_text or "").strip()
        if not text:
            return "रिपोर्टबाट पाठ पढ्न सकिएन।"

        emergency = assess_emergency(text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{REPORT_PROMPT}\nLanguage: {lang}\n\n{text[:6000]}",
            },
        ]
        try:
            answer = chat(
                messages, backend=self.backend, model=self.model, temperature=0.25
            )
        except (OllamaError, GeminiError, RuntimeError) as exc:
            return str(exc)

        if emergency["level"] in ("high", "medium"):
            note = emergency["message_ne"] if lang == "नेपाली" else emergency["message_en"]
            answer = f"{note}\n\n{answer}"
        return answer
