from __future__ import annotations

import re

from src.config import OLLAMA_MODEL
from src.emergency import assess_emergency
from src.ollama_client import OllamaError, chat

SYSTEM_PROMPT = """You are Jeevan Sangini (जीवन सङ्गिनी), an advanced maternal healthcare assistant.
You run locally (offline) for privacy. You support Nepali and English.

CORE RULES:
- Never diagnose with certainty. Never replace licensed doctors.
- Prioritize maternal safety. Be calm, empathetic, and simple.
- Use Nepali if the user writes in Nepali; otherwise English.
- No panic unless truly urgent. Encourage professional care when needed.
- Use affordable Nepal-friendly nutrition examples (dal bhat, saag, eggs, milk, gundruk).
- If unsure, say information may be incomplete and suggest clinical evaluation.

SYMPTOM SEVERITY (state one): Safe | Monitor Carefully | Urgent Medical Attention
EMERGENCY RISK (state one): Low Risk | Medium Risk | High Risk Emergency

For reports: explain Hb, BP, sugar, etc. in plain language; flag abnormal values carefully.
Never give unsafe medical advice."""

REPORT_PROMPT = """Analyze this maternal health document text.
1) Key values (Hb, BP, glucose, etc.)
2) What is slightly low/high vs typical pregnancy ranges (cautious language)
3) Simple next steps and when to see a doctor
4) Severity: Safe | Monitor Carefully | Urgent Medical Attention
Do not claim a definitive diagnosis."""


def _detect_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text or ""):
        return "नेपाली"
    return "English"


class HealthAssistant:
    def __init__(self, model: str | None = None):
        self.model = model or OLLAMA_MODEL

    def ask(
        self,
        user_query: str,
        context: str = "",
        lang: str | None = None,
        pregnancy_week: int | None = None,
        extra_system: str = "",
    ) -> str:
        lang = lang or _detect_language(user_query)
        emergency = assess_emergency(user_query)

        user_block = user_query.strip()
        if context:
            user_block = f"Relevant guideline context:\n{context[:2000]}\n\nUser: {user_query}"

        if pregnancy_week:
            user_block = f"Pregnancy week: {pregnancy_week}\n\n{user_block}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + (f"\n{extra_system}" if extra_system else "")},
            {
                "role": "user",
                "content": (
                    f"Respond in {lang}. Keep sentences short for low-literacy users.\n\n"
                    f"{user_block}"
                ),
            },
        ]

        try:
            answer = chat(messages, model=self.model, temperature=0.35)
        except OllamaError as exc:
            return str(exc)

        if emergency["level"] == "high":
            prefix = emergency["message_ne"] if lang == "नेपाली" else emergency["message_en"]
            answer = f"{prefix}\n\n---\n\n{answer}"
        elif emergency["level"] == "medium":
            prefix = emergency["message_ne"] if lang == "नेपाली" else emergency["message_en"]
            answer = f"{prefix}\n\n{answer}"

        return answer

    def analyze_report(self, report_text: str, lang: str | None = None) -> str:
        lang = lang or _detect_language(report_text)
        text = (report_text or "").strip()
        if not text:
            return "रिपोर्टबाट पाठ पढ्न सकिएन। स्पष्ट PDF वा फोटो पुन: अपलोड गर्नुहोस्।"

        emergency = assess_emergency(text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{REPORT_PROMPT}\nRespond in {lang}.\n\nReport text:\n{text[:6000]}"
                ),
            },
        ]
        try:
            answer = chat(messages, model=self.model, temperature=0.25)
        except OllamaError as exc:
            return str(exc)

        if emergency["level"] in ("high", "medium"):
            note = emergency["message_ne"] if lang == "नेपाली" else emergency["message_en"]
            answer = f"{note}\n\n{answer}"
        return answer
