"""Rule-based emergency screening (runs offline, before LLM)."""

from __future__ import annotations

HIGH_RISK_TERMS = [
    "heavy bleeding",
    "severe bleeding",
    "lots of blood",
    "धेरै रगत",
    "भारी रक्तस्राव",
    "रक्तस्राव",
    "severe abdominal pain",
    "घोर पेट दुखाइ",
    "धेरै पेट दुख्यो",
    "blurred vision",
    "धुंधला दृष्टि",
    "अन्धो जस्तो",
    "chest pain",
    "छाती दुखाइ",
    "breathing difficulty",
    "can't breathe",
    "श्वास फराकिलो",
    "सास फेर्न गाह्रो",
    "seizure",
    "झट्का",
    "unconscious",
    "बेहोस",
    "सचेत छैन",
    "severe swelling",
    "धेरै सुन्निएको",
    "अचानक सुन्निएको",
    "severe headache",
    "धेरै टाउको दुखाइ",
    "घोर टाउको दुखाइ",
    "reduced baby movement",
    "baby not moving",
    "बच्चा चल्दैन",
    "बच्चा कम चल्यो",
    "high fever",
    "धेरै ज्वरो",
    "104",
    "105",
    "severe dizziness",
    "धेरै चक्कर",
    "loss of consciousness",
    "preeclampsia",
    "प्रिक्लेम्प्सिया",
]

MEDIUM_RISK_TERMS = [
    "moderate bleeding",
    "spotting a lot",
    "persistent headache",
    "लगातार टाउको दुखाइ",
    "swelling",
    "सुन्निएको",
    "dizziness",
    "चक्कर",
    "fever",
    "ज्वरो",
    "vomiting a lot",
    "धेरै बान्ता",
    "pain",
    "दुखाइ",
]


def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def assess_emergency(text: str) -> dict:
    """
    Returns:
        level: 'low' | 'medium' | 'high'
        matched: list of matched concern labels
        message_ne: Nepali guidance snippet
        message_en: English guidance snippet
    """
    t = _normalize(text)
    if not t:
        return {"level": "low", "matched": [], "message_ne": "", "message_en": ""}

    high = [term for term in HIGH_RISK_TERMS if term.lower() in t]
    if high:
        return {
            "level": "high",
            "matched": high[:5],
            "message_ne": (
                "🚨 उच्च जोखिम: तुरुन्त नजिकको स्वास्थ्य चौकी वा अस्पताल जानुहोस्। "
                "सम्भव भए एम्बुलेन्स (१०२) वा परिवारलाई बोलाउनुहोस्।"
            ),
            "message_en": (
                "🚨 High risk: seek immediate care at the nearest health post or hospital. "
                "Call ambulance (102) or family if possible."
            ),
        }

    medium = [term for term in MEDIUM_RISK_TERMS if term.lower() in t]
    if len(medium) >= 2 or any(x in t for x in ("swelling", "सुन्निएको", "headache", "टाउको")):
        return {
            "level": "medium",
            "matched": medium[:5],
            "message_ne": (
                "⚠️ मध्यम जोखिम: आजै स्वास्थ्यकर्मीसँग सम्पर्क गर्नुहोस्। "
                "पानी पिउनुहोस्, आराम गर्नुहोस्, र लक्षण बढेमा तुरुन्त जानुहोस्।"
            ),
            "message_en": (
                "⚠️ Medium risk: contact a health worker today. "
                "Rest, hydrate, and go immediately if symptoms worsen."
            ),
        }

    return {"level": "low", "matched": [], "message_ne": "", "message_en": ""}
