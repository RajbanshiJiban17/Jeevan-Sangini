from gtts import gTTS
import tempfile


def text_to_speech(text, lang="ne"):

    try:
        if not text:
            return None

        # ✅ Clean text (Gemma output safety)
        clean_text = text.strip().replace("\n", " ")

        # ✅ limit (TTS crash रोक्न)
        clean_text = clean_text[:600]

        # language fallback safety
        if lang not in ["ne", "en"]:
            lang = "ne"

        tts = gTTS(text=clean_text, lang=lang)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

        tts.save(temp_file.name)

        return temp_file.name

    except Exception as e:
        print("TTS Error:", e)
        return None