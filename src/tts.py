from gtts import gTTS
import tempfile

def text_to_speech(text, lang="ne"):

    try:
        text = text[:600]

        tts = gTTS(text=text, lang=lang)

        file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(file.name)

        return file.name

    except Exception as e:
        print("TTS Error:", e)
        return None