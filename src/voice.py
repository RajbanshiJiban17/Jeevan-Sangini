from gtts import gTTS
import tempfile


def text_to_speech(text, lang="ne"):

    tts = gTTS(text=text, lang=lang)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    tts.save(temp.name)

    return temp.name