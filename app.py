import io
import os
import tempfile

import PyPDF2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.assistant import HealthAssistant
from src.config import DATA_DIR, OLLAMA_MODEL
from src.emergency import assess_emergency
from src.ollama_client import is_ollama_running, model_available
from src.processor import process_pdf_to_vectorstore
from src.stt import transcribe_audio
from src.tts import text_to_speech

st.set_page_config(
    page_title="Jeevan-Sangini AI",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #fffafb; }
    .report-card {
        padding: 1.25rem;
        border-radius: 12px;
        background: white;
        border-left: 8px solid #ff4b6b;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .sos-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background: #fff0f3;
        border: 1px solid #ff4b6b;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis" not in st.session_state:
    st.session_state.analysis = ""
if "pregnancy_week" not in st.session_state:
    st.session_state.pregnancy_week = 0
if "lang_pref" not in st.session_state:
    st.session_state.lang_pref = "नेपाली"
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "offline_voice" not in st.session_state:
    st.session_state.offline_voice = True

# --- Ollama gate ---
if not is_ollama_running():
    st.error("🚨 Ollama चलिरहेको छैन")
    st.markdown(
        f"""
        **अफलाइन AI सुरु गर्न:**
        1. [Ollama](https://ollama.com) इन्स्टल गर्नुहोस्  
        2. टर्मिनल: `ollama serve`  
        3. मोडेल: `ollama pull {OLLAMA_MODEL}`  
        4. जाँच: `python -m src.check_models`  
        5. एप: `streamlit run app.py`
        """
    )
    st.stop()

if "assistant" not in st.session_state:
    st.session_state.assistant = HealthAssistant()

if not model_available(OLLAMA_MODEL):
    st.warning(f"⚠️ मोडेल `{OLLAMA_MODEL}` भेटिएन। `ollama pull {OLLAMA_MODEL}` चलाउनुहोस्।")


@st.cache_resource
def load_vector_db(_force: bool = False):
    return process_pdf_to_vectorstore(DATA_DIR, force_rebuild=_force)


if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_db()

# --- Header ---
st.title("🤰 Jeevan-Sangini AI")
st.caption("अफलाइन मातृत्व स्वास्थ्य साथी · Gemma 4 + Ollama · गोपनीयता-प्रथम")

st.warning(
    "⚠️ यो जानकारीको लागि मात्र हो। निदान होइन। आपतकालमा तुरुन्त स्वास्थ्य चौकी/अस्पताल जानुहोस्।"
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ सेटिङ")
    st.session_state.lang_pref = st.selectbox(
        "भाषा / Language",
        ["नेपाली", "English"],
        index=0 if st.session_state.lang_pref == "नेपाली" else 1,
    )
    week = st.number_input(
        "गर्भावस्था हप्ता (० = थाहा छैन)",
        min_value=0,
        max_value=42,
        value=int(st.session_state.pregnancy_week or 0),
    )
    st.session_state.pregnancy_week = week if week > 0 else None

    st.session_state.voice_enabled = st.checkbox("🔊 आवाज जवाफ", value=st.session_state.voice_enabled)
    st.session_state.offline_voice = st.checkbox(
        "अफलाइन TTS (pyttsx3)",
        value=st.session_state.offline_voice,
        help="बन्द भए नेपालीका लागi अनलाइन gTTS प्रयास गर्छ",
    )

    st.markdown(
        '<div class="sos-box">🆘 <b>SOS</b><br>एम्बुलेन्स: 102<br>प्रहरी: 100</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("📄 PDF रिपोर्ट")
    report_file = st.file_uploader("ल्याब / ultrasound PDF", type=["pdf"])
    if report_file and st.button("विश्लेषण गर्नुहोस्", use_container_width=True):
        with st.spinner("PDF पढिँदैछ..."):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
                text = "\n".join(
                    p.extract_text() or "" for p in reader.pages
                ).strip()
            except Exception as exc:
                st.error(f"PDF पढ्न सकिएन: {exc}")
                text = ""

            if text:
                lang = st.session_state.lang_pref
                st.session_state.analysis = st.session_state.assistant.analyze_report(
                    text, lang=lang
                )
            else:
                st.session_state.analysis = (
                    "PDF बाट पाठ निकाल्न सकिएन। स्क्यान गरिएको PDF भए स्पष्ट कपी प्रयास गर्नुहोस्।"
                )
        st.rerun()

    st.divider()
    st.subheader("🎤 आवाज इनपुट")
    audio_file = st.file_uploader(
        "अडियो (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
    )
    if audio_file and st.button("आवाज बुझ्नुहोस्", use_container_width=True):
        suffix = os.path.splitext(audio_file.name)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.getvalue())
            audio_path = tmp.name
        with st.spinner("अफलाइन STT..."):
            text, err = transcribe_audio(audio_path)
        if err:
            st.warning(err)
        elif text:
            st.session_state.messages.append({"role": "user", "content": text})
            st.success(f"बुझिएको: {text[:120]}...")
            st.rerun()

    if st.button("📚 गाइडलाइन पुन: लोड", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.vector_db = process_pdf_to_vectorstore(DATA_DIR, force_rebuild=True)
        st.success("डाटा पुन: तयार भयो।")
        st.rerun()

    if st.button("🗑️ च्याट मेटाउनुहोस्", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.listdir(DATA_DIR):
        st.info(f"`{DATA_DIR}/` मा PDF राख्नुहोस् (अफलाइन RAG)।")

st.markdown(
    f'<span style="color:#666;font-size:0.85rem">मोडेल: <code>{OLLAMA_MODEL}</code> · स्थानीय Ollama</span>',
    unsafe_allow_html=True,
)

tab_chat, tab_report, tab_track = st.tabs(
    ["💬 परामर्श", "📊 रिपोर्ट", "📅 हप्ता"]
)

# --- Chat tab ---
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("audio"):
                st.audio(msg["audio"])

    prompt = st.chat_input("प्रश्न लेख्नुहोस्... (नेपाली वा English)")

    if prompt:
        risk = assess_emergency(prompt)
        if risk["level"] == "high":
            st.error(risk["message_ne"] if st.session_state.lang_pref == "नेपाली" else risk["message_en"])
        elif risk["level"] == "medium":
            st.warning(risk["message_ne"] if st.session_state.lang_pref == "नेपाली" else risk["message_en"])

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = ""
        if st.session_state.vector_db:
            try:
                docs = st.session_state.vector_db.similarity_search(prompt, k=3)
                context = "\n".join(d.page_content for d in docs)
            except Exception:
                context = ""

        with st.chat_message("assistant"):
            with st.spinner("विश्लेषण हुँदैछ (स्थानीय AI)..."):
                ans = st.session_state.assistant.ask(
                    prompt,
                    context=context,
                    lang=st.session_state.lang_pref,
                    pregnancy_week=st.session_state.pregnancy_week,
                )
            st.markdown(ans)

            audio_path = None
            if st.session_state.voice_enabled:
                lang_code = "ne" if st.session_state.lang_pref == "नेपाली" else "en"
                audio_path = text_to_speech(
                    ans,
                    lang=lang_code,
                    prefer_offline=st.session_state.offline_voice,
                )
                if audio_path:
                    st.audio(audio_path)

        st.session_state.messages.append(
            {"role": "assistant", "content": ans, "audio": audio_path}
        )

# --- Report tab ---
with tab_report:
    if st.session_state.analysis:
        st.markdown(
            f"<div class='report-card'>{st.session_state.analysis}</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.voice_enabled:
            ap = text_to_speech(
                st.session_state.analysis[:400],
                lang="ne" if st.session_state.lang_pref == "नेपाली" else "en",
                prefer_offline=st.session_state.offline_voice,
            )
            if ap:
                st.audio(ap)
    else:
        st.info("साइडबारबाट PDF अपलोड गरी विश्लेषण गर्नुहोस्।")

# --- Week tracker tab ---
with tab_track:
    w = st.session_state.pregnancy_week
    if w:
        trimester = "१" if w <= 13 else ("२" if w <= 27 else "३")
        st.subheader(f"हप्ता {w} · त्राइमेस्टर {trimester}")
        tips = {
            "नेपाली": (
                f"· नियमित ANC चेकअप\n"
                f"· दाल, भात, साग, अण्डा, दही\n"
                f"· धेरै पानी\n"
                f"· आराम र हल्का हिँडाइ\n"
                f"· घोर दुखाइ, धेरै रगत, चक्कर भए तुरुन्त स्वास्थ्यकर्मी"
            ),
            "English": (
                f"· Regular ANC visits\n"
                f"· Dal, rice, greens, eggs, curd\n"
                f"· Plenty of water\n"
                f"· Rest and light walking\n"
                f"· Seek care for severe pain, bleeding, or dizziness"
            ),
        }
        st.markdown(tips.get(st.session_state.lang_pref, tips["नेपाली"]))
        if st.button("यो हप्ताको बारेमा सोध्नुहोस्"):
            q = f"म {w} हप्ताकी गर्भवती हुँ। के ध्यान दिनुपर्छ?"
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
    else:
        st.info("साइडबारमा गर्भावस्था हप्ता राख्नुहोस्।")

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini · Offline-first · Gemma 4 + Ollama")
