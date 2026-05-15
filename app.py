"""Jeevan-Sangini — hybrid: Ollama (local CPU) or Gemini (Streamlit Cloud)."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

# Ensure imports work on Streamlit Cloud
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import PyPDF2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.assistant import HealthAssistant
from src.config import DATA_DIR, GEMINI_MODEL, OLLAMA_MODEL
from src.emergency import assess_emergency
from src.ollama_client import is_ollama_running, model_available
from src.processor import process_pdf_to_vectorstore
from src.runtime import is_streamlit_cloud, rag_enabled, resolve_backend
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
        padding: 1.25rem; border-radius: 12px; background: white;
        border-left: 8px solid #ff4b6b;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .sos-box {
        padding: 0.75rem 1rem; border-radius: 8px;
        background: #fff0f3; border: 1px solid #ff4b6b;
    }
    .mode-pill {
        display: inline-block; padding: 0.35rem 0.75rem;
        border-radius: 999px; font-size: 0.85rem; margin-bottom: 0.5rem;
    }
    .mode-local { background: #e8f5e9; color: #2e7d32; }
    .mode-cloud { background: #e3f2fd; color: #1565c0; }
  </style>
    """,
    unsafe_allow_html=True,
)

# --- Session ---
for key, default in [
    ("messages", []),
    ("analysis", ""),
    ("pregnancy_week", None),
    ("lang_pref", "नेपाली"),
    ("voice_enabled", True),
    ("offline_voice", not is_streamlit_cloud()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Backend ---
_ollama_up = is_ollama_running()
BACKEND = resolve_backend(_ollama_up)
on_cloud = is_streamlit_cloud()

if BACKEND == "none":
    st.error("🚨 AI backend उपलब्ध छैन")
    if on_cloud:
        st.markdown(
            """
            **Streamlit Cloud मा:**
            1. [share.streamlit.io](https://share.streamlit.io) → तपाईंको app → **Settings → Secrets**
            2. यो थप्नुहोस्:
            ```toml
            GOOGLE_API_KEY = "तपाईंको-gemini-key"
            ```
            3. **Save** → **Reboot app**

            API key: [Google AI Studio](https://aistudio.google.com/apikey)
            """
        )
    else:
        st.markdown(
            f"""
            **ल्यापटप (CPU, CUDA छैन):**
            1. [Ollama](https://ollama.com) install
            2. `ollama serve`
            3. सानो मोडेल (CPU): `ollama pull {OLLAMA_MODEL}`
            4. `streamlit run app.py`

            वा `.env` मा `GOOGLE_API_KEY=...` राख्नुहोस् (cloud जस्तै)।
            """
        )
    st.stop()

if "assistant" not in st.session_state:
    st.session_state.assistant = HealthAssistant(backend=BACKEND)

# --- RAG (lazy; off on cloud by default) ---
@st.cache_resource(show_spinner=False)
def load_vector_db():
    if not rag_enabled():
        return None
    try:
        return process_pdf_to_vectorstore(DATA_DIR)
    except Exception as exc:
        st.warning(f"RAG load skipped: {exc}")
        return None


if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_db()

# --- Header ---
st.title("🤰 Jeevan-Sangini AI")
mode_class = "mode-local" if BACKEND == "ollama" else "mode-cloud"
mode_label = (
    f"🖥️ Local · Ollama · {OLLAMA_MODEL}"
    if BACKEND == "ollama"
    else f"☁️ Cloud · Gemini · {GEMINI_MODEL}"
)
st.markdown(f'<span class="mode-pill {mode_class}">{mode_label}</span>', unsafe_allow_html=True)

if BACKEND == "ollama" and not model_available(OLLAMA_MODEL):
    st.warning(f"⚠️ `ollama pull {OLLAMA_MODEL}` चलाउनुहोस् (CPU-friendly सानो मोडेल)")

if on_cloud:
    st.info(
        "यो लिंक **Gemini cloud** मा चल्छ। पूर्ण **offline Ollama** का लागि ल्यापटपमा "
        "`pip install -r requirements-local.txt` र `streamlit run app.py` गर्नुहोस्।"
    )

st.warning("⚠️ जानकारी मात्र। निदान होइन। आपतकालमा तुरुन्त स्वास्थ्य चौकी/अस्पताल।")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ सेटिङ")
    st.caption(f"Backend: **{BACKEND}**")

    st.session_state.lang_pref = st.selectbox(
        "भाषा",
        ["नेपाली", "English"],
        index=0 if st.session_state.lang_pref == "नेपाली" else 1,
    )
    week = st.number_input("गर्भावस्था हप्ता (०=थाहा छैन)", 0, 42, value=0)
    st.session_state.pregnancy_week = week if week > 0 else None

    st.session_state.voice_enabled = st.checkbox("🔊 आवाज जवाफ", value=st.session_state.voice_enabled)
    if not on_cloud:
        st.session_state.offline_voice = st.checkbox(
            "अफलाइन TTS",
            value=st.session_state.offline_voice,
        )

    st.markdown(
        '<div class="sos-box">🆘 SOS<br>एम्बुलेन्स: 102<br>प्रहरी: 100</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("📄 PDF रिपोर्ट")
    pdf = st.file_uploader("ल्याब PDF", type=["pdf"])
    if pdf and st.button("विश्लेषण", use_container_width=True):
        with st.spinner("PDF..."):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(pdf.read()))
                text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
            except Exception as exc:
                st.error(str(exc))
                text = ""
            st.session_state.analysis = (
                st.session_state.assistant.analyze_report(text, lang=st.session_state.lang_pref)
                if text
                else "PDF बाट पाठ निकाल्न सकिएन।"
            )
        st.rerun()

    if not on_cloud:
        st.divider()
        st.subheader("🎤 आवाज")
        audio = st.file_uploader("अडियो", type=["wav", "mp3", "m4a"])
        if audio and st.button("बुझ्नुहोस्", use_container_width=True):
            suf = os.path.splitext(audio.name)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                tmp.write(audio.getvalue())
                path = tmp.name
            txt, err = transcribe_audio(path)
            if err:
                st.warning(err)
            elif txt:
                st.session_state.messages.append({"role": "user", "content": txt})
                st.rerun()

    if rag_enabled() and st.button("📚 RAG reload", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.vector_db = process_pdf_to_vectorstore(DATA_DIR, force_rebuild=True)
        st.rerun()

    if st.button("🗑️ च्याट मेटाउनु", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

tab_chat, tab_report, tab_track = st.tabs(["💬 परामर्श", "📊 रिपोर्ट", "📅 हप्ता"])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"])

    if prompt := st.chat_input("प्रश्न लेख्नुहोस्..."):
        risk = assess_emergency(prompt)
        if risk["level"] == "high":
            st.error(risk["message_ne"] if st.session_state.lang_pref == "नेपाली" else risk["message_en"])
        elif risk["level"] == "medium":
            st.warning(risk["message_ne"] if st.session_state.lang_pref == "नेपाली" else risk["message_en"])

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ctx = ""
        vdb = st.session_state.vector_db
        if vdb:
            try:
                docs = vdb.similarity_search(prompt, k=2)
                ctx = "\n".join(d.page_content for d in docs)
            except Exception:
                pass

        with st.chat_message("assistant"):
            with st.spinner("विश्लेषण..."):
                ans = st.session_state.assistant.ask(
                    prompt,
                    context=ctx,
                    lang=st.session_state.lang_pref,
                    pregnancy_week=st.session_state.pregnancy_week,
                )
            st.markdown(ans)
            ap = None
            if st.session_state.voice_enabled:
                lc = "ne" if st.session_state.lang_pref == "नेपाली" else "en"
                ap = text_to_speech(
                    ans[:400],
                    lang=lc,
                    prefer_offline=st.session_state.offline_voice and not on_cloud,
                )
                if ap:
                    st.audio(ap)
        st.session_state.messages.append({"role": "assistant", "content": ans, "audio": ap})

with tab_report:
    if st.session_state.analysis:
        st.markdown(
            f"<div class='report-card'>{st.session_state.analysis}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("PDF अपलोड गर्नुहोस्।")

with tab_track:
    w = st.session_state.pregnancy_week
    if w:
        tri = "१" if w <= 13 else ("२" if w <= 27 else "३")
        st.subheader(f"हप्ता {w} · त्राइमेस्टर {tri}")
        st.markdown("ANC · दालभात साग · पानी · आराम · गम्भीर लक्षणमा तुरुन्त डाक्टर")
    else:
        st.info("साइडबारमा हप्ता राख्नुहोस्।")

st.caption("© 2026 Jeevan-Sangini")
