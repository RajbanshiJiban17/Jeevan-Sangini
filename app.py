"""Jeevan-Sangini — hybrid: Ollama (local) or Gemini (cloud)."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import PyPDF2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.assistant import HealthAssistant
from src.config import DATA_DIR, GEMINI_MODEL, OLLAMA_MODEL,OLLAMA_BASE_URL
from src.emergency import assess_emergency
from src.ollama_client import is_ollama_running, list_models, model_available
from src.runtime import get_gemini_api_key, is_streamlit_cloud, rag_enabled, resolve_backend
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
    .stApp {
        background: linear-gradient(165deg, #fff5f7 0%, #fce4ec 45%, #f8bbd0 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4a1942 0%, #6d1b4d 100%) !important;
        border-right: 3px solid #ff4b6b;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #ffe4ec !important;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px dashed #ff4b6b !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: #374151 !important;
    }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1100px;
    }
    .report-card {
        padding: 1.25rem;
        border-radius: 12px;
        background: #ffffff;
        color: #1f2937;
        border-left: 8px solid #ff4b6b;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .sos-box {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        background: #ffebee;
        color: #b71c1c !important;
        border: 2px solid #ef5350;
        font-weight: 600;
    }
    .status-box {
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    .status-ok { background: #e8f5e9; color: #1b5e20; border: 1px solid #66bb6a; }
    .status-bad { background: #ffebee; color: #b71c1c; border: 1px solid #ef5350; }
    .mode-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .mode-local { background: #2e7d32; color: #fff; }
    .mode-cloud { background: #1565c0; color: #fff; }
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.85);
        border-radius: 12px;
        padding: 0.5rem;
    }
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
    ("vector_db", None),
    ("rag_loaded", False),
    ("ai_mode", "auto"),
    ("active_backend", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

_ollama_up = is_ollama_running(OLLAMA_BASE_URL)
_ollama_models = list_models() if _ollama_up else []
_has_gemini = bool(get_gemini_api_key())
on_cloud = is_streamlit_cloud()

# --- Sidebar: AI mode (first) ---
with st.sidebar:
    st.header("⚙️ सेटिङ")

    if _ollama_up:
        st.markdown(
            '<div class="status-box status-ok">🟢 Ollama चलिरहेको छ (Offline)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Models: " + ", ".join(_ollama_models[:3]) or "—")
    else:
        st.markdown(
            '<div class="status-box status-bad">🔴 Ollama बन्द छ<br>'
            "<small>Terminal: <b>ollama serve</b></small></div>",
            unsafe_allow_html=True,
        )

    mode_options = ["auto (सिफारिस)"]
    mode_map = {"auto (सिफारिस)": None}
    if _ollama_up:
        mode_options.append("अफलाइन Ollama")
        mode_map["अफलाइन Ollama"] = "ollama"
    if _has_gemini:
        mode_options.append("अनलाइन Gemini")
        mode_map["अनलाइन Gemini"] = "gemini"

    choice_label = st.radio("AI मोड", mode_options, index=0)
    user_backend_choice = mode_map.get(choice_label)

    if not _ollama_up and user_backend_choice != "gemini":
        st.warning("Offline को लागि अर्को terminal मा: `ollama serve`")

    st.divider()

BACKEND = resolve_backend(_ollama_up, user_backend_choice)

if BACKEND == "none":
    st.error("🚨 AI backend उपलब्ध छैन")
    st.markdown(
        """
        **Offline:** `ollama serve` → `ollama pull gemma2:2b`  
        **Online:** `.env` मा `GOOGLE_API_KEY=...`  
        **वा** sidebar मा Gemini मोड छान्नुहोस्।
        """
    )
    st.stop()

# Assistant (recreate if backend changed)
if (
    "assistant" not in st.session_state
    or st.session_state.active_backend != BACKEND
):
    st.session_state.assistant = HealthAssistant(backend=BACKEND)
    st.session_state.active_backend = BACKEND


@st.cache_resource(show_spinner="गाइडलाइन लोड...")
def load_vector_db():
    from src.processor import process_pdf_to_vectorstore

    return process_pdf_to_vectorstore(DATA_DIR)


# --- Main header ---
st.title("🤰 Jeevan-Sangini AI")
mode_class = "mode-local" if BACKEND == "ollama" else "mode-cloud"
mode_label = (
    f"🖥️ Offline · Ollama · {OLLAMA_MODEL}"
    if BACKEND == "ollama"
    else f"☁️ Online · Gemini · {GEMINI_MODEL}"
)
st.markdown(f'<span class="mode-pill {mode_class}">{mode_label}</span>', unsafe_allow_html=True)

if BACKEND == "ollama" and not model_available(OLLAMA_MODEL):
    st.error(f"मोडेल `{OLLAMA_MODEL}` छैन। Terminal: `ollama pull {OLLAMA_MODEL}`")

if BACKEND == "gemini" and not on_cloud and _ollama_up:
    st.info("💡 Offline चाहनुहुन्छ? Sidebar → **अफलाइन Ollama** छान्नुहोस्।")

st.warning("⚠️ जानकारी मात्र। आपतकालमा तुरुन्त स्वास्थ्य चौकी/अस्पताल।")

# --- Sidebar continued ---
with st.sidebar:
    st.caption(f"Active: **{BACKEND}**")

    st.session_state.lang_pref = st.selectbox(
        "भाषा",
        ["नेपाली", "English"],
        index=0 if st.session_state.lang_pref == "नेपाली" else 1,
    )
    week = st.number_input("गर्भावस्था हप्ता (०=थाहा छैन)", 0, 42, value=0)
    st.session_state.pregnancy_week = week if week > 0 else None

    st.session_state.voice_enabled = st.checkbox("🔊 आवाज जवाफ", value=st.session_state.voice_enabled)
    if not on_cloud:
        st.session_state.offline_voice = st.checkbox("अफलाइन TTS", value=st.session_state.offline_voice)

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
            from src.stt import transcribe_audio

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

    st.divider()
    st.subheader("📚 गाइडलाइन (RAG)")
    if st.button("गाइडलाइन लोड", use_container_width=True):
        with st.spinner("लोड..."):
            try:
                st.session_state.vector_db = load_vector_db()
                st.session_state.rag_loaded = st.session_state.vector_db is not None
                st.success("तयार!" if st.session_state.rag_loaded else f"`{DATA_DIR}/` मा PDF राख्नुहोस्")
            except Exception as exc:
                st.error(str(exc))

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
        if st.session_state.vector_db:
            try:
                docs = st.session_state.vector_db.similarity_search(prompt, k=2)
                ctx = "\n".join(d.page_content for d in docs)
            except Exception:
                pass

        with st.chat_message("assistant"):
            label = "Offline AI..." if BACKEND == "ollama" else "विश्लेषण..."
            with st.spinner(label):
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
        st.markdown("ANC · दालभात साग · पानी · आराम")
    else:
        st.info("साइडबारमा हप्ता राख्नुहोस्।")

st.caption("© 2026 Jeevan-Sangini")
