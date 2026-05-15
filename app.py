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
from src.config import DATA_DIR, GEMINI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL
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

# --- UI FIX: White/Text Visibility ---
st.markdown(
    """
    <style>
    /* मुख्य ब्याकग्राउन्ड र फन्ट सुधार */
    .stApp {
        background: linear-gradient(165deg, #fff5f7 0%, #fce4ec 45%, #f8bbd0 100%);
    }
    
    /* च्याट भित्रका अक्षरहरूलाई एकदमै स्पष्ट बनाउने */
    .stMarkdown p, .stMarkdown li {
        color: #1a1a1a !important; /* गाढा कालो */
        font-weight: 450 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }

    /* च्याट मेसेज बक्स सुधार */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid #ffccd5;
        border-radius: 15px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Sidebar का अक्षरहरू */
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-size: 1rem !important;
    }

    /* Tab अक्षरहरू */
    button[data-baseweb="tab"] p {
        color: #4a1942 !important;
        font-weight: 600 !important;
    }

    /* Warning/Error सन्देशको टेक्स्ट */
    div[data-testid="stNotification"] p {
        color: #333333 !important;
        font-weight: 500 !important;
    }

    .report-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: #ffffff;
        color: #1a1a1a;
        border-left: 8px solid #ff4b6b;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .mode-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .mode-local { background: #2e7d32; color: #fff; }
    .mode-cloud { background: #1565c0; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session Management ---
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

# Backend Logic
_ollama_up = is_ollama_running(OLLAMA_BASE_URL)
_ollama_models = list_models() if _ollama_up else []
_has_gemini = bool(get_gemini_api_key())
on_cloud = is_streamlit_cloud()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ सेटिङ")
    
    if _ollama_up:
        st.markdown('<div class="status-box status-ok" style="background:#e8f5e9; padding:10px; border-radius:5px; color:#2e7d32;">🟢 Ollama Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-bad" style="background:#ffebee; padding:10px; border-radius:5px; color:#c62828;">🔴 Ollama Offline</div>', unsafe_allow_html=True)

    mode_options = ["Auto"]
    mode_map = {"Auto": None}
    if _ollama_up:
        mode_options.append("Offline (Ollama)")
        mode_map["Offline (Ollama)"] = "ollama"
    if _has_gemini:
        mode_options.append("Online (Gemini)")
        mode_map["Online (Gemini)"] = "gemini"

    choice_label = st.radio("AI Backend", mode_options, index=0)
    user_backend_choice = mode_map.get(choice_label)

BACKEND = resolve_backend(_ollama_up, user_backend_choice)

if ( "assistant" not in st.session_state or st.session_state.active_backend != BACKEND ):
    st.session_state.assistant = HealthAssistant(backend=BACKEND)
    st.session_state.active_backend = BACKEND

# --- Main App Interface ---
st.title("🤰 Jeevan-Sangini AI")

mode_class = "mode-local" if BACKEND == "ollama" else "mode-cloud"
mode_label = f"🖥️ Local: {OLLAMA_MODEL}" if BACKEND == "ollama" else f"☁️ Cloud: {GEMINI_MODEL}"
st.markdown(f'<span class="mode-pill {mode_class}">{mode_label}</span>', unsafe_allow_html=True)

st.warning("⚠️ यो सफ्टवेयर जानकारीका लागि मात्र हो। आपतकालमा चिकित्सकसँग सम्पर्क गर्नुहोस्।")

tab_chat, tab_report, tab_track = st.tabs(["💬 परामर्श", "📊 रिपोर्ट", "📅 हप्ता"])

with tab_chat:
    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"])

    if prompt := st.chat_input("यहाँ केही लेख्नुहोस्..."):
        # Emergency Assessment
        risk = assess_emergency(prompt)
        if risk["level"] != "none":
            msg = risk["message_ne"] if st.session_state.lang_pref == "नेपाली" else risk["message_en"]
            if risk["level"] == "high": st.error(msg)
            else: st.warning(msg)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("सोच्दै..."):
                ans = st.session_state.assistant.ask(
                    prompt,
                    lang=st.session_state.lang_pref,
                    pregnancy_week=st.session_state.pregnancy_week
                )
            st.markdown(ans)
            
            # Voice handling
            ap = None
            if st.session_state.voice_enabled:
                lc = "ne" if st.session_state.lang_pref == "नेपाली" else "en"
                ap = text_to_speech(ans[:300], lang=lc)
                if ap: st.audio(ap)
            
            st.session_state.messages.append({"role": "assistant", "content": ans, "audio": ap})

with tab_report:
    if st.session_state.analysis:
        st.markdown(f"<div class='report-card'>{st.session_state.analysis}</div>", unsafe_allow_html=True)
    else:
        st.info("रिपोर्ट विश्लेषणका लागि साइडबारबाट PDF अपलोड गर्नुहोस्।")

with tab_track:
    w = st.session_state.pregnancy_week
    if w:
        st.subheader(f"गर्भावस्थाको हप्ता {w}")
        # यहाँ हप्ता अनुसारको डेटा थप्न सकिन्छ
        st.write("यो हप्ताको लागि मुख्य सल्लाह: सन्तुलित आहार र नियमित व्यायाम।")
    else:
        st.info("विवरण हेर्न साइडबारमा हप्ता सेट गर्नुहोस्।")

st.caption("© 2026 Jeevan-Sangini | Empowering Motherhood")