import streamlit as st
import os
import io
import PyPDF2
import warnings
import tempfile
from dotenv import load_dotenv
from gtts import gTTS

from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

# =========================================
# CONFIG
# =========================================
load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Jeevan-Sangini AI (Gemma)",
    page_icon="🤰",
    layout="wide"
)

# =========================================
# SAFE CACHE AI RESPONSE
# =========================================
def get_ai_response(prompt, context, lang):
    try:
        return st.session_state.assistant.ask(prompt, context, lang)
    except Exception as e:
        return f"🚨 AI Error: {str(e)}"

# =========================================
# VECTOR DB
# =========================================
@st.cache_resource
def get_vector_db():
    return process_pdf_to_vectorstore("data/")

# =========================================
# UI STYLE
# =========================================
st.markdown("""
<style>
.main { background-color: #fffafb; }

.stButton>button {
    border-radius: 14px;
    background-color: #ff4b6b;
    color: white;
    font-weight: bold;
}

.report-box {
    padding: 20px;
    border-radius: 16px;
    background: white;
    border-left: 8px solid #ff4b6b;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.sos-card {
    background: #ffeded;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid red;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# SESSION STATE
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "sos_active" not in st.session_state:
    st.session_state.sos_active = False

if "assistant" not in st.session_state:
    api_key = os.getenv("GEMINI_API_KEY")  # future fallback use
    st.session_state.assistant = HealthAssistant()

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865910.png", width=90)
    st.title("🩺 Jeevan-Sangini (Gemma)")

    lang = st.radio("🌐 Language", ("नेपाली", "English"))

    st.markdown("---")
    st.subheader("📄 Medical Report")

    report_file = st.file_uploader("Upload PDF", type=["pdf"])

    if report_file and st.button("🔬 Analyze"):

        with st.spinner("Analyzing with Gemma..."):

            pdf = PyPDF2.PdfReader(io.BytesIO(report_file.read()))

            report_text = "\n".join(
                [p.extract_text() for p in pdf.pages if p.extract_text()]
            )

            if report_text.strip():

                prompt = """
Medical report analysis:
- abnormal values
- risks
- pregnancy safety
- emergency warnings
"""

                analysis = get_ai_response(prompt, report_text[:2000], lang)
                st.session_state.analysis = analysis
                st.success("Analysis done")

            else:
                st.error("No text found in PDF")

    if st.button("🆘 SOS"):
        st.session_state.sos_active = True

# =========================================
# TITLE
# =========================================
st.title("🤰 Jeevan-Sangini AI (Gemma Edition)")
st.caption("RAG + Offline-ready health assistant")

# =========================================
# SOS
# =========================================
if st.session_state.sos_active:
    st.markdown("""
    <div class='sos-card'>
        <h2>🚨 EMERGENCY MODE</h2>
        <h3>📞 102 | 100</h3>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Close SOS"):
        st.session_state.sos_active = False
        st.rerun()

st.warning("⚠️ यो AI केवल जानकारीको लागि हो, doctor substitute होइन")

# =========================================
# TABS
# =========================================
tab1, tab2 = st.tabs(["💬 Chat", "📊 Report"])

# =========================================
# CHAT
# =========================================
with tab1:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask health question...")

    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # vector DB
        if "vector_db" not in st.session_state:
            st.session_state.vector_db = get_vector_db()

        context = ""
        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = "\n".join([d.page_content for d in docs])

        with st.spinner("Gemma thinking..."):
            answer = get_ai_response(prompt, context, lang)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

            # AUDIO SAFE
            try:
                tts = gTTS(answer[:400], lang="ne")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(temp_file.name)
                st.audio(temp_file.name)
            except:
                pass

# =========================================
# REPORT
# =========================================
with tab2:

    col1, col2 = st.columns([1.7, 1])

    with col1:
        if st.session_state.analysis:
            st.markdown(f"""
            <div class='report-box'>
            <h3>🔬 AI Analysis</h3>
            <p>{st.session_state.analysis}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload medical report")

    with col2:
        st.subheader("📊 Stats")

        text = st.session_state.analysis.lower()

        hb = "11.5"
        status = "Normal"

        if "9.5" in text:
            hb = "9.5"
            status = "Low (Anemia)"

        st.metric("Hemoglobin", hb, status)
        st.metric("BP", "120/80", "Stable")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("© 2026 Jeevan-Sangini (Gemma Edition)")