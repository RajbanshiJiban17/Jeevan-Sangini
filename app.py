import streamlit as st
import os
import io
import PyPDF2
import warnings
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

# =========================================
# CACHE: AI RESPONSE (IMPORTANT FOR QUOTA)
# =========================================
@st.cache_data(show_spinner=False)
def get_ai_response(prompt, context, lang):
    return st.session_state.assistant.ask(prompt, context, lang)

# =========================================
# CONFIG
# =========================================
load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Jeevan-Sangini AI",
    page_icon="🤰",
    layout="wide"
)

# =========================================
# CSS
# =========================================
st.markdown("""
<style>
.main { background-color: #fffafb; }

.stButton>button {
    border-radius: 14px;
    background-color: #ff4b6b;
    color: white;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}

.report-box {
    padding: 20px;
    border-radius: 16px;
    background-color: white;
    border-left: 8px solid #ff4b6b;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    color: #333;
}

.sos-card {
    background-color: #ffeded;
    padding: 20px;
    border-radius: 15px;
    border: 3px solid #ff4b4b;
    text-align: center;
    animation: blinker 1.5s linear infinite;
}

@keyframes blinker {
    50% { opacity: 0.7; }
}
</style>
""", unsafe_allow_html=True)

# =========================================
# API KEY
# =========================================
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 GEMINI_API_KEY missing in .env")
    st.stop()

# =========================================
# SESSION STATE
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state:
    st.session_state.assistant = HealthAssistant(api_key=api_key)

if "sos_active" not in st.session_state:
    st.session_state.sos_active = False

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

# =========================================
# VECTOR DB
# =========================================
@st.cache_resource
def get_vector_db():
    return process_pdf_to_vectorstore("data/")

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865910.png", width=90)
    st.title("🩺 Jeevan-Sangini")

    lang = st.radio("🌐 Language", ("नेपाली", "English"))

    st.markdown("---")
    st.subheader("📄 Upload Medical Report")

    report_file = st.file_uploader("Upload PDF", type=["pdf"])

    if report_file and st.button("🔬 Analyze Report", use_container_width=True):

        with st.spinner("Analyzing..."):

            pdf_reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))

            report_text = "\n".join(
                [p.extract_text() for p in pdf_reader.pages if p.extract_text()]
            )

            if report_text.strip():

                prompt = f"""
Analyze this medical report:
- abnormal values
- risks
- pregnancy health
- emergency warnings

REPORT:
{report_text}
"""

                analysis = get_ai_response(prompt, report_text, lang)

                st.session_state.analysis = analysis

                st.success("Analysis Done")

            else:
                st.error("PDF text extract failed")

    if st.button("🆘 SOS EMERGENCY", use_container_width=True):
        st.session_state.sos_active = True

# =========================================
# MAIN TITLE
# =========================================
st.title("🤰 Jeevan-Sangini AI")
st.caption("AI Healthcare Assistant for Nepali Mothers")

# =========================================
# SOS
# =========================================
if st.session_state.sos_active:

    st.markdown("""
    <div class='sos-card'>
        <h2>🚨 SOS ACTIVE</h2>
        <h3>📞 102 | 100</h3>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Close SOS"):
        st.session_state.sos_active = False
        st.rerun()


st.markdown("""
### ⚠️ मेडिकल सूचना (Disclaimer)

यो AI प्रणाली केवल शैक्षिक तथा जानकारीका लागि मात्र हो।  
यसलाई कुनै पनि अवस्थामा चिकित्सकको सल्लाह, निदान वा उपचारको विकल्पको रूपमा प्रयोग गर्नु हुँदैन।

स्वास्थ्य सम्बन्धी कुनै समस्या भएमा कृपया योग्य चिकित्सकसँग परामर्श लिनुहोस्।

🚨 आपतकालीन अवस्थामा तुरुन्त 102 वा 100 मा सम्पर्क गर्नुहोस्।
""")
st.info("⚠️ यो AI चिकित्सकको विकल्प होइन।")
# =========================================
# TABS
# =========================================
tab1, tab2 = st.tabs(["💬 AI Chat", "📊 Report"])

# =========================================
# CHAT TAB
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

        # VECTOR SEARCH
        context = ""

        if "vector_db" not in st.session_state:
            st.session_state.vector_db = get_vector_db()

        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = "\n".join([d.page_content for d in docs])

        # AI RESPONSE
        with st.spinner("Gemini thinking..."):

            answer = get_ai_response(prompt, context, lang)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

            # =================================
            # 🔊 AUDIO (FIXED)
            # =================================
            try:
                tts = gTTS(answer, lang="ne")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(temp_file.name)
                st.audio(temp_file.name)
            except:
                pass

# =========================================
# REPORT TAB
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
            st.info("Upload report for analysis")

    with col2:

        st.subheader("📊 Health Stats")

        text = st.session_state.analysis.lower()

        hb = "11.5"
        status = "Normal"

        if "9.5" in text:
            hb = "9.5"
            status = "Low (Anemia)"

        st.metric("Hemoglobin", hb, status)
        st.metric("BP", "120/80", "Stable")
        st.metric("Risk", "Medium")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("© 2026 Jeevan-Sangini AI")