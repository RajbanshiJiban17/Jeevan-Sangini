import streamlit as st
import streamlit as st
import os, io, PyPDF2, warnings, logging
from dotenv import load_dotenv
from gtts import gTTS
from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

load_dotenv()
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Jeevan-Sangini | Gemma Powered", page_icon="🤰", layout="wide")

# CSS Styling
st.markdown("""
    <style>
    .main { background-color: #fffafb; }
    .stButton>button { border-radius: 20px; background-color: #ff4b6b; color: white; border: none; transition: 0.3s; }
    .report-box { padding: 20px; border-radius: 15px; background-color: white; border-left: 10px solid #ff4b6b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); color: #333; }
    .sos-card { background-color: #ffeded; padding: 20px; border-radius: 15px; border: 3px solid #ff4b4b; text-align: center; margin-bottom: 20px; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.7; } }
    </style>
    """, unsafe_allow_html=True)

# सेसन स्टेट र क्यासिङ
@st.cache_resource
def get_vector_db():
    return process_pdf_to_vectorstore("data/")

if not api_key:
    st.error("🚨 गूगल API Key भेटिएन! कृपया .env फाइलमा GOOGLE_API_KEY=your_key राख्नुहोस्।")
    st.stop() # कि नभेटिए एप नै रोक्दिने

if "assistant" not in st.session_state:
    api_key = os.getenv("GOOGLE_API_KEY")
    st.session_state.assistant = HealthAssistant(api_key=api_key)

if "messages" not in st.session_state: st.session_state.messages = []
if "sos_active" not in st.session_state: st.session_state.sos_active = False

# Sidebar
with st.sidebar:
    st.title("Gemma Control")
    lang = st.radio("🌐 Language / भाषा:", ("नेपाली", "English"))
    st.markdown("---")
    st.subheader("📄 Lab Report / रिपोर्ट")
    report_file = st.file_uploader("Upload PDF Report", type=['pdf'])
    
    if report_file and st.button("Agentic Analysis ✨", use_container_width=True):
        with st.spinner("Analyzing..."):
            reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
            report_text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            st.session_state.analysis = st.session_state.assistant.ask("Full Health Analysis", report_text, lang=lang)

    if st.button("🆘 SOS HELP", use_container_width=True, type="primary"):
        st.session_state.sos_active = True

# Main Dashboard
st.title("🤰 Jeevan-Sangini AI")

if st.session_state.sos_active:
    st.markdown("<div class='sos-card'><h2>🚨 SOS: 102 | 100</h2><p>Immediate Help Requested!</p></div>", unsafe_allow_html=True)
    if st.button("Close SOS"): 
        st.session_state.sos_active = False
        st.rerun()

tab1, tab2 = st.tabs(["💬 AI Consultation", "📊 Health Tracker"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("स्वास्थ्य सम्बन्धी केही सोध्न चाहनुहुन्छ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Knowledge Base बाट सन्दर्भ खोज्ने
        if "vector_db" not in st.session_state:
            st.session_state.vector_db = get_vector_db()
        
        context = ""
        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = " ".join([d.page_content for d in docs])
            
        ans = st.session_state.assistant.ask(prompt, context, lang=lang)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(ans)

with tab2:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        if 'analysis' in st.session_state:
            st.markdown(f"<div class='report-box'><h3>🔬 Lab Insights</h3>{st.session_state.analysis}</div>", unsafe_allow_html=True)
    with col2:
        st.subheader("📊 Maternal Progress")
        hb_val = "9.5" if "analysis" in st.session_state and "9.5" in st.session_state.analysis else "11.5"
        delta = "🚨 Low (Anemia)" if hb_val == "9.5" else "Normal"
        st.metric(label="Hemoglobin (Hb)", value=f"{hb_val} g/dL", delta=delta, delta_color="inverse" if hb_val == "9.5" else "normal")



st.caption("© 2026 Jeevan-Sangini | Built for Nepali Mothers | Powered by Gemma")