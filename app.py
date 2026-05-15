import streamlit as st
import os, io, PyPDF2, tempfile
from dotenv import load_dotenv
from gtts import gTTS
from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

# १. सेटिङ
load_dotenv()
st.set_page_config(page_title="Jeevan-Sangini AI", page_icon="🤰", layout="wide")

# २. प्रोफेशनल लुकको लागि CSS
st.markdown("""
    <style>
    .stApp { background-color: #fffafb; }
    .report-card { padding: 20px; border-radius: 15px; background: white; border-left: 10px solid #ff4b6b; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .stChatFloatingInputContainer { background-color: #fffafb !important; }
    </style>
    """, unsafe_allow_html=True)

# ३. API Key तान्ने
api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else None)

if not api_key:
    st.error("🚨 GOOGLE_API_KEY भेटिएन।")
    st.stop()

# ४. एआई र डेटाबेस लोड
if "assistant" not in st.session_state:
    st.session_state.assistant = HealthAssistant(api_key)

@st.cache_resource
def load_db(): return process_pdf_to_vectorstore("data/")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_db()

if "messages" not in st.session_state: st.session_state.messages = []
if "analysis" not in st.session_state: st.session_state.analysis = ""

# ५. शीर्षक र डिस्क्लेमर
st.title("🤰 Jeevan-Sangini AI")
st.caption("Gemma 4 Local Intelligence for Global Health Equity")

st.warning("⚠️ यो एआई जानकारीको लागि मात्र हो। आकस्मिक अवस्थामा तुरुन्तै अस्पताल जानुहोस्।")

# ६. मुख्य भाग
tab1, tab2 = st.tabs(["💬 Consultation", "📊 Medical Insights"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("प्रश्न लेख्नुहोस्..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # RAG Context
        context = ""
        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = " ".join([d.page_content for d in docs])
        
        with st.spinner("विश्लेषण हुँदैछ..."):
            ans = st.session_state.assistant.ask(prompt, context, lang="नेपाली")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
                try:
                    tts = gTTS(ans[:250], lang='ne')
                    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(f.name)
                    st.audio(f.name)
                except: pass

with tab2:
    if st.session_state.analysis:
        st.markdown(f"<div class='report-card'><h3>🔬 Lab Analysis</h3>{st.session_state.analysis}</div>", unsafe_allow_html=True)
    else:
        st.info("रिपोर्ट विश्लेषणको लागि साइडबारबाट PDF अपलोड गर्नुहोस्।")

# ७. साइडबार
with st.sidebar:
    st.header("⚙️ Settings")
    report_file = st.file_uploader("Upload PDF", type=["pdf"])
    if report_file and st.button("Analyze Now"):
        with st.spinner("Processing..."):
            reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            st.session_state.analysis = st.session_state.assistant.ask("यो रिपोर्टको विश्लेषण गर:", text)
            st.rerun()

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini | Built for Kaggle Gemma 4 Challenge")