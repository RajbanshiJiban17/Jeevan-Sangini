import streamlit as st
import os, io, PyPDF2, tempfile
from dotenv import load_dotenv
from gtts import gTTS
from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

# १. कन्फिगरेसन
load_dotenv()
st.set_page_config(page_title="Jeevan-Sangini | Gemma 4 AI", page_icon="🤰", layout="wide")

# २. प्रोफेशनल लुक्सको लागि CSS (सेतो खाली ठाउँ हटाउन)
st.markdown("""
    <style>
    .stApp { background-color: #fffafb; }
    .report-box { padding: 20px; border-radius: 15px; background: white; border-left: 10px solid #ff4b6b; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .stChatFloatingInputContainer { background-color: #fffafb !important; }
    </style>
    """, unsafe_allow_html=True)

# ३. API Key तान्ने (Local र Streamlit Cloud दुवैको लागि)
api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else None)

if not api_key:
    st.error("🚨 GOOGLE_API_KEY भेटिएन। कृपया .env वा Streamlit Secrets मा मिलाउनुहोस्।")
    st.stop()

# ४. एआई लोड गर्ने
if "assistant" not in st.session_state:
    st.session_state.assistant = HealthAssistant(api_key)

@st.cache_resource
def get_vector_db():
    return process_pdf_to_vectorstore("data/")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = get_vector_db()

if "messages" not in st.session_state: st.session_state.messages = []
if "analysis" not in st.session_state: st.session_state.analysis = ""

# ५. मुख्य शीर्षक र चेतावनी
st.title("🤰 Jeevan-Sangini AI")
st.caption("Gemma 4 Local Intelligence for Global Health Equity")

st.info("⚠️ **मेडिकल डिस्क्लेमर:** यो एआई जानकारीको लागि मात्र हो। आकस्मिक अवस्थामा तुरुन्तै अस्पताल जानुहोस्।")

# ६. ट्याबहरू
tab1, tab2 = st.tabs(["💬 Consultation (परामर्श)", "📊 Analysis (विवरण)"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("तपाईँको स्वास्थ्य जिज्ञासा लेख्नुहोस्..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # RAG Context Retrieval
        context = ""
        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = " ".join([d.page_content for d in docs])
        
        with st.spinner("Gemma 4 विश्लेषण गर्दैछ..."):
            ans = st.session_state.assistant.ask(prompt, context, lang="नेपाली")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
                # अडियो फिचर
                try:
                    tts = gTTS(ans[:250], lang='ne')
                    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(f.name)
                    st.audio(f.name)
                except: pass

with tab2:
    if st.session_state.analysis:
        st.markdown(f"<div class='report-box'><h3>🔬 Lab Report Analysis</h3>{st.session_state.analysis}</div>", unsafe_allow_html=True)
    else:
        st.info("रिपोर्ट विश्लेषणको लागि साइडबारबाट PDF अपलोड गर्नुहोस्।")

# ७. साइडबार
with st.sidebar:
    st.header("⚙️ Controls")
    report_file = st.file_uploader("Upload Medical PDF", type=["pdf"])
    if report_file and st.button("Analyze Report"):
        with st.spinner("Analyzing..."):
            reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            st.session_state.analysis = st.session_state.assistant.ask("यो रिपोर्टको विस्तृत विश्लेषण गर:", text)
            st.rerun()

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini | Built for Kaggle Gemma 4 Challenge")