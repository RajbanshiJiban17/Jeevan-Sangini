import streamlit as st
import os, io, PyPDF2, tempfile
from dotenv import load_dotenv
from gtts import gTTS
from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

# १. वातावरण सेटिङ
load_dotenv()
st.set_page_config(page_title="Jeevan-Sangini | Gemma 4 AI", page_icon="🤰", layout="wide")

# २. एआई र डेटाबेस लोड
if "assistant" not in st.session_state:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🚨 GOOGLE_API_KEY भेटिएन। कृपया .env वा Secrets मिलाउनुहोस्।")
        st.stop()
    st.session_state.assistant = HealthAssistant(api_key)

@st.cache_resource
def load_rag_db():
    return process_pdf_to_vectorstore("data/")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_rag_db()

if "messages" not in st.session_state: st.session_state.messages = []
if "analysis" not in st.session_state: st.session_state.analysis = ""

# ३. CSS Style (Professional Look)
st.markdown("""
<style>
    .stApp { background-color: #fffafb; }
    .report-card { padding: 20px; border-radius: 15px; background: white; border-left: 10px solid #ff4b6b; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .sos-btn { background-color: red !important; color: white !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ४. साइडबार
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865910.png", width=80)
    st.title("Gemma 4 Control")
    lang = st.radio("Language / भाषा:", ("नेपाली", "English"))
    
    st.markdown("---")
    report_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
    
    if report_file and st.button("🔬 Analyze Report"):
        with st.spinner("Gemma 4 study in progress..."):
            reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            st.session_state.analysis = st.session_state.assistant.ask("यो मेडिकल रिपोर्टको गहिरो विश्लेषण गर र आमाको स्वास्थ्य अवस्था बताऊ।", text, lang)

    if st.button("🆘 SOS HELP", use_container_width=True):
        st.toast("SOS Activated! Emergency numbers: 102, 100", icon="🚨")

# ५. मुख्य ड्यासबोर्ड
st.title("🤰 Jeevan-Sangini AI")
st.caption("Empowering Maternal Health with Gemma 4 Local Intelligence")

tab1, tab2 = st.tabs(["💬 Health Consultation", "📊 Medical Insights"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("तपाईँको स्वास्थ्य सम्बन्धी प्रश्न सोध्नुहोस्..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # RAG Context Retrieval
        context = ""
        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = " ".join([d.page_content for d in docs])
        
        with st.spinner("Gemma 4 सोच्दैछ..."):
            ans = st.session_state.assistant.ask(prompt, context, lang)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
                # TTS (Audio)
                try:
                    tts = gTTS(ans[:250], lang='ne' if lang=="नेपाली" else 'en')
                    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(f.name)
                    st.audio(f.name)
                except: pass

with tab2:
    if st.session_state.analysis:
        st.markdown(f"<div class='report-card'><h3>🔬 Lab Analysis Result</h3>{st.session_state.analysis}</div>", unsafe_allow_html=True)
    else:
        st.info("रिपोर्ट विश्लेषणको लागि कृपया साइडबारबाट फाइल अपलोड गर्नुहोस्।")

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini | Built for Kaggle Gemma 4 Impact Challenge")