import streamlit as st
import os
import pandas as pd
import warnings
import io
import PyPDF2
import logging
from dotenv import load_dotenv
from gtts import gTTS
from src.processor import process_pdf_to_vectorstore
from src.assistant import HealthAssistant

# १. वातावरण र वार्निङ फिल्टर (Optimization for Speed)
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# २. पेज कन्फिगरेसन र स्टाइल
st.set_page_config(page_title="Jeevan-Sangini | Gemma 4 Powered", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fffafb; }
    .stButton>button { border-radius: 20px; background-color: #ff4b6b; color: white; border: none; transition: 0.3s; }
    .stButton>button:hover { background-color: #e63958; transform: scale(1.05); }
    .report-box { padding: 20px; border-radius: 15px; background-color: white; border-left: 10px solid #ff4b6b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); color: #333; }
    .sos-card { background-color: #ffeded; padding: 20px; border-radius: 15px; border: 3px solid #ff4b4b; text-align: center; margin-bottom: 20px; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.7; } }
    </style>
    """, unsafe_allow_html=True)

# ३. क्यासिङ लोजिक (एप छिटो खुल्ने बनाउन - Fast Loading)
@st.cache_resource
def get_vector_db(folder_path):
    """PDF लोड गर्ने र Vector Store बनाउने कार्यलाई क्यास गर्छ ताकि प्रत्येक पटक लोड नहोस्।"""
    return process_pdf_to_vectorstore(folder_path)

@st.cache_resource
def init_assistant():
    """Gemma 4 आधारित एसासिस्टेन्ट सुरु गर्छ।"""
    # नोट: Groq मा Gemma 4 को पछिल्लो मोडेल नेम 'gemma2-9b-it' वा 'gemma-7b-it' हुन सक्छ। 
    # च्यालेन्जको लागि हामी 'gemma2-9b-it' (वा उपलब्ध पछिल्लो) प्रयोग गर्छौँ।
    return HealthAssistant(api_key=os.getenv("GOOGLE_API_KEY"), model_name="models/gemma-1.1-7b-it")

# ४. सेसन स्टेट सेटिङ
if "assistant" not in st.session_state:
    st.session_state.assistant = init_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sos_active" not in st.session_state:
    st.session_state.sos_active = False

# ५. साइडबार र रिपोर्ट अपलोड
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865910.png", width=80)
    st.title("Gemma 4 Control")
    
    lang = st.radio("🌐 Language / भाषा:", ("नेपाली", "English"))
    
    st.markdown("---")
    st.subheader("📄 Lab Report / रिपोर्ट")
    report_file = st.file_uploader("Upload PDF Report", type=['pdf'])
    
    with st.sidebar:
        st.title("🛡️ सुरक्षा जानकारी")
        st.info(
        "यो एआई (Gemma 2) ले दिएको जानकारी नेपालको स्वास्थ्य निर्देशिकामा आधारित छ। "
        "तर, यो डाक्टरको विकल्प होइन। कुनै पनि समस्या भएमा तुरुन्त स्वास्थ्य चौकी जानुहोस्।"
    )
    
    if report_file:
        if st.button("Agentic Analysis ✨", use_container_width=True):
            with st.spinner("Gemma 4 विश्लेषण गर्दैछ..."):
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
                    report_text = " ".join([p.extract_text() for p in reader.pages])
                    # रिपोर्टमा खतरा देखिए एजेन्टले आफैँ चेतावनी दिन्छ
                    st.session_state.analysis = st.session_state.assistant.ask(
                        "Analyze this lab report for risk signs.", report_text, lang=lang, mode="report"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.error("🚨 EMERGENCY")
    if st.button("🆘 SOS HELP", use_container_width=True, type="primary"):
        st.session_state.sos_active = True

# ६. मुख्य ड्यासबोर्ड र Knowledge Base
st.title("🤰 Jeevan-Sangini AI" if lang == "English" else "🤰 जीवन-सङ्गिनी AI")
st.caption("Powered by Gemma 4 | Frontier Intelligence for Maternal Health")

if "vector_db" not in st.session_state:
    with st.spinner("Knowledge Base तयार हुँदैछ (यसले पहिलो पटक मात्र समय लिन्छ)..."):
        try:
            st.session_state.vector_db = get_vector_db("data/")
            st.success("Grounded Knowledge Base Active! ✅")
        except Exception as e:
            st.error(f"Data Load Error: {e}")

# SOS कार्ड
if st.session_state.sos_active:
    st.markdown(f"""
        <div class='sos-card'>
            <h2 style='color:#ff4b4b;'>🚨 SOS ALERT ACTIVE</h2>
            <p style='font-size: 20px;'>Ambulance: <b>102</b> | Police: <b>100</b></p>
            <p>Seeking immediate medical attention is advised.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Close SOS"):
        st.session_state.sos_active = False
        st.rerun()

# ७. ट्याब सिस्टम
tab1, tab2 = st.tabs(["💬 AI Consultation", "📊 Health Tracker"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agentic RAG Search
        context = ""
        if "vector_db" in st.session_state and st.session_state.vector_db:
            # Similarity search to ground the model
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            context = " ".join([d.page_content for d in docs])

        with st.chat_message("assistant"):
            with st.spinner("Thinking with Gemma 4..."):
                answer = st.session_state.assistant.ask(prompt, context, lang=lang)
                st.markdown(answer)
                
                # Audio response for Inclusivity
                try:
                    v_lang = 'ne' if lang == "नेपाली" else 'en'
                    tts = gTTS(text=answer, lang=v_lang)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format="audio/mp3")
                except:
                    pass
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        if 'analysis' in st.session_state:
            st.markdown(f"<div class='report-box'><h3>🔬 Lab Insights</h3>{st.session_state.analysis}</div>", unsafe_allow_html=True)
        else:
            st.info("Upload a report to see AI-driven insights.")
    
    with col2:
        st.subheader("📊 Maternal Progress")
        st.metric("Hb Level", "11.5 g/dL", "Stable")
        chart_data = pd.DataFrame({'Week': [4, 8, 12, 16, 20], 'Weight': [50.0, 52.5, 54.0, 57.2, 60.5]}).set_index('Week')
        st.line_chart(chart_data)

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini |")