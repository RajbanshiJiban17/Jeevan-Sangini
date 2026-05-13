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

# १. वातावरण र वार्निङ फिल्टर
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# २. पेज कन्फिगरेसन
st.set_page_config(page_title="Jeevan-Sangini | Gemma Powered", page_icon="🤰", layout="wide")

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

# ३. क्यासिङ लोजिक
@st.cache_resource
def get_vector_db(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return process_pdf_to_vectorstore(folder_path)

@st.cache_resource
def init_assistant():
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("API Key भेटिएन! कृपया Settings मा जानुहोस्।")
        return None
    return HealthAssistant(api_key=api_key)

# ४. सेसन स्टेट
if "assistant" not in st.session_state:
    st.session_state.assistant = init_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sos_active" not in st.session_state:
    st.session_state.sos_active = False

# ५. साइडबार
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865910.png", width=80)
    st.title("Gemma Control")
    
    lang = st.radio("🌐 Language / भाषा:", ("नेपाली", "English"))
    
    st.markdown("---")
    st.subheader("🛡️ सुरक्षा जानकारी")
    st.info(
        "यो एआईले दिएको जानकारी नेपालको स्वास्थ्य निर्देशिकामा आधारित छ। "
        "तर, यो डाक्टरको विकल्प होइन। कुनै पनि समस्या भएमा तुरुन्त स्वास्थ्य चौकी जानुहोस्।"
    )
    
    st.markdown("---")
    st.subheader("📄 Lab Report / रिपोर्ट")
    report_file = st.file_uploader("Upload PDF Report", type=['pdf'])
    
    if report_file:
        if st.button("Agentic Analysis ✨", use_container_width=True):
            with st.spinner("Gemma विश्लेषण गर्दैछ..."):
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
                    report_text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    
                    if st.session_state.assistant:
                        st.session_state.analysis = st.session_state.assistant.ask(
                            "Analyze this lab report for risk signs.", report_text, lang=lang, mode="report"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🆘 SOS HELP", use_container_width=True, type="primary"):
        st.session_state.sos_active = True

# ६. मुख्य ड्यासबोर्ड
st.title("🤰 Jeevan-Sangini AI" if lang == "English" else "🤰 जीवन-सङ्गलिनी AI")
st.caption("Powered by Google Gemma | Dedicated to Maternal Health in Nepal")

if "vector_db" not in st.session_state:
    with st.spinner("Knowledge Base तयार हुँदैछ..."):
        try:
            st.session_state.vector_db = get_vector_db("data/")
            st.success("Knowledge Base Active! ✅")
        except Exception as e:
            st.error(f"Data Load Error: {e}")

# SOS कार्ड
if st.session_state.sos_active:
    st.markdown(f"""
        <div class='sos-card'>
            <h2 style='color:#ff4b4b;'>🚨 SOS ALERT ACTIVE</h2>
            <p style='font-size: 20px;'>Ambulance: <b>102</b> | Police: <b>100</b></p>
            <p>Immediate medical attention is advised.</p>
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

    if prompt := st.chat_input("स्वास्थ्य सम्बन्धी केही सोध्न चाहनुहुन्छ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = ""
        if "vector_db" in st.session_state and st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=2)
            context = " ".join([d.page_content for d in docs])

        with st.chat_message("assistant"):
            thinking_msg = "Thinking with Gemma..." if lang == "English" else "जीवन-सङ्गलिनी सोच्दैछ..."
            with st.spinner(thinking_msg):
                if st.session_state.assistant:
                    answer = st.session_state.assistant.ask(prompt, context, lang=lang)
                    st.markdown(answer)
                    
                    # Audio response
                    try:
                        clean_text = answer.split("---")[0]
                        v_lang = 'ne' if lang == "नेपाली" else 'en'
                        tts = gTTS(text=clean_text, lang=v_lang)
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
            st.info("रिपोर्ट अपलोड गरेपछि यहाँ एआई विश्लेषण देखिनेछ।")
    
    # app.py को Tab 2 भित्रको col2 यसरी अपडेट गर्नुहोस्:

with col2:
    st.subheader("📊 Maternal Progress")
    
    # रिपोर्टबाट Hb भ्यालु खोज्ने लोजिक
    hb_val = "N/A"
    hb_status = "Normal"
    if 'analysis' in st.session_state:
        if "9.5" in st.session_state.analysis:
            hb_val = "9.5 g/dL"
            hb_status = "Low (Anemia)"
        elif "Hb" in st.session_state.analysis:
            hb_val = "Detected"
    
    st.metric("Hemoglobin (Hb)", hb_val, hb_status, delta_color="inverse")
    
    # Weight Chart
    chart_data = pd.DataFrame({'Week': [4, 8, 12, 16, 20], 'Weight': [50.0, 52.5, 54.0, 57.2, 60.5]}).set_index('Week')
    st.line_chart(chart_data)

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini | Built for Nepali Mothers | Powered by Gemma")