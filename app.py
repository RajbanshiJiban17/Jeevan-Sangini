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

# २. पेज कन्फिगरेसन र स्टाइल
st.set_page_config(page_title="Jeevan-Sangini", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fffafb; }
    .stButton>button { border-radius: 20px; background-color: #ff4b6b; color: white; border: none; }
    .report-box { padding: 20px; border-radius: 15px; background-color: white; border-left: 10px solid #ff4b6b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); color: #333; }
    .sos-card { background-color: #ffeded; padding: 20px; border-radius: 15px; border: 3px solid #ff4b4b; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ३. सेसन स्टेट सेटिङ
if "assistant" not in st.session_state:
    st.session_state.assistant = HealthAssistant(os.getenv("GROQ_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sos_active" not in st.session_state:
    st.session_state.sos_active = False

# ४. साइडबार
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865910.png", width=80)
    st.title("Settings / सेटिङ")
    
    lang = st.radio("🌐 Language / भाषा:", ("नेपाली", "English"))
    
    st.markdown("---")
    st.subheader("📄 Lab Report / रिपोर्ट")
    report_file = st.file_uploader("Upload PDF Report", type=['pdf'])
    
    if report_file:
        if st.button("Analyze ✨", use_container_width=True):
            with st.spinner("AI विश्लेषण गर्दैछ..."):
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
                    report_text = " ".join([p.extract_text() for p in reader.pages])
                    st.session_state.analysis = st.session_state.assistant.ask(
                        "Analyze this report", report_text, lang=lang, mode="report"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.error("🚨 EMERGENCY")
    if st.button("🆘 SOS HELP", use_container_width=True, type="primary"):
        st.session_state.sos_active = True

# ५. मुख्य ड्यासबोर्ड र Multi-PDF Load
st.title("🤰 Jeevan-Sangini" if lang == "English" else "🤰 जीवन-सङ्गिनी")

if "vector_db" not in st.session_state:
    with st.spinner("स्वास्थ्य ज्ञान भण्डार (Knowledge Base) तयार हुँदैछ..."):
        try:
            # यहाँ 'data/' फोल्डर पठाउने जसले भित्रका सबै PDF हरू लोड गर्छ
            st.session_state.vector_db = process_pdf_to_vectorstore("data/")
            if st.session_state.vector_db:
                st.success("सबै मेडिकल गाइडहरू लोड भए!")
        except Exception as e:
            st.error(f"डेटा लोड हुन सकेन: {e}")

# SOS अलर्ट
if st.session_state.sos_active:
    st.markdown(f"""
        <div class='sos-card'>
            <h2 style='color:#ff4b4b;'>🚨 आपतकालीन अलर्ट सक्रिय (SOS)</h2>
            <p style='font-size: 20px;'>एम्बुलेन्स: <b>१०२</b> | प्रहरी: <b>१००</b></p>
            <p>नजिकैको अस्पताल तुरुन्तै जानुहोस्।</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Close SOS / बन्द गर्नुहोस्"):
        st.session_state.sos_active = False
        st.rerun()

# ६. ट्याब सिस्टम
tab1, tab2 = st.tabs(["💬 Consultation", "📊 Tracker"])

with tab1:
    # च्याट हिस्ट्री देखाउने
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # च्याट इनपुट
    if prompt := st.chat_input("Ask about pregnancy or general health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Search
        context = ""
        if "vector_db" in st.session_state and st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            context = " ".join([d.page_content for d in docs])

        with st.chat_message("assistant"):
            with st.spinner("AI जवाफ तयार पार्दैछ..."):
                answer = st.session_state.assistant.ask(prompt, context, lang=lang)
                st.markdown(answer)
                
                # Audio response
                try:
                    v_lang = 'ne' if lang == "नेपाली" else 'en'
                    tts = gTTS(text=answer, lang=v_lang)
                    tts.save("response.mp3")
                    st.audio("response.mp3")
                except:
                    pass
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        if 'analysis' in st.session_state:
            st.markdown(f"<div class='report-box'><h3>🔬 Analysis Result</h3>{st.session_state.analysis}</div>", unsafe_allow_html=True)
        else:
            st.info("रिपोर्ट विश्लेषणको नतिजा यहाँ देखिनेछ।")
    
    with col2:
        st.subheader("📊 Statistics")
        st.metric("Hb Level", "11.5 g/dL", "Normal")
        
        chart_data = pd.DataFrame({
            'Week': [4, 8, 12, 16, 20], 
            'Weight': [50.0, 52.5, 54.0, 57.2, 60.5]
        }).set_index('Week')
        
        st.line_chart(chart_data)

st.markdown("---")
st.caption("© 2026 Jeevan-Sangini | Digital Health Companion")