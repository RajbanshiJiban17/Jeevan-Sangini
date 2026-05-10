# 🤰 Jeevan-Sangini (जीवन-सङ्गिनी)
### *Digital Health Companion for Mothers & General Wellness*
### *आमा र सामान्य स्वास्थ्यका लागि डिजिटल स्वास्थ्य साथी*

Jeevan-Sangini is an AI-powered health assistant designed to provide reliable medical guidance based on official health guidelines. It uses **RAG (Retrieval-Augmented Generation)** to process medical manuals and analyze lab reports, delivering answers in both **Nepali** and **English**.

---

## 🚀 Key Features | मुख्य विशेषताहरू

### 1. **Multi-PDF Knowledge Base**
- **English:** Fetches real-time information from official government guidelines regarding Pregnancy, Nutrition, and Non-Communicable Diseases (NCDs).
- **नेपाली:** आधिकारिक सरकारी निर्देशिकाहरूबाट गर्भावस्था, पोषण, र नसर्ने रोगहरू सम्बन्धी भरपर्दो जानकारी खोज्छ।

### 2. **AI Medical Report Analysis**
- **English:** Automatically analyzes uploaded Lab Reports (PDFs) to explain complex clinical data (Hb, WBC, Sugar, TSH) in simple terms.
- **नेपाली:** अपलोड गरिएको ल्याब रिपोर्ट (PDF) विश्लेषण गरी जटिल मेडिकल डेटालाई सरल भाषामा बुझाउँछ।

### 3. **Bilingual Support & Voice**
- **English:** Full support for English and Nepali text-to-speech for better accessibility.
- **नेपाली:** नेपाली र अंग्रेजी दुवै भाषामा कुराकानी र आवाज (Voice) सुन्न मिल्ने सुविधा।

### 4. **Emergency SOS System**
- **English:** One-click access to emergency numbers (Ambulance 102, Police 100).
- **नेपाली:** आपतकालीन अवस्थाका लागि एम्बुलेन्स र प्रहरीको नम्बरमा तुरुन्त पहुँच।

---

## 🛠️ How it Works | यसले कसरी काम गर्छ?

This project is built using the **RAG** architecture to ensure zero hallucinations and high factual accuracy:

1.  **Ingestion:** Medical PDFs in the `data/` folder are processed and split into chunks.
2.  **Embedding:** Text chunks are converted into vector embeddings using `HuggingFace (all-MiniLM-L6-v2)`.
3.  **Storage:** Vectors are stored in a `FAISS` vector database for high-speed similarity search.
4.  **Retrieval:** When a user asks a question, the system retrieves relevant context from the official documents.
5.  **Generation:** The `Groq Cloud (LLaMA-3-70B)` model synthesizes the context into a human-like response.

---

## 🔮 Future Roadmap & Improvements | भविष्यका योजनाहरू

*   **Refined Medical Logic:** Enhancing interpretation of 'Normal' vs 'Abnormal' ranges based on trimester-specific standards.
*   **OCR Integration:** Ability to analyze reports directly from a mobile camera photo.
*   **Wearable Sync:** Integration with smartwatches to monitor real-time vital signs.
*   **Local Dialects:** Expanding support to Maithili, Bhojpuri, and other local languages.

---

## 📚 Tech Stack | प्रविधिहरू

- **LLM:** Groq (LLaMA-3-70B)
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace
- **Speech:** gTTS (Google Text-to-Speech)

---

## ⚠️ Disclaimer & Ethics | डिस्क्लेमर र नैतिक पक्ष

- **Not a Medical Device:** This is an informational tool and **NOT** a substitute for professional medical advice.
- **Verification:** Always verify AI-generated data with a qualified healthcare professional.
- **Privacy:** We do not store sensitive personal health data on public servers.

---
depolyed:https://jeevan-sangini.streamlit.app/
