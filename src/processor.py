import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_pdf_to_vectorstore(data_source="data/"):
    """स्थानीय PDF हरूलाई AI ले बुझ्ने गरी प्रोसेस गर्छ।"""
    if not os.path.exists(data_source) or not os.listdir(data_source):
        return None

    all_docs = []
    for file in os.listdir(data_source):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(data_source, file))
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    if not all_docs: return None

    # टेक्स्टलाई स-साना टुक्रामा बाँड्ने (Gemma को लागि सजिलो हुन्छ)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # फ्री एम्बेडेड मोडेल (यो डाउनलोड भएपछि अफलाइन चल्छ)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # FAISS भेक्टर स्टोर निर्माण
    return FAISS.from_documents(chunks, embeddings)