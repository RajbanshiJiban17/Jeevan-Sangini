import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_pdf_to_vectorstore(data_source="data/"):
    """मेडिकल फाइलहरूलाई स्थानीय भेक्टर डेटाबेसमा बदल्छ (Offline Intelligence)"""
    if not os.path.exists(data_source) or not os.listdir(data_source):
        return None

    all_docs = []
    for file in os.listdir(data_source):
        if file.endswith(".pdf"):
            path = os.path.join(data_source, file)
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")

    if not all_docs:
        return None

    # Gemma 4 को लागि राम्रो कन्टेक्स्ट दिन टेक्स्टलाई टुक्रा पार्ने
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    # स्थानीय रूपमा चल्ने एम्बेडेड मोडेल (No API Key needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # स्थानीय FAISS डेटाबेस निर्माण
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db