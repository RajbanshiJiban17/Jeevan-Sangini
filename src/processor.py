import os
import pandas as pd
import io
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def process_pdf_to_vectorstore(data_source="data/"):
    """
    डाटा फोल्डर भित्रका सबै PDF हरूलाई लोड गर्ने र एउटै Vector Database बनाउने।
    """
    try:
        all_documents = []
        
        # १. फोल्डर भित्रका सबै PDF हरू खोज्ने र लोड गर्ने
        if isinstance(data_source, str) and os.path.isdir(data_source):
            pdf_files = [f for f in os.listdir(data_source) if f.endswith('.pdf')]
            
            if not pdf_files:
                print(f"No PDF files found in {data_source}")
                return None
                
            for pdf_file in pdf_files:
                file_path = os.path.join(data_source, pdf_file)
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
                print(f"Successfully loaded: {pdf_file}")
        
        # यदि सिधै एउटा फाइल पाथ मात्र पठाइएको छ भने
        elif isinstance(data_source, str) and data_source.endswith('.pdf'):
            loader = PyPDFLoader(data_source)
            all_documents = loader.load()

        # यदि स्ट्रिमलिटबाट फाइल अपलोड गरिएको हो भने (Bytes)
        else:
            pdf_data = data_source.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    all_documents.append(Document(page_content=content))

        if not all_documents:
            print("No text found in any PDF.")
            return None

        # २. Text Splitting (Chunks बनाउने)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, # थोरै ठूलो साइज राख्दा धेरै फाइलको कन्टेक्स्ट राम्रो आउँछ
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )
        texts = text_splitter.split_documents(all_documents)

        # ३. Embeddings र Vector Store (FAISS)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(texts, embeddings)
        
        print(f"Vector Database created with {len(texts)} chunks from multiple files.")
        return vector_db

    except Exception as e:
        print(f"Error in processing PDFs: {e}")
        return None

def load_qa_data(parquet_path):
    """Parquet फाइलबाट डाटा लोड गर्न"""
    try:
        return pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading Parquet: {e}")
        return None