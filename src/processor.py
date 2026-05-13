import os
import io
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def process_pdf_to_vectorstore(data_source="data/"):
    try:
        all_documents = []
        if os.path.isdir(data_source):
            for file in os.listdir(data_source):
                f_path = os.path.join(data_source, file)
                # PDF को लागि
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(f_path)
                    all_documents.extend(loader.load())
                # Parquet को लागि (तपाईँको pregnancy_data.parquet)
                elif file.endswith('.parquet'):
                    df = pd.read_parquet(f_path)
                    for _, row in df.iterrows():
                        content = f"Instruction: {row.get('instruction','')} Output: {row.get('output','')}"
                        all_documents.append(Document(page_content=content))

        # डेटा सफा गर्ने (Cleaning bugs like $ sign)
        for doc in all_documents:
            doc.page_content = doc.page_content.replace('$', '').replace('\n', ' ')

        if not all_documents: return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(all_documents)
        
        # Local Embedding - यसले API कोटा खर्च गर्दैन
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"Processor Error: {e}")
        return None