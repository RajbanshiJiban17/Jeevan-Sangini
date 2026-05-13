import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def process_pdf_to_vectorstore(data_source="data/"):
    try:
        all_documents = []
        if os.path.exists(data_source):
            for file in os.listdir(data_source):
                f_path = os.path.join(data_source, file)
                # PDF प्रशोधन
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(f_path)
                    all_documents.extend(loader.load())
                # Parquet प्रशोधन
                elif file.endswith('.parquet'):
                    df = pd.read_parquet(f_path)
                    for _, row in df.iterrows():
                        content = f"Question: {row.get('instruction','')} Answer: {row.get('output','')}"
                        all_documents.append(Document(page_content=content))

        # $ चिन्ह र अनावश्यक स्पेस सफा गर्ने
        for doc in all_documents:
            doc.page_content = doc.page_content.replace('$', '').replace('\n', ' ').strip()

        if not all_documents: 
            return None

        # चङ्किङ (Chunks) गर्ने
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_documents)
        
        # लोकल एम्बेडिङ (API कोटा बचत गर्छ)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"Processor Error: {e}")
        return None