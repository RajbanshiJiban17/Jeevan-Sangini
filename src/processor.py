import os
import io
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def process_pdf_to_vectorstore(data_source="data/"):
    try:
        all_documents = []
        if isinstance(data_source, str) and os.path.isdir(data_source):
            pdf_files = [f for f in os.listdir(data_source) if f.endswith('.pdf')]
            for pdf_file in pdf_files:
                loader = PyPDFLoader(os.path.join(data_source, pdf_file))
                all_documents.extend(loader.load())
        
        elif isinstance(data_source, str) and data_source.endswith('.pdf'):
            loader = PyPDFLoader(data_source)
            all_documents = loader.load()

        else: # Bytes stream from Streamlit
            pdf_data = data_source.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            # process_pdf_to_vectorstore फङ्सन भित्र:
        for page in pdf_reader.pages:
          content = page.extract_text()
        if content:
        # $ र + चिन्ह हटाउने ताकि एआई नझुक्कियोस्
         content = content.replace('$', '').replace('+', ' Plus').replace('\n', ' ')
        all_documents.append(Document(page_content=content))

        if not all_documents: return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(texts, embeddings)
    except Exception as e:
        print(f"Error: {e}")
        return None