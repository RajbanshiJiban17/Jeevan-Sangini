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

                path = os.path.join(data_source, file)

                if file.endswith(".pdf"):

                    loader = PyPDFLoader(path)

                    docs = loader.load()

                    all_documents.extend(docs)

        if not all_documents:
            return None

        # smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_documents)

        print(f"✅ Total chunks: {len(chunks)}")

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vectorstore

    except Exception as e:

        print("Processor Error:", e)

        return None