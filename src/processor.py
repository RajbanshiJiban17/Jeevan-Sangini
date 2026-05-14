import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def process_pdf_to_vectorstore(data_source="data/"):

    try:

        all_documents = []

        if not os.path.exists(data_source):
            print("⚠️ Data folder not found")
            return None

        # ======================================
        # LOAD PDF FILES ONLY (OPTIMIZED)
        # ======================================

        for file in os.listdir(data_source):

            if not file.endswith(".pdf"):
                continue

            file_path = os.path.join(data_source, file)

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # clean text
                for d in docs:
                    d.page_content = d.page_content.replace("\n", " ").strip()

                all_documents.extend(docs)

            except Exception as e:
                print(f"PDF load error {file}: {e}")
                continue

        if len(all_documents) == 0:
            print("⚠️ No PDF documents found")
            return None

        # ======================================
        # SMART CHUNKING (OPTIMIZED FOR MEDICAL DATA)
        # ======================================

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )

        chunks = splitter.split_documents(all_documents)

        print(f"✅ PDF Loaded: {len(all_documents)} pages")
        print(f"✅ Total chunks: {len(chunks)}")

        # ======================================
        # LIGHTWEIGHT EMBEDDINGS (FAST + FREE)
        # ======================================

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ======================================
        # VECTOR STORE
        # ======================================

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        print("✅ Vector DB created successfully")

        return vectorstore

    except Exception as e:

        print("❌ Processor Error:", str(e))

        return None