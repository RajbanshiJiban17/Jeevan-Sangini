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

                file_path = os.path.join(data_source, file)

                # PDF files
                if file.endswith(".pdf"):

                    loader = PyPDFLoader(file_path)

                    docs = loader.load()

                    all_documents.extend(docs)

                # Parquet files
                elif file.endswith(".parquet"):

                    df = pd.read_parquet(file_path)

                    for _, row in df.iterrows():

                        content = f"""
Question:
{row.get('instruction', '')}

Answer:
{row.get('output', '')}
"""

                        all_documents.append(
                            Document(page_content=content)
                        )

        # Clean text
        for doc in all_documents:

            doc.page_content = (
                doc.page_content
                .replace("\n", " ")
                .replace("$", "")
                .strip()
            )

        if not all_documents:
            print("⚠️ No documents found")
            return None

        # Split chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(all_documents)

        print(f"✅ Total chunks: {len(chunks)}")

        # Local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        print("✅ Vector DB ready")

        return vectorstore

    except Exception as e:

        print(f"❌ Processor Error: {e}")

        return None