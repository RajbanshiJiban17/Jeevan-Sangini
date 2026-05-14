import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def process_pdf_to_vectorstore(data_source="data/"):

    all_docs = []

    if not os.path.exists(data_source):
        return None

    for file in os.listdir(data_source):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(data_source, file)

        try:
            loader = PyPDFLoader(path)
            docs = loader.load()

            for d in docs:
                d.page_content = d.page_content.replace("\n", " ")

            all_docs.extend(docs)

        except:
            continue

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db