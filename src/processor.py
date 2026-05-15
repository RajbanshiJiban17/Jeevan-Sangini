import hashlib
import os

from src.config import DATA_DIR, EMBEDDING_MODEL, VECTOR_CACHE_DIR


def _folder_fingerprint(data_source: str) -> str:
    if not os.path.isdir(data_source):
        return ""
    parts: list[str] = []
    for name in sorted(os.listdir(data_source)):
        if name.lower().endswith(".pdf"):
            path = os.path.join(data_source, name)
            parts.append(f"{name}:{os.path.getmtime(path)}:{os.path.getsize(path)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest() if parts else ""


def process_pdf_to_vectorstore(data_source: str | None = None, force_rebuild: bool = False):
    """Build FAISS index from PDFs. Returns None if RAG deps missing or no PDFs."""
    # CPU-only (no CUDA required)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "RAG को लागि: pip install -r requirements-local.txt"
        ) from exc

    data_source = data_source or DATA_DIR
    if not os.path.exists(data_source):
        os.makedirs(data_source, exist_ok=True)
        return None

    pdfs = [f for f in os.listdir(data_source) if f.lower().endswith(".pdf")]
    if not pdfs:
        return None

    fingerprint = _folder_fingerprint(data_source)
    cache_dir = VECTOR_CACHE_DIR
    meta_path = os.path.join(cache_dir, "fingerprint.txt")
    index_path = os.path.join(cache_dir, "index.faiss")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    if (
        not force_rebuild
        and fingerprint
        and os.path.isfile(index_path)
        and os.path.isfile(meta_path)
    ):
        try:
            with open(meta_path, encoding="utf-8") as f:
                if f.read().strip() == fingerprint:
                    return FAISS.load_local(
                        cache_dir,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
        except Exception as e:
            print(f"Cache load failed: {e}")

    all_docs = []
    for file in pdfs:
        try:
            loader = PyPDFLoader(os.path.join(data_source, file))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"PDF error {file}: {e}")

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(cache_dir, exist_ok=True)
    db.save_local(cache_dir)
    if fingerprint:
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(fingerprint)
    return db
