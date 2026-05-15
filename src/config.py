import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

DATA_DIR = os.getenv("DATA_DIR", "data")
VECTOR_CACHE_DIR = os.getenv("VECTOR_CACHE_DIR", os.path.join(DATA_DIR, "vectorstore"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
