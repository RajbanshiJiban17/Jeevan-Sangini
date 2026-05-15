import os

# --- Ollama (local CPU; small model default) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# e2b = smaller, better for CPU / low RAM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# --- Gemini (Streamlit Cloud) ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# --- RAG ---
DATA_DIR = os.getenv("DATA_DIR", "data")
VECTOR_CACHE_DIR = os.getenv("VECTOR_CACHE_DIR", os.path.join(DATA_DIR, "vectorstore"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# auto | ollama | gemini
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
