"""Run: python -m src.check_models"""

from src.config import GEMINI_MODEL, OLLAMA_MODEL
from src.ollama_client import is_ollama_running, list_models, model_available
from src.runtime import get_gemini_api_key, resolve_backend


def main():
    print("Jeevan Sangini — system check\n")
    ollama = is_ollama_running()
    gemini = bool(get_gemini_api_key())
    backend = resolve_backend(ollama)

    print(f"Ollama running: {ollama}")
    print(f"Gemini key set: {gemini}")
    print(f"Active backend: {backend}\n")

    if ollama:
        print("Ollama models:")
        for n in list_models():
            print(f"  - {n}")
        if model_available(OLLAMA_MODEL):
            print(f"\n✅ CPU model '{OLLAMA_MODEL}' ready")
        else:
            print(f"\n⚠️  Run: ollama pull {OLLAMA_MODEL}")

    if gemini:
        print(f"\n✅ Gemini configured ({GEMINI_MODEL})")
    elif not ollama:
        print("\n❌ Set GOOGLE_API_KEY or start Ollama")


if __name__ == "__main__":
    main()
