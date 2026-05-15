"""Run: python -m src.check_models"""

from src.config import OLLAMA_MODEL
from src.ollama_client import is_ollama_running, list_models, model_available


def main():
    print("Jeevan Sangini — Ollama check\n")
    if not is_ollama_running():
        print("❌ Ollama is not running.")
        print("   Start: ollama serve")
        print(f"   Then:  ollama pull {OLLAMA_MODEL}")
        return

    print("✅ Ollama is running.\nInstalled models:")
    for name in list_models():
        print(f"  - {name}")

    if model_available(OLLAMA_MODEL):
        print(f"\n✅ Configured model '{OLLAMA_MODEL}' is available.")
    else:
        print(f"\n⚠️  Model '{OLLAMA_MODEL}' not found.")
        print(f"   Run: ollama pull {OLLAMA_MODEL}")


if __name__ == "__main__":
    main()
