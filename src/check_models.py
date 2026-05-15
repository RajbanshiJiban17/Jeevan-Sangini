"""Run: python -m src.check_models"""

from src.config import GEMINI_MODEL, OLLAMA_MODEL,OLLAMA_BASE_URL
from src.ollama_client import is_ollama_running, list_models, model_available
from src.runtime import get_gemini_api_key, resolve_backend


def main():
    print("Jeevan Sangini — system check\n")
    print(f"DEBUG: Looking for Ollama at: {OLLAMA_BASE_URL}")
    
    # 1. Ollama को स्थिति चेक गर्दा BASE_URL पठाउनुहोस्
    # पक्का गर्नुहोस् कि तपाइँको is_ollama_running ले base_url एर्ग्युमेन्ट लिन्छ
    ollama = is_ollama_running(base_url=OLLAMA_BASE_URL) 
    
    gemini = bool(get_gemini_api_key())
    backend = resolve_backend(ollama)

    print(f"Ollama Base URL: {OLLAMA_BASE_URL}") # यो थप्नुहोस् ताकि कुन URL प्रयोग भइरहेको छ थाहा होस्
    print(f"Ollama running: {ollama}")
    print(f"Gemini key set: {gemini}")
    print(f"Active backend: {backend}\n")

    if ollama:
        print("Ollama models:")
        # 2. मोडेल लिस्ट गर्दा पनि BASE_URL पठाउनुहोस्
        models = list_models(base_url=OLLAMA_BASE_URL)
        for n in models:
            print(f"   - {n}")
        
        if model_available(OLLAMA_MODEL, base_url=OLLAMA_BASE_URL):
            print(f"\n✅ CPU model '{OLLAMA_MODEL}' ready")
        else:
            print(f"\n⚠️  Model '{OLLAMA_MODEL}' not found on server.")
            print(f"Run: ollama pull {OLLAMA_MODEL} on your local machine.")

    if gemini:
        print(f"\n✅ Gemini configured ({GEMINI_MODEL})")
    elif not ollama:
        print("\n❌ System Offline: Set GOOGLE_API_KEY or ensure Ngrok + Ollama are running.")

if __name__ == "__main__":
    main()
