from google import genai
import time

class HealthAssistant:

    def __init__(self, api_key):

        if not api_key:
            raise ValueError("API Key missing")

        # NEW Gemini client
        self.client = genai.Client(api_key=api_key)

        # ✅ YOUR BEST AVAILABLE MODEL
        self.models = [
                      "models/gemini-2.0-flash",
                      "models/gemini-2.0-flash-lite"]

        # Optional: list models (debug only)
        try:
            models = self.client.models.list()
            print("📌 Available Models:")
            for m in models:
                print(" -", m.name)
        except Exception as e:
            print(f"Model list error: {e}")

        print("✅ Gemini client initialized successfully")

    def ask(self, user_query, context="", lang="नेपाली"):

        try:
            time.sleep(1)

            prompt = f"""
तिमी 'जीवन-सङ्गलिनी' AI Health Assistant हौ।

सन्दर्भ:
{context}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- सरल, स्पष्ट र professional जवाफ देऊ।
- Medical emergency भए तुरुन्त doctor सल्लाह देऊ।
- Hb 9.5 भन्दा कम भए anemia warning देऊ।
- Lab report वा PDF analysis गर्दा values explain गर।

प्रश्न:
{user_query}
"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            if response and hasattr(response, "text"):
                return response.text

            return "⚠️ Empty response"

        except Exception as e:
            return f"🚨 AI Error: {str(e)}"