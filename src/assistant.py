from google import genai
import time

class HealthAssistant:

    def __init__(self, api_key):

        if not api_key:
            raise ValueError("API Key missing")

        self.client = genai.Client(api_key=api_key)

        # fallback models
        self.models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]

        print("✅ Gemini client initialized successfully")

    def ask(self, user_query, context="", lang="नेपाली"):

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

        # fallback system
        for model in self.models:
            try:
                time.sleep(1)

                response = self.client.models.generate_content(
                    model=model,   # ✅ FIXED HERE
                    contents=prompt
                )

                if response and hasattr(response, "text"):
                    return response.text

            except Exception as e:
                print(f"Model failed {model}: {e}")
                continue

        return "🚨 AI Error: सबै models fail भए"