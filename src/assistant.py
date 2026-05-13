from google import genai
import time

class HealthAssistant:

    def __init__(self, api_key):

        if not api_key:
            raise ValueError("API Key missing")

        # NEW SDK client
        self.client = genai.Client(api_key=api_key)

        self.model = "gemini-1.5-flash"

        # OPTIONAL: list available models (debug only)
        try:
            models = self.client.models.list()
            for m in models:
                print(m.name)
        except Exception as e:
            print(f"Model list error: {e}")

        print("✅ Gemini client initialized")

    def ask(self, user_query, context="", lang="नेपाली"):

        try:

            time.sleep(1)

            prompt = f"""
तिमी 'जीवन-सङ्गलिनी' AI Health Assistant हौ।

सन्दर्भ:
{context}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- छोटो, स्पष्ट र professional जवाफ देऊ।
- Medical emergency भए तुरुन्त doctor सल्लाह देऊ।
- Hb 9.5 भन्दा कम भए anemia warning देऊ।
- PDF report analyze गर्दा values explain गर।

प्रश्न:
{user_query}
"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            return response.text if response else "⚠️ Empty response"

        except Exception as e:
            return f"🚨 AI Error: {str(e)}"