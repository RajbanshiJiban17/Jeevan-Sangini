from google import genai
import time

class HealthAssistant:
    def __init__(self, api_key):

        if not api_key:
            raise ValueError("API Key missing")

        genai.configure(api_key=api_key)
        for m in genai.list_models():
            print(m.name)

        self.model = None

        try:
            # Available stable Gemini model
            self.model = genai.GenerativeModel(
                model_name="models/gemini-1.5-flash"
            )

            print("✅ Gemini model loaded successfully")

        except Exception as e:
            print(f"❌ Model Error: {e}")

    def ask(self, user_query, context="", lang="नेपाली"):

        if not self.model:
            return "🚨 Gemini model load भएन।"

        try:

            time.sleep(1)

            prompt = f"""
तिमी 'जीवन-सङ्गलिनी' AI Health Assistant हौ।

सन्दर्भ:
{context}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- छोटो, स्पष्ट र professional जवाफ देऊ।
- मेडिकल emergency भए तुरुन्त doctor सल्लाह देऊ।
- Hb 9.5 भन्दा कम भए anemia warning देऊ।
- PDF report analyse गर्दा important values explain गर।

प्रश्न:
{user_query}
"""

            response = self.model.generate_content(prompt)

            if response and hasattr(response, "text"):
                return response.text

            return "⚠️ Response empty आयो"

        except Exception as e:
            return f"🚨 AI Error: {str(e)}"