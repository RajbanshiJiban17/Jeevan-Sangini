import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing")

        genai.configure(api_key=api_key)
        for m in genai.list_models():
         print(m.name)

        self.model = None

        try:
            # Stable working model
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash"
            )
            print("✅ Gemini model loaded successfully")

        except Exception as e:
            print(f"❌ Model Load Error: {e}")

    def ask(self, user_query, context, lang="नेपाली"):
        try:
            if not self.model:
                return "🚨 AI model load भएको छैन।"

            # Rate limit avoid
            time.sleep(1)

            prompt = f"""
तिमी 'जीवन-सङ्गलिनी' एआई हौ।

सन्दर्भ:
{context}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- सरल र बुझिने भाषा प्रयोग गर।
- यदि Hb ९.५ भन्दा कम छ भने 'उच्च जोखिम' चेतावनी देऊ।
- मेडिकल emergency भए तुरुन्त doctor सल्लाह सुझाव देऊ।

प्रश्न:
{user_query}
"""

            response = self.model.generate_content(prompt)

            return response.text

        except Exception as e:
            return f"🚨 एआईमा समस्या आयो: {str(e)}"