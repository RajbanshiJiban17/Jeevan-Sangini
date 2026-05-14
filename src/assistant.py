from google import genai
import time

class HealthAssistant:

    def __init__(self, api_key):

        self.client = genai.Client(api_key=api_key)

        # quota-safe models
        self.models = [
            "models/gemini-2.0-flash-lite",
            "models/gemini-2.0-flash"
        ]

    def ask(self, user_query, context="", lang="नेपाली"):

        prompt = f"""
तिमी 'जीवन-सङ्गलिनी' AI Health Assistant हौ।

सन्दर्भ:
{context[:1500]}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- छोटो, स्पष्ट जवाफ देऊ।
- emergency भए doctor सल्लाह देऊ।
- Hb 9.5 भन्दा कम भए anemia warning देऊ।

प्रश्न:
{user_query}
"""

        for model in self.models:

            try:

                print(f"Trying: {model}")

                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response and response.text:
                    return response.text

            except Exception as e:

                print(f"Model Failed: {model}")
                print(e)

                if "429" in str(e):

                    return """
🚨 AI quota temporarily सकिएको छ।

⏳ केही समयपछि फेरि प्रयास गर्नुहोस्।
"""

                continue

        return "🚨 अहिले AI response उपलब्ध छैन।"