from transformers import pipeline
import time

class HealthAssistant:

    def __init__(self):

        # ✅ GEMMA LOCAL MODEL (NO API NEEDED)
        self.model = pipeline(
            "text-generation",
            model="google/gemma-2b-it",
            device_map="auto"
        )

        print("✅ Gemma model loaded successfully")

    def ask(self, user_query, context="", lang="नेपाली"):

        try:
            time.sleep(0.3)

            prompt = f"""
तिमी 'जीवन-संगिनी' AI Health Assistant हौ।

सन्दर्भ:
{context[:2000]}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- सरल, स्पष्ट र helpful जवाफ देऊ।
- medical emergency भए तुरुन्त doctor सल्लाह देऊ।
- Hb 9.5 भन्दा कम भए anemia warning देऊ।
- Nepal को maternal health focus गर।

प्रश्न:
{user_query}

उत्तर:
"""

            response = self.model(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )

            # clean output
            return response[0]["generated_text"]

        except Exception as e:
            return f"🚨 Gemma Error: {str(e)}"