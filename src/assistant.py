from transformers import pipeline
import time

class HealthAssistant:

    def __init__(self):

        print("⏳ Loading Gemma model...")

        self.model = pipeline(
            "text-generation",
            model="google/flan-t5-small",
            device_map="auto"
        )

        print("✅ Gemma loaded successfully")

    def ask(self, user_query, context="", lang="नेपाली"):

        try:
            time.sleep(0.3)

            prompt = f"""
तिमी 'जीवन-संगिनी' AI Health Assistant हौ।

सन्दर्भ:
{context[:2000]}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ
- सरल र medical safe उत्तर देऊ
- emergency भए doctor suggest गर
- Hb 9.5 भन्दा कम भए anemia warning देऊ

प्रश्न:
{user_query}

उत्तर:
"""

            result = self.model(
                prompt,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True
            )

            return result[0]["generated_text"]

        except Exception as e:
            return f"🚨 Gemma Error: {str(e)}"