from transformers import pipeline
import time


class HealthAssistant:

    def __init__(self):

        print("⏳ Loading AI model...")

        # ✅ Streamlit Cloud safe model
        self.model = pipeline(
            task="text2text-generation",
            model="google/flan-t5-small"
        )

        print("✅ AI model loaded successfully")

    def ask(self, user_query, context="", lang="नेपाली"):

        try:

            time.sleep(0.2)

            # ✅ context limit
            safe_context = context[:1000]

            # ✅ prompt
            prompt = f"""
तिमी 'जीवन-संगिनी' AI Health Assistant हौ।

नियम:
- सधैं {lang} भाषामा जवाफ देऊ।
- छोटो, स्पष्ट र helpful जवाफ देऊ।
- Medical emergency भए तुरुन्त doctor सल्लाह देऊ।
- Hb 9.5 भन्दा कम भए anemia warning देऊ।
- Nepal को maternal health focus गर।

सन्दर्भ:
{safe_context}

प्रश्न:
{user_query}

उत्तर:
"""

            # ✅ generate response
            result = self.model(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.5
            )

            # ✅ clean output
            answer = result[0]["generated_text"]

            answer = answer.replace(prompt, "").strip()

            # empty safety
            if not answer:
                return "⚠️ उत्तर generate गर्न सकेन।"

            return answer

        except Exception as e:

            print("Assistant Error:", e)

            return f"🚨 AI Error: {str(e)}"