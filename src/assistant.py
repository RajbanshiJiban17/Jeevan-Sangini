from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time


class HealthAssistant:

    def __init__(self):

        print("⏳ Loading AI model...")

        model_name = "google/flan-t5-small"

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print("✅ AI model loaded successfully")

    def ask(self, user_query, context="", lang="नेपाली"):

        try:

            time.sleep(0.2)

            prompt = f"""
तिमी 'जीवन-संगिनी' AI Health Assistant हौ।

सन्दर्भ:
{context[:1500]}

नियम:
- सधैं {lang} भाषामा जवाफ देऊ
- सरल र safe medical उत्तर देऊ
- emergency भए doctor सल्लाह देऊ
- Hb 9.5 भन्दा कम भए anemia warning देऊ

प्रश्न:
{user_query}

उत्तर:
"""

            # tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )

            # decode
            answer = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return answer

        except Exception as e:

            print("AI ERROR:", e)

            return f"🚨 AI Error: {str(e)}"