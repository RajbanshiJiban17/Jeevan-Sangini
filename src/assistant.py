import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing")
        genai.configure(api_key=api_key)
        
        # ४०४ एररबाट बच्न हामीले जनरेट गर्ने बेलामा मोडेल छान्छौँ
        self.model_ids = ['gemini-1.5-flash', 'gemini-pro']

    def ask(self, user_query, context="", lang="नेपाली"):
        # दुवै मोडेल पालैपालो ट्राइ गर्ने ताकि एरर नआओस्
        for model_id in self.model_ids:
            try:
                time.sleep(1) # Free Tier Rate Limit को लागि
                model = genai.GenerativeModel(model_id)
                
                prompt = f"""
                You are 'Jeevan-Sangini', a maternal health assistant.
                Context: {context[:1500]}
                Language: {lang}
                
                Instructions:
                1. Always advise consulting a doctor.
                2. If the user mentions Hb (Hemoglobin) < 9.5, warn as "🚨 High Risk Anemia".
                3. Use empathetic language.
                
                Question: {user_query}
                """
                
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if model_id == self.model_ids[-1]: # अन्तिम मोडेल पनि फेल भएमा मात्र एरर दिने
                    return f"एआई सेवामा समस्या आयो: {str(e)}"
                continue