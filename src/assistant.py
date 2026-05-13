import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        try:
             self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        except:
             self.model = genai.GenerativeModel(model_name="gemini-pro")

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # API Rate Limit बचाउन १ सेकेन्डको ग्याप
            time.sleep(1)
            
            clean_context = context.replace('$', '').strip()
            
            prompt = f"""
            तिमी 'जीवन-सङ्गलिनी' एआई हौ। दिइएको सन्दर्भ (Context) र डाटाबाट मात्र जवाफ देऊ।
            CONTEXT: {clean_context}
            
            नियमहरू:
            १. यदि 'Sarita Thapa' को रिपोर्ट हो भने Hb 9.5 (कम/Anemia) र Sugar 155 (उच्च) स्पष्ट देखाऊ।
            २. चिकित्सा सल्लाह दिँदा 'Pregnancy Manual' को पालना गर।
            ३. सधैं {lang} भाषामा जवाफ देऊ।
            
            USER QUESTION: {user_query}
            """

            response = self.model.generate_content(prompt, generation_config={"temperature": 0.1})
            return response.text
        except Exception as e:
            if "429" in str(e):
                return "🚨 कोटा सकियो! कृपया ३० सेकेन्ड पर्खिएर फेरि प्रयास गर्नुहोस्।"
            return f"Error: {str(e)}"