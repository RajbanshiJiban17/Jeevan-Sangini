import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key खाली छ! कृपया .env फाइल चेक गर्नुहोस्।")
        
        genai.configure(api_key=api_key)
        
        # ४०४ एरर हटाउन सिधै यो नाम मात्र प्रयोग गर्नुहोस्
        # धेरै नामहरू चेक गर्दा पनि झुक्किन सक्छ, त्यसैले 'gemini-1.5-flash' बेस्ट छ
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            # यदि फ्ल्यास चलेन भने प्रो (Pro) ट्राई गर्ने
            self.model = genai.GenerativeModel('gemini-pro')

    def ask(self, user_query, context, lang="नेपाली"):
        try:
            time.sleep(1) # Rate Limit बचाउन
            
            # प्रोम्प्टलाई एकदम सफा बनाउने
            prompt = f"""
            भूमिका: तिमी 'जीवन-सङ्गलिनी' एआई हौ। 
            सन्दर्भ: {context}
            नियम: सधैं {lang} भाषामा जवाफ देऊ। यदि रिपोर्टमा Hb ९.५ छ भने 'उच्च जोखिम' चेतावनी देऊ।
            
            प्रश्न: {user_query}
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"एआईमा समस्या आयो: {str(e)}"