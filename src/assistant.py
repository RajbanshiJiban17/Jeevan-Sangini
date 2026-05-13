import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing")
        
        genai.configure(api_key=api_key)
        
        # ४०४ एरर हटाउन सिधै मोडेलको नाम मात्र प्रयोग गर्ने
        # 'models/' प्रिफिक्स नराख्दा यो एरर हराउँछ
        try:
            # १.५ फ्ल्यास ट्राय गर्ने
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except:
            try:
                # फ्ल्यास चलेन भने प्रो (Pro) ट्राय गर्ने
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"Model Load Error: {e}")

    def ask(self, user_query, context, lang="नेपाली"):
        try:
            # Rate Limit (429) एररबाट बच्न
            time.sleep(1)
            
            # प्रोम्प्ट सफा र प्रस्ट बनाउने
            prompt = f"""
            तिमी 'जीवन-सङ्गलिनी' एआई हौ। 
            सन्दर्भ: {context}
            नियम: सधैं {lang} भाषामा जवाफ देऊ। यदि Hb ९.५ छ भने 'उच्च जोखिम' भनी चेतावनी देऊ।
            
            प्रश्न: {user_query}
            """
            
            if self.model:
                # यहाँ सिधै जेनेरेट गर्ने
                response = self.model.generate_content(prompt)
                return response.text
            return "🚨 मोडेल लोड हुन सकेन। कृपया लाइब्रेरी अपडेट गर्नुहोस्।"
            
        except Exception as e:
            # यदि ४०४ एरर अझै आयो भने यहाँबाट थाहा हुन्छ
            return f"एआईमा समस्या आयो: {str(e)}"