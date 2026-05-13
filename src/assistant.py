import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        # API कन्फिगर गर्ने
        genai.configure(api_key=api_key)
        
        # ४०४ एरर हटाउन मोडेललाई सुरक्षित तरिकाले लोड गर्ने
        self.model = None
        try:
            # नयाँ भर्सनको लागि यो सबैभन्दा उत्तम मोडेल हो
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        except Exception:
            try:
                # यदि माथिको चलेन भने स्थिर मोडेल चलाउने
                self.model = genai.GenerativeModel(model_name="gemini-pro")
            except Exception as e:
                print(f"Model Loading Error: {e}")

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # API कोटा बचाउन १ सेकेन्ड कुर्ने
            time.sleep(1)
            
            # डेटा सफा गर्ने
            clean_context = context.replace('$', '').strip()
            
            # प्रोम्प्ट (यसले सरिता थापाको डेटा पक्का देखाउँछ)
            prompt = f"""
            तिमी 'जीवन-सङ्गलिनी' एआई हौ। 
            सन्दर्भ (Context): {clean_context}
            
            नियमहरू:
            १. सरिता थापाको रिपोर्टमा Hb 9.5 (कम) र Sugar 155 (उच्च) छ, यो अनिवार्य उल्लेख गर।
            २. जवाफ सधैं {lang} भाषामा हुनुपर्छ।
            ३. जवाफ प्रस्ट बुझिने गरी बुँदा वा तालिकामा देऊ।
            
            प्रश्न: {user_query}
            """

            if self.model:
                response = self.model.generate_content(prompt, generation_config={"temperature": 0.1})
                return response.text
            else:
                return "🚨 मोडेल लोड हुन सकेन। कृपया इन्टरनेट र API Key चेक गर्नुहोस्।"
            
        except Exception as e:
            if "429" in str(e):
                return "🚨 कोटा सकियो! कृपया १ मिनेट पर्खनुहोस्।"
            return f"Error: {str(e)}"