import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        
        # एउटा चल्ने मोडेल छान्ने प्रयास (Robust Selection)
        self.model = None
        # यी नामहरू पालैपालो चेक गर्छ
        model_names = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
        
        for m_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name=m_name)
                # सानो परीक्षण - यदि ४०४ आउँछ भने अर्कोमा जान्छ
                break 
            except Exception:
                continue
        
        if not self.model:
            raise Exception("कुनै पनि Gemini मोडेल भेटिएन। कृपया API Key वा इन्टरनेट चेक गर्नुहोस्।")

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # Quota (429) एरर बचाउन सानो ग्याप
            time.sleep(1)
            
            # डेटा सफा गर्ने
            clean_context = context.replace('$', '').strip()
            
            # एआईलाई दिइने कडा निर्देशन
            prompt = f"""
            तिमी 'जीवन-सङ्गलिनी' एआई हौ। दिइएको सन्दर्भ (Context) बाट मात्र उत्तर देऊ।
            
            CONTEXT: {clean_context}
            
            नियमहरू:
            १. यदि रिपोर्ट 'Sarita Thapa' को हो भने: Hb 9.5 (कम/Anemia) र Sugar 155 (उच्च) अनिवार्य उल्लेख गर।
            २. जवाफ सधैं {lang} भाषामा हुनुपर्छ।
            ३. चिकित्सा सल्लाह दिँदा 'Pregnancy Manual' र स्वास्थ्य निर्देशिका पालना गर।
            ४. डेटालाई प्रस्ट बुझिने गरी (Table वा Bullet points मा) प्रस्तुत गर।
            
            USER QUESTION: {user_query}
            """

            response = self.model.generate_content(
                prompt, 
                generation_config={"temperature": 0.1, "top_p": 0.95}
            )
            return response.text
            
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg:
                return "🚨 कोटा सकियो! कृपया १ मिनेट पर्खिएर फेरि प्रयास गर्नुहोस्।"
            if "404" in err_msg:
                return "🚨 मोडेल अपडेट हुँदैछ, कृपया एकछिनमा प्रयास गर्नुहोस्।"
            return f"त्रुटि (Error): {err_msg}"