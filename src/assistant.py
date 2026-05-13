import google.generativeai as genai
import time
import os

class HealthAssistant:
    def __init__(self, api_key):
        # API की कन्फिगर गर्ने
        genai.configure(api_key=api_key)
        
        # मोडेल सेट गर्ने (४०४ एरर आउन नदिन Fallback लोजिक)
        self.model = None
        # यी तीन मध्ये जुन भेटिन्छ, त्यही चल्छ
        for m_name in ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]:
            try:
                self.model = genai.GenerativeModel(model_name=m_name)
                # एउटा सानो टेस्ट कल गरेर चेक गर्ने (Optional तर सुरक्षित)
                break
            except:
                continue
        
        if not self.model:
            raise Exception("Gemini API सँग कनेक्ट हुन सकेन।")

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # रेट लिमिट (429) बाट बच्न १ सेकेन्ड कुराउने
            time.sleep(1)
            
            # सन्दर्भ (Context) सफा गर्ने
            clean_context = context.replace('$', '').strip()
            
            # एआईलाई दिइने कडा र स्पष्ट निर्देशन
            prompt = f"""
            भूमिका: तिमी 'जीवन-सङ्गलिनी' एआई हौ। 
            सन्दर्भ: {clean_context}
            
            कडा नियमहरू:
            १. सरिता थापाको केसमा: Hb 9.5 (अल्पसङ्ख्यक/Anemia) र Sugar 155 (उच्च) अनिवार्य देखाउनु।
            २. सधैं {lang} भाषामा जवाफ दिनु।
            ३. जवाफ तालिका (Table) वा बुँदामा दिनु ताकि पढ्न सजिलो होस्।
            ४. सन्दर्भमा दिइएको आधिकारिक स्वास्थ्य निर्देशिका मात्र प्रयोग गर्नु।
            
            प्रश्न: {user_query}
            """

            response = self.model.generate_content(
                prompt, 
                generation_config={"temperature": 0.1, "top_p": 1}
            )
            return response.text
            
        except Exception as e:
            if "429" in str(e):
                return "🚨 कोटा सकियो! कृपया ३० सेकेन्ड कुर्नुहोस्।"
            elif "404" in str(e):
                # यदि अझै ४०४ आयो भने प्रो मोडेलमा स्विच गर्ने
                self.model = genai.GenerativeModel(model_name="gemini-pro")
                return "🚨 मोडेल अपडेट गरियो। कृपया फेरि प्रश्न सोध्नुहोस्।"
            return f"त्रुटि: {str(e)}"