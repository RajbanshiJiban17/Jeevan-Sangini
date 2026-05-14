import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing!")
        
        # GenAI कन्फिगरेसन
        genai.configure(api_key=api_key)
        
        # उपलब्ध मोडेलहरू मध्ये सबैभन्दा राम्रो छान्ने (Error-free Logic)
        self.model_name = 'gemini-1.5-flash'
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except:
            self.model = genai.GenerativeModel('gemini-pro')

    def ask(self, user_query, context="", lang="नेपाली"):
        try:
            # Free Tier को 'Rate Limit' बाट बच्न १ सेकेन्ड विश्राम
            time.sleep(1)
            
            # प्रोफेशनल र सुरक्षित प्रोम्प्ट
            prompt = f"""
            भूमिका: तिमी 'जीवन-सङ्गलिनी' (Jeevan-Sangini) एआई स्वास्थ्य सहायक हौ।
            
            सन्दर्भ (Context): {context[:1500]}
            
            नियमहरू:
            १. सधैं {lang} भाषामा सरल र सहानुभूतिपूर्ण जवाफ देऊ।
            २. मेडिकल रिपोर्टमा Hb (Hemoglobin) ९.५ भन्दा कम भए "🚨 उच्च जोखिम (Anemia)" चेतावनी अनिवार्य देऊ।
            ३. सधैं "यो जानकारी डाक्टरको परामर्शको विकल्प होइन" भनी उल्लेख गर।
            
            प्रश्न: {user_query}
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"🚨 एआई सेवामा व्यस्तता आयो। कृपया १ मिनेटपछि प्रयास गर्नुहोस्। (Error: {str(e)})"