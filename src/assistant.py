import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = None
        # ४०४ एरर हटाउन Fallback लोजिक
        model_names = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]
        for m_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name=m_name)
                break
            except:
                continue

    def ask(self, user_query, context, lang="नेपाली"):
        try:
            time.sleep(1) # Quota बचाउन सानो Delay
            clean_context = context.replace('$', '').strip()
            
            prompt = f"""
            तिमी 'जीवन-सङ्गलिनी' एआई हौ। मेडिकल रिपोर्टको आधारमा सल्लाह देऊ।
            
            सन्दर्भ (Context): {clean_context}
            
            नियमहरू:
            १. रिपोर्टबाट बिरामीको नाम पत्ता लगाऊ।
            २. यदि Hb १० भन्दा कम छ वा सुगर १४० भन्दा धेरै छ भने 'High Risk' चेतावनी देऊ।
            ३. सधैं {lang} भाषामा बुँदागत जवाफ देऊ।
            
            प्रश्न वा डेटा: {user_query}
            """
            
            if self.model:
                response = self.model.generate_content(prompt, generation_config={"temperature": 0.1})
                return response.text
            return "🚨 मोडेल लोड हुन सकेन।"
        except Exception as e:
            return f"Error: {str(e)}"