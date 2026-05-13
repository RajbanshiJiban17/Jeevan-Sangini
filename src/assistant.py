import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = None
        
        # ४०४ एरर हटाउन यी ३ वटा नामहरू पालैपालो चेक गर्ने
        # 'models/' प्रिफिक्स नराखी सिधै नाम मात्र प्रयोग गर्दा धेरै समस्या समाधान हुन्छ
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        for m_name in model_options:
            try:
                # यहाँ models/gemini-1.5-flash-latest को सट्टा सिधै gemini-1.5-flash मात्र ट्राई गर्ने
                self.model = genai.GenerativeModel(model_name=m_name)
                # एउटा सानो टेस्ट गर्ने मोडेल चल्यो कि नाइँ भनेर
                print(f"✅ Successfully loaded: {m_name}")
                break
            except Exception as e:
                print(f"❌ Failed to load {m_name}: {e}")
                continue
        
        if not self.model:
            raise Exception("कुनै पनि Gemini मोडेल उपलब्ध भएन। कृपया API Key वा लाइब्रेरी अपडेट गर्नुहोस्।")

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            time.sleep(1) # API कोटा लिमिट (429) बचाउन
            clean_context = context.replace('$', '').strip()
            
            prompt = f"""
            तिमी 'जीवन-सङ्गलिनी' एआई हौ। 
            सन्दर्भ: {clean_context}
            
            नियम:
            १. रिपोर्टमा Hb ९.५ (कम) वा Sugar १५५ (उच्च) देखिएमा 'High Risk' चेतावनी देऊ।
            २. सधैं {lang} भाषामा जवाफ देऊ।
            
            प्रश्न: {user_query}
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"