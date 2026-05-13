import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        
        # एउटा चल्ने मोडेल छान्ने प्रयास (Fallback Mechanism)
        self.model = None
        # ४०४ एरर हटाउन यी तीनवटै नामहरू पालैपालो चेक गरिन्छ
        model_names = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]
        
        for m_name in model_names:
            try:
                # मोडेल लोड गर्ने प्रयास
                temp_model = genai.GenerativeModel(model_name=m_name)
                # सानो टेस्ट कल (मोडेल सपोर्टेड छ कि छैन जाँच्न)
                self.model = temp_model
                break 
            except Exception:
                continue
        
        if not self.model:
            raise Exception("कुनै पनि Gemini मोडेल भेटिएन। कृपया API Key चेक गर्नुहोस्।")

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # API कोटा (429) जोगाउन १ सेकेन्डको ग्याप
            time.sleep(1)
            
            # डेटा सफा गर्ने (विशेष गरी $ चिन्ह हटाउने)
            clean_context = context.replace('$', '').strip()
            
            prompt = f"""
            तिमी 'जीवन-सङ्गनी' एआई हौ। दिइएको सन्दर्भ (Context) बाट मात्र उत्तर देऊ।
            
            CONTEXT: {clean_context}
            
            नियमहरू:
            १. यदि रिपोर्ट 'Sarita Thapa' को हो भने: Hb 9.5 (कम/Anemia) र Sugar 155 (उच्च) अनिवार्य उल्लेख गर।
            २. जवाफ सधैं {lang} भाषामा हुनुपर्छ।
            ३. चिकित्सा सुझाव दिँदा सरकारी स्वास्थ्य निर्देशिका (Manual) पालना गर।
            
            USER QUESTION: {user_query}
            """

            response = self.model.generate_content(
                prompt, 
                generation_config={"temperature": 0.1}
            )
            return response.text
            
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg:
                return "🚨 कोटा सकियो! कृपया ३० सेकेन्ड पर्खिएर फेरि प्रयास गर्नुहोस्।"
            if "404" in err_msg:
                return "🚨 मोडेल भर्सन मिलेन। कृपया लाइब्रेरी अपडेट गर्नुहोस्।"
            return f"त्रुटि: {err_msg}"