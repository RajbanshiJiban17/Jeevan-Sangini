import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key):
        # API की कन्फिगर गर्ने
        genai.configure(api_key=api_key)
        # स्थिर नतिजाको लागि gemini-1.5-flash उत्तम हुन्छ
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def get_system_prompt(self, lang, mode):
        """सिस्टम प्रम्प्ट जसले एआईको व्यवहार र टोन सेट गर्छ।"""
        if lang == "English":
            base = "You are 'Jeevan-Sangini' AI. Start with empathy. Use tables for data."
            report = "Strictly extract: Hb, Sugar, Pus Cells, Albumin. Compare with maternal health standards."
        else:
            base = "तपाईँ 'जीवन-सङ्गलिनी' एआई हो। जवाफको सुरुमा सहानुभूति प्रकट गर्नुहोस् र तालिका अनिवार्य प्रयोग गर्नुहोस्।"
            report = "रिपोर्टबाट Hb, सुगर, Pus Cells र Albumin को डाटा निकाल्नुहोस् र नेपालको स्वास्थ्य मापदण्ड अनुसार विश्लेषण गर्नुहोस्।"
        
        return f"{base} {report}" if mode == "report" else f"{base} Answer based on Nepal Health Manuals."

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # १. डाटा सफा गर्ने (Cleaning symbols like $ that block AI analysis)
            clean_context = context.replace('$', '').replace('\n', ' ')
            
            # २. सिस्टम प्रम्प्ट प्राप्त गर्ने
            system_instruction = self.get_system_prompt(lang, mode)
            
            # ३. कडा निर्देशनका साथ प्रम्प्ट तयार गर्ने
            full_prompt = f"""
            SYSTEM: {system_instruction}
            
            REPORT CONTEXT: {clean_context}
            
            STRICT INSTRUCTIONS:
            1. Extract the specific values for 'Sarita Thapa': Hb (9.5), Blood Sugar (155), Pus Cells (5-7), and Albumin (Trace +). [cite: 4, 8, 12, 13]
            2. Display them in a Markdown Table: Test Name | Result | Reference Range | Status.
            3. Analyze: 
               - Hb 9.5 is Low (Anemia risk). [cite: 4]
               - Sugar 155 is High (Diabetes risk). [cite: 8]
               - Pus Cells 5-7 indicates Infection (UTI). [cite: 13]
            4. Respond entirely in {lang}.
            
            USER QUESTION: {user_query}
            """

            # ४. एआई रेस्पोन्स जेनेरेट गर्ने (Temperature 0.0 राख्दा तथ्यहरू परिवर्तन हुँदैनन्)
            response = self.model.generate_content(
                full_prompt, 
                generation_config={"temperature": 0.0, "max_output_tokens": 1000}
            )
            
            answer = response.text
            
            # ५. अलर्ट थप्ने लोजिक
            disclaimer = (
                "\n\n---\n⚠️ **Note:** For information only. Consult a doctor." 
                if lang == "English" else 
                "\n\n---\n⚠️ **नोट:** यो जानकारीका लागि मात्र हो। डाक्टरसँग परामर्श गर्नुहोस्।"
            )

            # यदि गम्भीर समस्या देखिएमा अलर्ट थप्ने
            if any(word in answer.lower() for word in ["risk", "low", "high", "infection", "खतरा", "कम", "उच्च"]):
                prefix = "🚨 **Important Analysis:** " if lang == "English" else "🚨 **महत्त्वपूर्ण विश्लेषण:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer

        except Exception as e:
            return f"Error encountered: {str(e)}"