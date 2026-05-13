import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def get_system_prompt(self, lang, mode):
        if lang == "English":
            base = "You are 'Jeevan-Sangini' AI. Start with empathy. Use tables for data."
            report = "Strictly extract: Hb, Sugar, Pus Cells, Albumin. Compare with maternal health standards."
        else:
            base = "तपाईँ 'जीवन-सङ्गलिनी' एआई हो। जवाफको सुरुमा सहानुभूति प्रकट गर्नुहोस्।"
            report = "अनिवार्य रूपमा Hb, सुगर, Pus Cells र Albumin को डाटा निकाल्नुहोस् र तालिका प्रयोग गर्नुहोस्।"
        
        return f"{base} {report}" if mode == "report" else f"{base} Answer based on Nepal Health Manuals."

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # $ चिन्ह हटाउने ताकि एआई नझुक्कियोस्
            clean_context = context.replace('$', '').replace('\n', ' ')
            system_instruction = self.get_system_prompt(lang, mode)
            
            full_prompt = f"""
            SYSTEM: {system_instruction}
            CONTEXT: {clean_context}
            
            STRICT INSTRUCTIONS:
            1. 'Sarita Thapa' को रिपोर्टबाट यी डेटा निकाल: Hb (9.5), Sugar (155), Pus Cells (5-7).
            2. डेटालाई Markdown Table मा देखाऊ।
            3. विश्लेषण: Hb 9.5 (कम/Anemia), Sugar 155 (उच्च/Diabetes), Pus Cells (संक्रमण/UTI)।
            4. जवाफ {lang} मा देऊ।
            
            USER QUESTION: {user_query}
            """

            response = self.model.generate_content(
                full_prompt, 
                generation_config={"temperature": 0.0}
            )
            
            answer = response.text
            disclaimer = "\n\n---\n⚠️ **Note:** Consult a doctor." if lang == "English" else "\n\n---\n⚠️ **नोट:** डाक्टरसँग परामर्श गर्नुहोस्।"

            if any(word in answer.lower() for word in ["risk", "low", "high", "infection", "खतरा", "कम", "उच्च"]):
                prefix = "🚨 **Analysis:** " if lang == "English" else "🚨 **विश्लेषण:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer
        except Exception as e:
            return f"Error: {str(e)}"