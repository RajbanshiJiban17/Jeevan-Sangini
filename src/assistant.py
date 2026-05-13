import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash") # Stable for reports

    def get_system_prompt(self, lang, mode):
        if lang == "English":
            base = "You are 'Jeevan-Sangini' AI. Start with empathy."
            report = "EXTRACT: Hb, Sugar, Pus Cells, Albumin. Table is mandatory. Use Reference Ranges."
        else:
            base = "तपाईँ 'जीवन-सङ्गलिनी' एआई हो। सुरुमा सहानुभूति प्रकट गर्नुहोस्।"
            report = "अनिवार्य रूपमा Hb, सुगर, पिसाबमा Pus Cells र Albumin को डेटा निकाल्नुहोस्। तालिका देखाउनुहोस्।"
        return f"{base} {report}" if mode == "report" else f"{base} Answer based on Nepal Health Manuals."

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # $ हटाउने र डेटा सफा गर्ने
            clean_context = context.replace('$', '').replace('\n', ' ')
            system_prompt = self.get_system_prompt(lang, mode)
            
            full_prompt = f"""
            SYSTEM: {system_prompt}
            CONTEXT: {clean_context}
            TASK: {user_query}
            STRICT RULES:
            1. Create Markdown Table: Test Name | Result | Normal Range | Status.
            2. If Hb < 11, Status = ⚠️ Low (Anemia).
            3. If Sugar > 140, Status = ⚠️ High.
            4. If Pus Cells > 3, Status = ⚠️ Infection (UTI).
            5. Respond in {lang}.
            """
            response = self.model.generate_content(full_prompt, generation_config={"temperature": 0.0})
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"