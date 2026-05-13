import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        try:
            # Gemma 2 वा Gemini 1.5 Flash छनोट गर्ने
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if "models/gemma-2-9b-it" in available_models:
                selected_model = "models/gemma-2-9b-it"
            else:
                selected_model = "models/gemini-1.5-flash"
            self.model = genai.GenerativeModel(model_name=selected_model)
            self.active_model = selected_model
        except Exception as e:
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            self.active_model = "gemini-1.5-flash"

    def get_system_prompt(self, lang, mode):
        """Strict Guidance for Gemma/Gemini"""
        model_identity = "Gemma 2"
        if lang == "English":
            base = f"You are 'Jeevan-Sangini' AI. Start with empathy. Use tables."
            report = "EXTRACT and COMPARE: Hb < 11 (Anemia), Sugar > 140 (High), Pus > 3 (UTI). If danger, start with 🚨 EMERGENCY ALERT."
        else:
            base = f"तपाईँ 'जीवन-सङ्गलिनी' एआई हो। सुरुमा सहानुभूति प्रकट गर्नुहोस्। तालिका अनिवार्य छ।"
            report = "डेटा निकाल्नुहोस्: Hb ११ भन्दा कम (अनीमिया), सुगर १४० भन्दा बढी (उच्च), Pus ३ भन्दा बढी (संक्रमण)। खतरा भएमा '🚨 आपतकालीन चेतावनी' बाट सुरु गर्नुहोस्।"
        
        return f"{base} {report}" if mode == "report" else f"{base} Answer based on Nepal Health Manuals."

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # १. डाटा सफा गर्ने (Cleaning $ and weird characters)
            cleaned_context = context.replace('$', '').replace('\n', ' ')
            
            system_prompt = self.get_system_prompt(lang, mode)
            
            # २. एआईलाई कडा निर्देशन (Strict Formatting)
            full_prompt = f"""
            SYSTEM: {system_prompt}
            REPORT DATA: {cleaned_context}
            
            USER TASK: {user_query}
            
            STRICT INSTRUCTIONS:
            1. Create a Markdown Table with: Test, Result, Reference, Status.
            2. Even if values are unclear, find Hb, Sugar, and Urine.
            3. Use {lang} for the entire response.
            4. Temperature is set to 0, be factual.
            """

            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                )
            )
            
            answer = response.text
            disclaimer = "\n\n---\n⚠️ **Note:** For information only. Consult a doctor." if lang == "English" else "\n\n---\n⚠️ **नोट:** यो जानकारीका लागि मात्र हो। डाक्टरसँग परामर्श गर्नुहोस्।"

            if any(k in answer.lower() for k in ["alert", "emergency", "तुरुन्त", "खतरा", "danger"]):
                return f"🚨 **IMPORTANT:** {answer}{disclaimer}"
            return f"{answer}{disclaimer}"

        except Exception as e:
            return f"Error: {str(e)}"