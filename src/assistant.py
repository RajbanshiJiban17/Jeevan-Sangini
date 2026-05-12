import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if "models/gemma-2-9b-it" in available_models:
                selected_model = "models/gemma-2-9b-it"
            elif "models/gemini-1.5-flash" in available_models:
                selected_model = "models/gemini-1.5-flash"
            else:
                selected_model = available_models[0] if available_models else "gemini-1.5-flash"
            self.model = genai.GenerativeModel(model_name=selected_model)
            self.active_model = selected_model
        except Exception as e:
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            self.active_model = "gemini-1.5-flash (fallback)"

    def get_system_prompt(self, lang, mode):
        """Gemma लाई समानुभूतिपूर्ण र व्यवस्थित बनाउन सुधारिएको प्रम्प्ट"""
        model_identity = "Gemma 2"
        
        # साझा निर्देशन (Shared Instructions)
        if lang == "English":
            base_instruction = (
                f"You are 'Jeevan-Sangini' AI powered by {model_identity}. "
                "Always start your response with a kind, empathetic sentence that validates the user's concern. "
                "Use bullet points for clarity. Use ⚠️ for danger signs and **bold text** for important advice."
            )
        else:
            base_instruction = (
                f"तपाईँ {model_identity} मा आधारित 'जीवन-सङ्गलिनी' एआई हो। "
                "जहिले पनि जवाफको सुरुमा प्रयोगकर्ताको समस्या वा भावनाप्रति सहानुभूति प्रकट गर्ने एउटा मिठो वाक्य लेख्नुहोस्। "
                "जानकारीहरूलाई बुँदागत रूपमा राख्नुहोस्। खतराको संकेत भएमा ⚠️ इमोजी र **बोल्ड अक्षर** प्रयोग गर्नुहोस्।"
            )

        if mode == "report":
            if lang == "English":
                return f"{base_instruction} Analyze this medical report precisely like a senior obstetrician."
            else:
                return f"{base_instruction} वरिष्ठ प्रसूति विशेषज्ञको रूपमा यो मेडिकल रिपोर्टको सूक्ष्म विश्लेषण गर्नुहोस्।"
        else:
            if lang == "English":
                return f"{base_instruction} Answer strictly based on the provided Nepal Health Manuals context."
            else:
                return f"{base_instruction} उपलब्ध नेपाल स्वास्थ्य निर्देशिका (Context) को आधारमा मात्र जवाफ दिनुहोस्।"

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # १. Identity Check
            id_keywords = ["who are you", "timi ko hau", "तपाईं को हो", "परिचय", "your name"]
            if any(k in user_query.lower() for k in id_keywords):
                if lang == "English":
                    return "I am 'Jeevan-Sangini' AI, a maternal health companion powered by Google's Gemma technology."
                else:
                    return "म 'जीवन-सङ्गिनी' एआई हुँ। म गुगलको Gemma प्रविधिमा आधारित डिजिटल स्वास्थ्य सहायक हुँ।"

            # २. सिस्टम प्रम्प्ट र पेलोड
            system_prompt = self.get_system_prompt(lang, mode)
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context: {context}\n\n"
                f"User Question: {user_query}\n\n"
                f"Instruction: Be empathetic, structured, and use {lang}."
            )

            # ३. API कल
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, # अलि बढी 'Human-like' बनाउन थोरै बढाएको
                    max_output_tokens=1024,
                )
            )
            answer = response.text
            
            # ४. डिस्क्लेमर
            disclaimer = (
                "\n\n---\n⚠️ **Note:** For information only. Consult a doctor for medical advice." 
                if lang == "English" else 
                "\n\n---\n⚠️ **नोट:** यो जानकारीका लागि मात्र हो। स्वास्थ्य सल्लाहको लागि डाक्टरसँग परामर्श गर्नुहोस्।"
            )

            if mode == "report" and ("danger" in answer.lower() or "तुरुन्त" in answer or "alert" in answer.lower()):
                prefix = "⚠️ **URGENT:** " if lang == "English" else "⚠️ **महत्त्वपूर्ण चेतावनी:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer
            
        except Exception as e:
            return f"प्राविधिक समस्या आयो: ({str(e)})"