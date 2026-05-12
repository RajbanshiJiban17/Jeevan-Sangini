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
        """Gemma लाई Detailed Report Analysis र Empathy सिकाउने अन्तिम प्रम्प्ट"""
        model_identity = "Gemma 2"
        
        if lang == "English":
            base_instruction = (
                f"You are 'Jeevan-Sangini' AI powered by {model_identity}. "
                "Start with a kind, empathetic sentence. Use bullet points and tables. "
                "Use ⚠️ for danger signs and **bold text** for critical values."
            )
            report_instruction = (
                "1. Extract and list key medical parameters (Hb, Sugar, BP, Protein, etc.) in a Markdown Table. "
                "2. Clearly state if each value is 'Normal', 'Low', or 'High' based on standard maternal health ranges. "
                "3. If any value is life-threatening, start the response with '🚨 EMERGENCY ALERT'."
            )
        else:
            base_instruction = (
                f"तपाईँ {model_identity} मा आधारित 'जीवन-सङ्गलिनी' एआई हो। "
                "जहिले पनि जवाफको सुरुमा सहानुभूति प्रकट गर्ने मिठो वाक्य लेख्नुहोस्। "
                "जानकारीलाई बुँदा र तालिका (Table) मा देखाउनुहोस्। खतराको संकेत भएमा ⚠️ र **बोल्ड अक्षर** प्रयोग गर्नुहोस्।"
            )
            report_instruction = (
                "१. रिपोर्टबाट मुख्य कुराहरू (Hb, सुगर, रक्तचाप, आदि) झिकेर एउटा तालिका (Table) मा देखाउनुहोस्। "
                "२. प्रत्येक नतिजा सामान्य छ कि छैन, प्रस्ट लेख्नुहोस्। "
                "३. यदि रिपोर्टमा गम्भीर समस्या देखिएमा '🚨 आपतकालीन चेतावनी' बाट जवाफ सुरु गर्नुहोस्।"
            )

        if mode == "report":
            return f"{base_instruction} {report_instruction}"
        else:
            return f"{base_instruction} Answer based strictly on Nepal Health Manuals context."

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        try:
            # १. Identity Check
            id_keywords = ["who are you", "timi ko hau", "तपाईं को हो", "परिचय", "your name"]
            if any(k in user_query.lower() for k in id_keywords):
                return "म 'जीवन-सङ्गलिनी' एआई हुँ, गुगलको Gemma प्रविधिमा आधारित डिजिटल स्वास्थ्य सहायक।" if lang != "English" else "I am 'Jeevan-Sangini' AI, a maternal health companion powered by Gemma."

            # २. सिस्टम प्रम्प्ट र पेलोड
            system_prompt = self.get_system_prompt(lang, mode)
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context/Report Data: {context}\n\n"
                f"User Question: {user_query}\n\n"
                f"Instruction: Be highly detailed. If it's a report, compare values with standards. Use {lang}."
                f"Special Instruction: The report may contain LaTeX symbols like $ or (+). "
                f"Please clean the text in your mind and extract only the numerical values and units."
                f"Extract values even if they have $ symbols"
            )

            # ३. API कल (Temperature अलि कम राखिएको छ ताकि डाटा सही आओस्)
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, 
                    max_output_tokens=2000,
                )
            )
            answer = response.text
            
            # ४. डिस्क्लेमर
            disclaimer = (
                "\n\n---\n⚠️ **Note:** For information only. Consult a doctor immediately for medical decisions." 
                if lang == "English" else 
                "\n\n---\n⚠️ **नोट:** यो जानकारीका लागि मात्र हो। मेडिकल निर्णय लिनुअघि डाक्टरसँग परामर्श गर्नुहोस्।"
            )

            # ५. आपतकालीन अलर्ट थप्ने (यदि एजेन्टले खतरा देख्यो भने)
            if "alert" in answer.lower() or "emergency" in answer.lower() or "तुरुन्त" in answer:
                prefix = "🚨 **EMERGENCY NOTICE:** " if lang == "English" else "🚨 **आपतकालीन सूचना:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer
            
        except Exception as e:
            return f"Error: ({str(e)})"