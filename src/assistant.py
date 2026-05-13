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
                return "म 'जीवन-सङ्गलिनी' एआई हुँ।" if lang != "English" else "I am 'Jeevan-Sangini' AI."

            # २. सिस्टम प्रम्प्ट र सुधारिएको पेलोड
            system_prompt = self.get_system_prompt(lang, mode)
            
            # --- यहाँ छ मुख्य फिक्स (The Final Fix) ---
            full_prompt = f"""
            {system_prompt}
            
            REPORT DATA:
            {context}
            
            INSTRUCTION FOR DEEP ANALYSIS:
            - The data above has medical values. Ignore '$' signs.
            - Extract these EXACT values if present: Hb (9.5), Sugar (155), Pus Cells (5-7), Albumin (Trace +).
            - You MUST create a Markdown Table with columns: Test, Result, Normal Range, and Status.
            - Compare results with Maternal Health standards (Hb < 11 is Anemia, Sugar > 140 is High).
            - End with a summary of risks (Anemia, UTI, Diabetes risk).
            
            USER QUESTION: {user_query}
            LANGUAGE: {lang}
            """

            # ३. API कल
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, # Fact-based नतिजाको लागि ०.०
                    max_output_tokens=2000,
                )
            )
            answer = response.text
            
            # ४. डिस्क्लेमर र अलर्ट लोजिक
            disclaimer = (
                "\n\n---\n⚠️ **Note:** For information only. Consult a doctor." 
                if lang == "English" else 
                "\n\n---\n⚠️ **नोट:** यो जानकारीका लागि मात्र हो। डाक्टरसँग परामर्श गर्नुहोस्।"
            )

            if any(word in answer.lower() for word in ["alert", "emergency", "तुरुन्त", "खतरा", "risk"]):
                prefix = "🚨 **IMPORTANT ANALYSIS:** " if lang == "English" else "🚨 **महत्त्वपूर्ण विश्लेषण:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer
            
        
        except Exception as e:
            return f"Error: ({str(e)})"