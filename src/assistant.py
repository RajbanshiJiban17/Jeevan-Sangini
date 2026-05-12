import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key):
        # १. Google AI Studio Configuration
        genai.configure(api_key=api_key)
        
        # २. Dynamic Model Routing: उपलब्ध मोडलहरू चेक गर्ने
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            # प्राथमिकता: १. Gemma 2 (ह्याकाथनको लागि), २. Gemini 1.5 Flash (ब्याकअप), ३. अन्य
            if "models/gemma-2-9b-it" in available_models:
                selected_model = "models/gemma-2-9b-it"
            elif "models/gemini-1.5-flash" in available_models:
                selected_model = "models/gemini-1.5-flash"
            else:
                # यदि माथिका कुनै भेटिएनन् भने उपलब्ध मध्ये पहिलो रोज्ने
                selected_model = available_models[0] if available_models else "gemini-1.5-flash"
                
            self.model = genai.GenerativeModel(model_name=selected_model)
            self.active_model = selected_model
        except Exception as e:
            # सुरक्षित रहनको लागि डिफल्ट मोडल राख्ने
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            self.active_model = "gemini-1.5-flash (fallback)"

    def get_system_prompt(self, lang, mode):
        """Gemma/Gemini को लागि विशेष Agentic System Prompt"""
        # यहाँ हामी Gemma 2 नै लेख्छौँ किनकि प्रम्प्ट इन्जिन Gemma कै आधारमा छ
        model_identity = "Gemma 2"
        
        if mode == "report":
            if lang == "English":
                return (
                    f"You are a Senior Obstetrician powered by {model_identity}. Analyze medical reports with high precision. "
                    "If you detect danger signs (e.g., low Hb, high BP), explicitly mention 'EMERGENCY ALERT'. "
                    "Be factual, grounded, and always provide a medical disclaimer."
                )
            else:
                return (
                    f"तपाईँ {model_identity} द्वारा सञ्चालित वरिष्ठ प्रसूति विशेषज्ञ हो। मेडिकल रिपोर्टको सूक्ष्म विश्लेषण गर्नुहोस्। "
                    "यदि रिपोर्टमा खतराको संकेत (जस्तै: कम रगत, उच्च रक्तचाप) देखिएमा 'आपतकालीन चेतावनी' प्रस्ट दिनुहोस्। "
                    "तथ्यमा आधारित रहेर सरल नेपालीमा बुझाउनुहोस् र डिस्क्लेमर अनिवार्य राख्नुहोस्।"
                )
        else:
            if lang == "English":
                return (
                    f"You are 'Jeevan-Sangini' AI, a maternal health assistant built on {model_identity}. "
                    "Your responses must be grounded strictly in the provided context from Nepal Health Manuals. "
                    "For emergencies, guide them to SOS. If info is missing in context, politely defer to a doctor."
                )
            else:
                return (
                    f"तपाईँ {model_identity} मा आधारित 'जीवन-सङ्गलिनी' एआई स्वास्थ्य सहायक हुनुहुन्छ। "
                    "तपाईँको जवाफ नेपाल सरकारको आधिकारिक स्वास्थ्य निर्देशिका (Context) मा मात्र आधारित हुनुपर्छ। "
                    "यदि जानकारी उपलब्ध छैन भने नम्रताका साथ डाक्टरसँग परामर्श गर्न भन्नुहोस्। मनगढन्ते कुरा नगर्नुहोस्।"
                )

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        """Grounded Reasoning र Identity Check सहितको मुख्य फङ्सन"""
        try:
            # १. Identity Check
            id_keywords = ["who are you", "timi ko hau", "तपाईं को हो", "परिचय", "your name"]
            if any(k in user_query.lower() for k in id_keywords):
                if lang == "English":
                    return f"I am 'Jeevan-Sangini' AI, a maternal health companion powered by Google's Gemma technology."
                else:
                    return f"म 'जीवन-सङ्गलिनी' एआई हुँ। म गुगलको Gemma प्रविधिमा आधारित डिजिटल स्वास्थ्य सहायक हुँ।"

            # २. सिस्टम प्रम्प्ट र पेलोड तयार गर्ने
            system_prompt = self.get_system_prompt(lang, mode)
            
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context: {context}\n\n"
                f"User Question: {user_query}\n\n"
                f"Instruction: Provide a grounded and empathetic response in {lang}."
            )

            # ३. API कल
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )
            
            answer = response.text
            
            # ४. डिस्क्लेमर र सुरक्षा जाँच
            disclaimer = (
                "\n\n---\n⚠️ **Note:** This AI is for information only. Please consult a doctor for medical advice." 
                if lang == "English" else 
                "\n\n---\n⚠️ **नोट:** यो एआई केवल जानकारीका लागि हो। स्वास्थ्य सम्बन्धी सल्लाहका लागि अनिवार्य रूपमा डाक्टरसँग परामर्श गर्नुहोस्।"
            )

            if mode == "report" and ("danger" in answer.lower() or "तुरुन्त" in answer or "alert" in answer.lower()):
                prefix = "⚠️ **URGENT:** " if lang == "English" else "⚠️ **महत्त्वपूर्ण चेतावनी:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer
            
        except Exception as e:
            return f"एआई इन्जिनमा प्राविधिक समस्या आयो: ({str(e)})"