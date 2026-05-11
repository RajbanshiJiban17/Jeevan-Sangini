import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key, model_name="models/gemma-2-9b"):
        # Google AI Studio Configuration
        genai.configure(api_key=api_key)
        # GenerativeModel इन्स्टन्स बनाउने
        self.model = genai.GenerativeModel(
            model_name=model_name
        )
        
    def get_system_prompt(self, lang, mode):
        """Gemma 2 को लागि विशेष Agentic System Prompt"""
        if mode == "report":
            if lang == "English":
                return (
                    "You are a Senior Obstetrician powered by Gemma 2. Analyze medical reports with high precision. "
                    "If you detect danger signs (e.g., low Hb, high BP), explicitly mention 'EMERGENCY ALERT'. "
                    "Be factual, grounded, and always provide a medical disclaimer."
                )
            else:
                return (
                    "तपाईँ Gemma 2 द्वारा सञ्चालित वरिष्ठ प्रसूति विशेषज्ञ हो। मेडिकल रिपोर्टको सूक्ष्म विश्लेषण गर्नुहोस्। "
                    "यदि रिपोर्टमा खतराको संकेत (जस्तै: कम रगत, उच्च रक्तचाप) देखिएमा 'आपतकालीन चेतावनी' प्रस्ट दिनुहोस्। "
                    "तथ्यमा आधारित रहेर सरल नेपालीमा बुझाउनुहोस् र डिस्क्लेमर अनिवार्य राख्नुहोस्।"
                )
        else:
            if lang == "English":
                return (
                    "You are 'Jeevan-Sangini' AI, a maternal health assistant built on Gemma 2. "
                    "Your responses must be grounded strictly in the provided context from Nepal Health Manuals. "
                    "For emergencies, guide them to SOS. If info is missing in context, politely defer to a doctor."
                )
            else:
                return (
                    "तपाईँ Gemma 2 मा आधारित 'जीवन-सङ्गलिनी' एआई स्वास्थ्य सहायक हुनुहुन्छ। "
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
                    return "I am 'Jeevan-Sangini' AI, a maternal health companion powered by Google's Gemma 2 technology."
                else:
                    return "म 'जीवन-सङ्गिनी' एआई हुँ। म गुगलको Gemma 2 प्रविधिमा आधारित डिजिटल स्वास्थ्य सहायक हुँ।"

            # २. सिस्टम प्रम्प्ट र पेलोड तयार गर्ने
            system_prompt = self.get_system_prompt(lang, mode)
            
            # Google API को लागि प्रम्प्ट स्ट्रक्चर
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context: {context}\n\n"
                f"User Question: {user_query}\n\n"
                f"Instruction: Provide a grounded and empathetic response in {lang}."
            )

            # ३. Google AI Studio (Gemini/Gemma) API कल
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # स्थिरताका लागि कम टेम्परेचर
                    max_output_tokens=1000,
                )
            )
            
            answer = response.text
            
            # ४. डिस्क्लेमर थप्ने (तपाईँले भन्नुभएको मुख्य कुरा)
            disclaimer = (
                "\n\n---\n⚠️ **Note:** This AI is for information only. Please consult a doctor for medical advice." 
                if lang == "English" else 
                "\n\n---\n⚠️ **नोट:** यो एआई केवल जानकारीका लागि हो। स्वास्थ्य सम्बन्धी सल्लाहका लागि अनिवार्य रूपमा डाक्टरसँग परामर्श गर्नुहोस्।"
            )

            # ४. Safety Audit
            if mode == "report" and ("danger" in answer.lower() or "तुरुन्त" in answer or "alert" in answer.lower()):
                prefix = "⚠️ **URGENT:** " if lang == "English" else "⚠️ **महत्त्वपूर्ण चेतावनी:** "
                return prefix + answer + disclaimer
            
            return answer + disclaimer
            
        except Exception as e:
            error_msg = "Technical error with Gemma Engine." if lang == "English" else "Gemma इन्जिनमा प्राविधिक समस्या आयो।"
            return f"{error_msg} ({str(e)})"