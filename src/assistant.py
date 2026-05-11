import os
import google.generativeai as genai

class HealthAssistant:
    def __init__(self, api_key, model_name="gemma2-9b-it"):
        self.client = genai(api_key=api_key)
        # Kaggle Gemma Challenge को लागि Gemma मोडेल प्रयोग गरिएको
        self.model = model_name 
        
    def get_system_prompt(self, lang, mode):
        """Gemma 4 को लागि विशेष Agentic System Prompt"""
        if mode == "report":
            if lang == "English":
                return (
                    "You are a Senior Obstetrician powered by Gemma 4. Analyze medical reports with high precision. "
                    "If you detect danger signs (e.g., low Hb, high BP), explicitly mention 'EMERGENCY ALERT'. "
                    "Be factual, grounded, and always provide a medical disclaimer."
                )
            else:
                return (
                    "तपाईँ Gemma 4 द्वारा सञ्चालित वरिष्ठ प्रसूति विशेषज्ञ हो। मेडिकल रिपोर्टको सूक्ष्म विश्लेषण गर्नुहोस्। "
                    "यदि रिपोर्टमा खतराको संकेत (जस्तै: कम रगत, उच्च रक्तचाप) देखिएमा 'आपतकालीन चेतावनी' प्रस्ट दिनुहोस्। "
                    "तथ्यमा आधारित रहेर सरल नेपालीमा बुझाउनुहोस् र डिस्क्लेमर अनिवार्य राख्नुहोस्।"
                )
        else:
            if lang == "English":
                return (
                    "You are 'Jeevan-Sangini' AI, a maternal health assistant built on Gemma 4. "
                    "Your responses must be grounded strictly in the provided context from Nepal Health Manuals. "
                    "For emergencies, guide them to SOS. If info is missing in context, politely defer to a doctor."
                )
            else:
                return (
                    "तपाईँ Gemma 4 मा आधारित 'जीवन-सङ्गिनी' एआई स्वास्थ्य सहायक हुनुहुन्छ। "
                    "तपाईँको जवाफ नेपाल सरकारको आधिकारिक स्वास्थ्य निर्देशिका (Context) मा मात्र आधारित हुनुपर्छ। "
                    "यदि जानकारी उपलब्ध छैन भने नम्रताका साथ डाक्टरसँग परामर्श गर्न भन्नुहोस्। मनगढन्ते कुरा नगर्नुहोस्।"
                )

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        """Grounded Reasoning र Identity Check सहितको मुख्य फङ्सन"""
        try:
            # १. Identity Check (Gemma 4 Powered branding)
            id_keywords = ["who are you", "timi ko hau", "तपाईं को हो", "परिचय", "your name"]
            if any(k in user_query.lower() for k in id_keywords):
                if lang == "English":
                    return "I am 'Jeevan-Sangini' AI, a maternal health companion powered by Google's Gemma 4 technology, designed to provide grounded medical insights."
                else:
                    return "म 'जीवन-सङ्गिनी' एआई हुँ। म गुगलको Gemma 4 प्रविधिमा आधारित डिजिटल स्वास्थ्य सहायक हुँ, जसले तपाईंलाई सुरक्षित गर्भावस्थाका लागि मद्दत गर्छ।"

            # २. सिस्टम प्रम्प्ट तयार गर्ने
            system_prompt = self.get_system_prompt(lang, mode)
            
            # ३. Agentic Retrieval Logic: रिपोर्ट मोडमा अझ बढी सतर्कता
            # Gemma 4 ले इन्स्ट्रक्सनहरू राम्रोसँग पछ्याउँछ (Instruction Following)
            prompt_payload = f"Context from Medical Manuals: {context}\n\nUser Query: {user_query}\n\nInstruction: Provide a grounded, safe, and empathetic response in {lang}."

            # ४. Groq API कल
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_payload}
                ],
                temperature=0.1, # मेडिकल जवाफमा स्थिरताका लागि कम टेम्परेचर
                max_tokens=1000,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content

            # ५. Safety Audit (Double Check for critical issues)
            # यदि रिपोर्टमा खतरा छ तर एआईले भन्न बिर्सियो भने यो लोजिकले मद्दत गर्छ
            if mode == "report" and ("danger" in answer.lower() or "तुरुन्त" in answer):
                prefix = "⚠️ **URGENT:** " if lang == "English" else "⚠️ **महत्त्वपूर्ण चेतावनी:** "
                return prefix + answer
            
            return answer
            
        except Exception as e:
            error_msg = f"Technical error with Gemma Engine." if lang == "English" else f"Gemma इन्जिनमा प्राविधिक समस्या आयो।"
            return f"{error_msg} ({str(e)})"