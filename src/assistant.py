import os
from groq import Groq

class HealthAssistant:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        # मोडेल: llama-3.1-8b-instant (Fast & Accurate)
        self.model = "llama-3.1-8b-instant"
        
    def get_system_prompt(self, lang, mode):
        """भाषा र मोड अनुसारको एआईको भूमिका (Persona) तय गर्छ"""
        if mode == "report":
            if lang == "English":
                return "You are a professional Obstetrician. Analyze medical reports strictly on facts. Explain findings simply. Always add a disclaimer."
            else:
                return "तपाईँ एक विशेषज्ञ प्रसूति रोग विशेषज्ञ हो। मेडिकल रिपोर्टको सही विश्लेषण गरी सरल नेपालीमा बुझाउनुहोस् र अन्त्यमा डिस्क्लेमर अनिवार्य राख्नुहोस्।"
        else:
            if lang == "English":
                return "You are 'Jeevan-Sangini' AI, a maternal health assistant. Provide kind advice based ONLY on the context. If the answer isn't in the context, say you don't know politely."
            else:
                return "तपाईँ 'जीवन-सङ्गिनी' एआई हुनुहुन्छ। उपलब्ध गराइएको जानकारी (Context) को आधारमा मात्र आदरार्थी नेपालीमा जवाफ दिनुहोस्। मनगढन्ते उत्तर नदिनुहोस्।"

    def ask(self, user_query, context, lang="नेपाली", mode="chat"):
        """प्रश्नको उत्तर दिने मुख्य फङ्सन - Identity Check सहित"""
        try:
            # १. Identity Fix: परिचय सोध्दा सिधै उत्तर दिने
            id_keywords = ["who are you", "timi ko hau", "तपाईं को हो", "परिचय", "your name"]
            if any(k in user_query.lower() for k in id_keywords):
                if lang == "English":
                    return "I am 'Jeevan-Sangini' AI, your digital pregnancy companion. I help you understand health advice and lab reports."
                else:
                    return "म 'जीवन-सङ्गिनी' एआई हुँ। म तपाईंलाई गर्भावस्था सम्बन्धी सल्लाह दिन र मेडिकल रिपोर्टहरू बुझ्न मद्दत गर्छु।"

            # २. सिस्टम प्रम्प्ट तयार गर्ने
            system_prompt = self.get_system_prompt(lang, mode)
            
            # ३. Groq API कल
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}\nAnswer in: {lang}"}
                ],
                temperature=0.2, 
                max_tokens=800
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Sorry, a technical error occurred." if lang == "English" else f"माफ गर्नुहोस्, प्राविधिक समस्या आयो।"
            return f"{error_msg} ({str(e)})"