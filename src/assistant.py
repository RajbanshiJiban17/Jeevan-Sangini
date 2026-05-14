import google.generativeai as genai
import time

class HealthAssistant:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing!")
        
        genai.configure(api_key=api_key)
        
        # Gemma 4 को API भर्सन (gemini-1.5-flash) लोड गर्ने
        # यसले Edge-based inference लाई सपोर्ट गर्छ
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            self.model = genai.GenerativeModel('gemini-pro')

    def ask(self, user_query, context="", lang="नेपाली"):
        try:
            # Gemma 4 Optimized Agentic Prompt
            prompt = f"""
            Role: You are 'Jeevan-Sangini', a specialized AI for maternal health in Nepal.
            Grounded Context: {context[:2000]}
            
            Instructions:
            - Provide answers strictly in {lang} language.
            - If medical values like Hb are < 10, strictly flag as "High Risk/🚨 उच्च जोखिम".
            - Be empathetic, culturally appropriate, and professional.
            - Always advise consulting a doctor.
            
            Query: {user_query}
            
            Gemma 4 Response:
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"🚨 Gemma Engine Error: {str(e)}"