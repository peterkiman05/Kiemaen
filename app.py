import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Ultimate AI Core - Multi-Persona", version="2.0")

class PromptRequest(BaseModel):
    prompt: str
    persona: str = "general"  # Options: general, engineering, trading, coding
    model: str = "openai/gpt-oss-20b:free"

# Define different expert system prompts
PERSONAS = {
    "general": "You are the core intelligence of an advanced, multi-system AI.",
    "engineering": "You are an expert Civil Engineering assistant specializing in mechanics, structures, material science, and numerical methods.",
    "trading": "You are an expert financial trading and proprietary risk-management strategist familiar with MetaTrader, market structures, and price action.",
    "coding": "You are an elite software developer and debugging expert proficient in Python, TypeScript, and cloud deployment pipelines."
}

@app.get("/")
def home():
    return {
        "status": "Online", 
        "version": "2.0 Multi-Persona Active",
        "available_personas": list(PERSONAS.keys())
    }

@app.post("/generate")
def generate_text(request: PromptRequest):
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Configuration Error: AI_API_KEY is missing.")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/peterkiman05/Kiemaen",
        "X-Title": "Ultimate AI System"
    }
    
    # Select the system prompt based on the chosen persona
    system_prompt = PERSONAS.get(request.persona, PERSONAS["general"])
    
    payload = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response_data = response.json()
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response_data)
            
        ai_reply = response_data["choices"][0]["message"]["content"]
        return {
            "persona_used": request.persona,
            "model_used": request.model,
            "response": ai_reply
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
