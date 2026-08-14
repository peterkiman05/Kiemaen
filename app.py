import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from supabase import create_client, Client

app = FastAPI(title="Ultimate AI Core - Memory & Personas", version="2.1")

# Initialize Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class PromptRequest(BaseModel):
    prompt: str
    persona: str = "general"
    model: str = "openai/gpt-oss-20b:free"

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
        "version": "2.1 Memory Enabled",
        "database_connected": bool(supabase)
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
        
        # Save interaction to Supabase if database is configured
        if supabase:
            try:
                supabase.table("chat_history").insert({
                    "persona": request.persona,
                    "user_prompt": request.prompt,
                    "ai_response": ai_reply
                }).execute()
            except Exception as db_err:
                print(f"Database save warning: {db_err}")

        return {
            "persona_used": request.persona,
            "model_used": request.model,
            "response": ai_reply,
            "saved_to_memory": bool(supabase)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        response = supabase.table("chat_history").select("*").order("created_at", desc=True).limit(10).execute()
        return {"recent_history": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
