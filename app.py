import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import requests
from supabase import create_client, Client

app = FastAPI(title="Ultimate AI Core - Extended Scope")

# Initialize Supabase connection safely
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase init warning: {e}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ChatRequest(BaseModel):
    prompt: str
    persona: str = "general"

PERSONAS = {
    "general": "You are a helpful, versatile personal AI collaborator.",
    "engineering": "You are an expert civil engineering assistant specializing in structural analysis, mechanics of materials, and design standards. When performing calculations or design checks, always structure your output clearly using markdown tables for parameters, formulas, and results.",
    "trading": "You are a professional financial trading mentor specializing in proprietary trading challenges, risk management, and technical analysis. When evaluating trades or accounts, structure risk metrics (Risk-to-Reward, Lot Size, Max Drawdown) in clean data tables.",
    "coding": "You are an expert software developer proficient in Python, TypeScript, and modern web application deployment. Always provide clean, production-ready code blocks with brief execution notes."
}

@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "<h3>Ultimate AI Core is Online, but index.html was not found.</h3>"

@app.post("/generate")
def generate_ai(request: ChatRequest):
    system_prompt = PERSONAS.get(request.persona, PERSONAS["general"])
    ai_response = "AI service unavailable."

    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.prompt}
                ],
                "temperature": 0.7
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                ai_response = data["choices"][0]["message"]["content"]
            else:
                ai_response = f"API Error ({response.status_code}): {response.text}"
        except Exception as e:
            ai_response = f"Connection error: {str(e)}"
    else:
        ai_response = "GROQ_API_KEY is not configured on the server."

    # Save to Supabase if connected
    if supabase:
        try:
            supabase.table("chat_history").insert({
                "persona": request.persona,
                "user_prompt": request.prompt,
                "ai_response": ai_response
            }).execute()
        except Exception as db_error:
            print(f"Database logging error: {db_error}")

    return {"response": ai_response}

@app.get("/history")
def get_history():
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured.")
    try:
        response = supabase.table("chat_history").select("*").order("created_at", desc=True).limit(20).execute()
        return {"history": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/clear")
def clear_history():
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured.")
    try:
        response = supabase.table("chat_history").delete().neq("id", 0).execute()
        return {"status": "success", "message": "Chat history cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
