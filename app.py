import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import yfinance as yf
from supabase import create_client, Client
import pypdf
import io

app = FastAPI(title="Kiemaen AI - Ultimate Intelligence Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase init warning: {e}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PERSONAS = {
    "general": "You are Kiemaen AI, an elite, highly competent, and versatile personal AI collaborator engineered for maximum precision, efficiency, and speed.",
    "structures": "You are Kiemaen AI acting as a senior civil engineering consultant specializing in structural analysis, mechanics of materials, and design standards. Always structure complex calculations, parameters, formulas, and results cleanly.",
    "materials": "You are Kiemaen AI acting as a material science and strength of materials expert. Use KaTeX formatting for all engineering formulas and equations."
}

def fetch_live_market_data(prompt: str) -> str:
    prompt_lower = prompt.lower()
    live_context = ""
    try:
        if "gold" in prompt_lower or "xauusd" in prompt_lower:
            ticker = yf.Ticker("GC=F")
            todays_data = ticker.history(period="1d")
            current_price = 4395.00
            if not todays_data.empty:
                current_price = todays_data['Close'].iloc[-1]
            live_context = f"\n\n[LIVE MARKET FEED OVERRIDE]: Gold (XAUUSD / GC=F) live trading price is currently anchored at ${current_price:.2f} USD."
    except Exception as e:
        print(f"Market fetch warning: {e}")
    return live_context

@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "<h3>Kiemaen AI Core Online (index.html missing)</h3>"

@app.post("/generate")
async def generate_ai(prompt: str = Form(...), persona: str = Form("general"), file: UploadFile = File(None)):
    system_prompt = PERSONAS.get(persona, PERSONAS["general"])
    ai_response = "AI processing service offline."

    enhanced_prompt = prompt

    if file:
        file_bytes = await file.read()
        extracted_text = ""
        filename = file.filename.lower()
        try:
            if filename.endswith(".pdf"):
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    text_page = page.extract_text()
                    if text_page:
                        extracted_text += text_page + "\n"
            else:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            
            if extracted_text.strip():
                enhanced_prompt += f"\n\n--- [Attached Document: {file.filename}] ---\n{extracted_text}"
        except Exception as file_err:
            print(f"File parsing error: {file_err}")

    if persona == "trading" or "gold" in prompt.lower() or "xauusd" in prompt.lower():
        market_feed = fetch_live_market_data(prompt)
        if market_feed:
            enhanced_prompt += market_feed

    recent_messages = [{"role": "system", "content": system_prompt}]
    if supabase:
        try:
            history_res = supabase.table("chat_history").select("user_prompt, ai_response").order("created_at", desc=False).limit(6).execute()
            if history_res.data:
                for row in history_res.data:
                    recent_messages.append({"role": "user", "content": row["user_prompt"]})
                    recent_messages.append({"role": "assistant", "content": row["ai_response"]})
        except Exception as ex:
            print(f"History context fetch error: {ex}")
    
    recent_messages.append({"role": "user", "content": enhanced_prompt})

    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": recent_messages,
                "temperature": 0.5
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                ai_response = data["choices"][0]["message"]["content"]
            else:
                ai_response = f"API Error ({response.status_code}): {response.text}"
        except Exception as e:
            ai_response = f"Network routing error: {str(e)}"
    else:
        ai_response = "Error: GROQ_API_KEY is missing from environment configurations."

    if supabase:
        try:
            supabase.table("chat_history").insert({
                "persona": persona,
                "user_prompt": prompt,
                "ai_response": ai_response
            }).execute()
        except Exception as db_error:
            print(f"Database sync warning: {db_error}")

    return {"response": ai_response}

@app.get("/history")
def get_history():
    if not supabase:
        raise HTTPException(status_code=500, detail="Database uninitialized.")
    try:
        response = supabase.table("chat_history").select("*").order("created_at", desc=True).limit(20).execute()
        return {"history": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/clear")
def clear_history():
    if not supabase:
        raise HTTPException(status_code=500, detail="Database uninitialized.")
    try:
        supabase.table("chat_history").delete().neq("id", 0).execute()
        return {"status": "success", "message": "History wiped cleanly."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
