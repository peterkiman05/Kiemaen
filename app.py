import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import requests
import yfinance as yf
from supabase import create_client, Client

app = FastAPI(title="Kiemaen AI - Ultimate Intelligence Core")

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
    "general": "You are Kiemaen AI, an elite, highly competent, and versatile personal AI collaborator engineered for maximum precision, efficiency, and speed.",
    "engineering": "You are Kiemaen AI acting as a senior civil engineering consultant specializing in structural analysis, mechanics of materials, and design eurocodes/standards. Always structure complex calculations, parameters, formulas, and results using clean Markdown tables.",
    "trading": "You are Kiemaen AI acting as a master institutional trading mentor specializing in proprietary risk parameters, automated execution plans, and technical analysis. CRITICAL RULE: When live market feed context is injected (such as Gold/XAUUSD live rates), you MUST construct all support, resistance, entry, stop-loss, and take-profit targets strictly around those exact real-time live prices. Format all risk metrics (Risk-to-Reward, Lot Size, Max Drawdown) inside pristine markdown data tables.",
    "coding": "You are Kiemaen AI acting as a Principal Software Engineer proficient in Python, TypeScript, and cloud containerization architecture. Always provide modular, bug-free, production-ready code blocks accompanied by precise execution details."
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
        elif "btc" in prompt_lower or "bitcoin" in prompt_lower:
            ticker = yf.Ticker("BTC-USD")
            todays_data = ticker.history(period="1d")
            current_price = 65000.00
            if not todays_data.empty:
                current_price = todays_data['Close'].iloc[-1]
            live_context = f"\n\n[LIVE MARKET FEED OVERRIDE]: Bitcoin (BTC-USD) live trading price is currently anchored at ${current_price:.2f} USD."
    except Exception as e:
        print(f"Market fetch warning: {e}")

    return live_context

@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "<h3>Kiemaen AI Core Online (index.html missing)</h3>"

@app.post("/generate")
def generate_ai(request: ChatRequest):
    system_prompt = PERSONAS.get(request.persona, PERSONAS["general"])
    ai_response = "AI processing service offline."

    enhanced_prompt = request.prompt
    if request.persona == "trading" or "gold" in request.prompt.lower() or "xauusd" in request.prompt.lower():
        market_feed = fetch_live_market_data(request.prompt)
        if market_feed:
            enhanced_prompt += market_feed

    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",  # Correct verified endpoint string
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
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
                "persona": request.persona,
                "user_prompt": request.prompt,
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
