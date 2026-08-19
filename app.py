import os
import io
import traceback
import contextlib
import sympy as sp
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
import httpx
import yfinance as yf
from supabase import create_client, Client
from pypdf import PdfReader
from duckduckgo_search import DDGS

app = FastAPI(title="Kiemaen AI - Plan-and-Execute Multi-Agent Core")

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
    "engineering": "You are Kiemaen AI acting as a senior civil engineering consultant specializing in structural analysis, mechanics of materials, and design codes. Always structure complex calculations, parameters, formulas, and results using clean Markdown tables.",
    "trading": "You are Kiemaen AI acting as a master institutional trading mentor specializing in proprietary risk parameters, automated execution plans, and technical analysis. CRITICAL RULE: When live market feed context is injected, construct all support, resistance, entry, stop-loss, and take-profit targets strictly around those exact real-time live prices. Format all risk metrics inside pristine markdown data tables.",
    "coding": "You are Kiemaen AI acting as a Principal Software Engineer proficient in Python, TypeScript, and cloud containerization architecture. Always provide modular, bug-free, production-ready code blocks accompanied by precise execution details."
}

def research_agent_node(query: str) -> str:
    """Specialized Agent: Autonomous web research extractor."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if results:
                snippets = "\n".join([f"- Title: {r.get('title')}\n  Snippet: {r.get('body')}" for r in results])
                return f"\n\n[Plan-and-Execute Phase 1 - Research Findings]:\n{snippets}"
    except Exception as e:
        print(f"Research agent warning: {e}")
    return ""

def computation_agent_node(code_string: str) -> str:
    """Specialized Agent: Isolated mathematical & analytical executor."""
    output = io.StringIO()
    try:
        local_vars = {"sp": sp, "np": np, "result": None}
        with contextlib.redirect_stdout(output):
            exec(code_string, {"__builtins__": {
                "print": print, "range": range, "len": len, "round": round, 
                "abs": abs, "min": min, "max": max, "sum": sum, "float": float, "int": int
            }}, local_vars)
        
        captured = output.getvalue()
        if local_vars.get("result") is not None:
            captured += f"\n[Agent Computed Result]: {local_vars['result']}"
        return captured.strip() or "Execution completed successfully with no console output."
    except Exception as e:
        return f"Computation Agent Error: {str(e)}\n{traceback.format_exc()}"

def fetch_live_market_data(prompt: str) -> str:
    prompt_lower = prompt.lower()
    live_context = ""
    try:
        if "gold" in prompt_lower or "xauusd" in prompt_lower:
            ticker = yf.Ticker("GC=F")
            todays_data = ticker.history(period="1d")
            current_price = todays_data['Close'].iloc[-1] if not todays_data.empty else 4395.00
            live_context = f"\n\n[LIVE MARKET FEED OVERRIDE]: Gold (XAUUSD / GC=F) live trading price is currently anchored at ${current_price:.2f} USD."
        elif "btc" in prompt_lower or "bitcoin" in prompt_lower:
            ticker = yf.Ticker("BTC-USD")
            todays_data = ticker.history(period="1d")
            current_price = todays_data['Close'].iloc[-1] if not todays_data.empty else 65000.00
            live_context = f"\n\n[LIVE MARKET FEED OVERRIDE]: Bitcoin (BTC-USD) live trading price is currently anchored at ${current_price:.2f} USD."
    except Exception as e:
        print(f"Market fetch warning: {e}")
    return live_context

@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "<h3>Kiemaen AI Plan-and-Execute Core Online</h3>"

@app.post("/generate")
async def generate_ai(
    prompt: str = Form(...),
    persona: str = Form("general"),
    file: UploadFile = File(None)
):
    system_prompt = PERSONAS.get(persona, PERSONAS["general"])
    ai_response = "AI processing service offline."

    enhanced_prompt = prompt
    
    # Trigger Market Data Feed if matching financial intent
    if persona == "trading" or any(k in prompt.lower() for k in ["gold", "xauusd", "btc", "bitcoin"]):
        market_feed = fetch_live_market_data(prompt)
        if market_feed:
            enhanced_prompt += market_feed

    # Trigger Research Agent if query demands live data/documentation lookup
    if any(k in prompt.lower() for k in ["search", "latest", "documentation", "standard", "eurocode", "news", "how to"]):
        research_context = research_agent_node(prompt)
        if research_context:
            enhanced_prompt += research_context

    # Handle file/PDF document attachments
    if file:
        file_bytes = await file.read()
        extracted_text = ""
        if file.filename.endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
            except Exception as pdf_err:
                extracted_text = f"[Error reading PDF: {pdf_err}]"
        else:
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                extracted_text = "[Binary file uploaded]"
        
        enhanced_prompt += f"\n\n[Attached Document Context ({file.filename})]:\n```\n{extracted_text[:10000]}\n```"

    # Assemble state context history from Supabase
    recent_messages = [{"role": "system", "content": system_prompt}]
    if supabase:
        try:
            history_res = supabase.table("chat_history").select("user_prompt, ai_response").order("created_at", desc=False).limit(6).execute()
            if history_res.data:
                for row in history_res.data:
                    recent_messages.append({"role": "user", "content": row["user_prompt"]})
                    recent_messages.append({"role": "assistant", "content": row["ai_response"]})
        except Exception as ex:
            print(f"History fetch error: {ex}")
    
    recent_messages.append({"role": "user", "content": enhanced_prompt})

    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": recent_messages,
                "temperature": 0.3
            }
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    ai_response = data["choices"][0]["message"]["content"]
                    
                    # Computational Sandbox Node execution check
                    if "```python:run" in ai_response:
                        try:
                            start_idx = ai_response.find("```python:run") + 13
                            end_idx = ai_response.find("```", start_idx)
                            if end_idx != -1:
                                code_to_run = ai_response[start_idx:end_idx].strip()
                                execution_output = computation_agent_node(code_to_run)
                                ai_response += f"\n\n**Multi-Agent Execution Pipeline Output:**\n```text\n{execution_output}\n```"
                        except Exception as exec_err:
                            print(f"Computation execution runner error: {exec_err}")
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
