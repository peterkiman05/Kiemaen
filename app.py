import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Ultimate AI Core", version="1.0")

class PromptRequest(BaseModel):
    prompt: str
    model: str = "openai/gpt-oss-20b:free"  # Default open-tier model via OpenRouter

@app.get("/")
def home():
    return {
        "status": "Online", 
        "architecture": "Cloud-Native GitHub + Render Pipeline",
        "message": "Your ultimate AI core is successfully deployed and ready."
    }

@app.post("/generate")
def generate_text(request: PromptRequest):
    # Securely fetch your free API key from Render's environment variables
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="Configuration Error: AI_API_KEY is missing in environment variables."
        )
    
    # OpenRouter endpoint (compatible with OpenAI standard request formats)
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ultimate-ai-core", # Required by OpenRouter for ranking
        "X-Title": "Ultimate AI System"
    }
    
    payload = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": "You are the core intelligence of an advanced, multi-system AI."},
            {"role": "user", "content": request.prompt}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response_data = response.json()
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response_data)
            
        ai_reply = response_data["choices"][0]["message"]["content"]
        return {"model_used": request.model, "response": ai_reply}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
