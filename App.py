import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Ultimate AI Core", version="1.0")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "Online", "message": "Ultimate AI backend is running smoothly via GitHub and Render."}

@app.post("/generate")
def generate_text(request: PromptRequest):
    # We use an external high-speed API wrapper approach so your server 
    # doesn't crash from heavy local model memory loads.
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured in environment variables.")
    
    # Example structure for calling an external intelligent model endpoint
    try:
        # Integration logic placeholder for your chosen free AI provider (e.g., Google AI Studio / OpenRouter)
        return {"response": f"Processed prompt successfully: {request.prompt}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
