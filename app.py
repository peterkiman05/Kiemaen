from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
import os

app = FastAPI()

# Initialize the modern Google GenAI client
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None
except Exception:
    client = None

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the index.html user interface directly at the root URL."""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Kiemaen AI Core Online</h1><p>Error: index.html not found in project root directory.</p>"

@app.post("/generate")
async def generate_response(
    prompt: str = Form(...),
    persona: str = Form("general"),
    file: UploadFile = File(None)
):
    try:
        if not client:
            raise HTTPException(
                status_code=500, 
                detail="Gemini client not initialized. Ensure GEMINI_API_KEY is set in your Render environment variables."
            )
            
        contents = [prompt]
        
        # Handle optional file uploads
        if file:
            file_bytes = await file.read()
            contents.append(
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=file.content_type
                )
            )

        # Persona instructions mapping
        system_instructions = {
            "engineering": "You are Kiemaen AI configured as a professional Civil Engineering expert. Use precise calculations and technical standards.",
            "trading": "You are Kiemaen AI configured as an Institutional Trading core. Analyze market liquidity, technical patterns, and risk.",
            "coding": "You are Kiemaen AI configured as an advanced Software Engineering core. Provide clean, efficient code blocks.",
            "general": "You are Kiemaen AI, an autonomous intelligence core."
        }

        # Enable Google Search grounding tool using proper types config
        config = types.GenerateContentConfig(
            system_instruction=system_instructions.get(persona, system_instructions["general"]),
            tools=[{"google_search": {}}],
            temperature=0.3
        )

        # Generate response using the model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )

        return {"response": response.text}

    except Exception as e:
        return {"response": f"System Fault: {str(e)}"}
