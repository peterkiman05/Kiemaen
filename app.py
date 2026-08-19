from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import google.generativeai as genai
import os

app = FastAPI()

# Configure the standard Google Generative AI library directly with your API key
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

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
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="GEMINI_API_KEY is missing from environment variables."
            )

        # Persona instructions mapping
        system_instructions = {
            "engineering": "You are Kiemaen AI configured as a professional Civil Engineering expert. Use precise calculations and technical standards.",
            "trading": "You are Kiemaen AI configured as an Institutional Trading core. Analyze market liquidity, technical patterns, and risk.",
            "coding": "You are Kiemaen AI configured as an advanced Software Engineering core. Provide clean, efficient code blocks.",
            "general": "You are Kiemaen AI, an autonomous intelligence core."
        }

        # Setup model with system instructions and Google Search grounding enabled
        model_name = "gemini-2.5-flash"
        system_instruction = system_instructions.get(persona, system_instructions["general"])
        
        generation_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            tools='default_search'  # Enables built-in Google Search grounding
        )

        # Handle contents (file bytes + prompt text)
        contents = [prompt]
        if file:
            file_bytes = await file.read()
            contents.insert(0, {
                'mime_type': file.content_type,
                'data': file_bytes
            })

        # Generate the response
        response = generation_model.generate_content(contents)

        return {"response": response.text}

    except Exception as e:
        return {"response": f"System Fault: {str(e)}"}
