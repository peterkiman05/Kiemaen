from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
import os

app = FastAPI()

# Safely initialize the GenAI client (picks up GEMINI_API_KEY environment variable)
try:
    client = genai.Client()
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
            raise HTTPException(status_code=500, detail="Gemini client not initialized. Check API key configuration.")
            
        contents = [prompt]
        
        if file:
            file_bytes = await file.read()
            contents.append(
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=file.content_type
                )
            )

        system_instructions = {
            "engineering": "You are Kiemaen AI configured as a professional Civil Engineering expert.",
            "trading": "You are Kiemaen AI configured as an Institutional Trading core.",
            "coding": "You are Kiemaen AI configured as an advanced Software Engineering core.",
            "general": "You are Kiemaen AI, an autonomous intelligence core."
        }

        # Enable live internet-connected web search grounding
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        config = types.GenerateContentConfig(
            system_instruction=system_instructions.get(persona, system_instructions["general"]),
            tools=[grounding_tool],
            temperature=0.3
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )

        return {"response": response.text}

    except Exception as e:
        return {"response": f"System Fault: {str(e)}"}
