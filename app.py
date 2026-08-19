from fastapi import FastAPI, File, Form, UploadFile
from google import genai
from google.genai import types
import os

app = FastAPI()
client = genai.Client()  # Automatically picks up GEMINI_API_KEY from environment variables

@app.post("/generate")
async def generate_response(
    prompt: str = Form(...),
    persona: str = Form("general"),
    file: UploadFile = File(None)
):
    try:
        contents = [prompt]
        
        # Handle optional file uploads (documents, images, etc.)
        if file:
            file_bytes = await file.read()
            contents.append(
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=file.content_type
                )
            )

        # Persona instructions
        system_instructions = {
            "engineering": "You are Kiemaen AI configured as a professional Civil Engineering expert. Use precise calculations and technical standards.",
            "trading": "You are Kiemaen AI configured as an Institutional Trading core. Analyze market liquidity, technical patterns, and risk.",
            "coding": "You are Kiemaen AI configured as an advanced Software Engineering core. Provide clean, efficient code blocks.",
            "general": "You are Kiemaen AI, an autonomous intelligence core."
        }

        # Configure Google Search grounding tool for real-time live web access
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        config = types.GenerateContentConfig(
            system_instruction=system_instructions.get(persona, system_instructions["general"]),
            tools=[grounding_tool],  # Enables live web lookups beyond any training cutoff
            temperature=0.3
        )

        # Generate content with live search context enabled
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )

        return {"response": response.text}

    except Exception as e:
        return {"response": f"System Fault: {str(e)}"}
