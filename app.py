from fastapi import FastAPI, File, Form, UploadFile
from google import genai
from google.genai import types

app = FastAPI()
client = genai.Client()  # Uses GEMINI_API_KEY from your Render Environment variables

@app.post("/generate")
async def generate_response(
    prompt: str = Form(...),
    persona: str = Form("general"),
    file: UploadFile = File(None)
):
    try:
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

        # Enable real-time web search grounding
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
