import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS

app = FastAPI(title="J.A.R.V.I.S. API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class UserMessage(BaseModel):
    message: str

def buscar_en_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([r['body'] for r in results])
    except Exception as e:
        print(f"Error en búsqueda: {e}")
    return "No se encontraron resultados en tiempo real."

JARVIS_SYSTEM_PROMPT = """
Eres J.A.R.V.I.S., una inteligencia artificial de última generación.
Estamos en el año 2026. Tienes acceso a información en tiempo real.

Reglas obligatorias de comportamiento:
1. JAMÁS digas que tu base de datos llega a 2024, ni menciones a OpenAI.
2. Si el usuario te pregunta sobre noticias, clima o datos actuales, utiliza el contexto proporcionado para responder con precisión, elegancia británica y sarcasmo refinado.
3. Trata SIEMPRE al usuario como 'Señor' o 'Sir'.
4. Tus respuestas deben ser cortas (máximo 2 a 3 oraciones) para ser leídas en voz alta.
"""

@app.post("/api/chat")
async def chat_with_jarvis(data: UserMessage):
    try:
        if not data.message.strip():
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

        # Búsqueda en tiempo real
        contexto_web = buscar_en_web(data.message)

        user_content = f"Pregunta del usuario: {data.message}\nInformación en tiempo real encontrada: {contexto_web}"

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": JARVIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.6,
            max_tokens=200,
        )

        reply = completion.choices[0].message.content
        return {"response": reply}

    except Exception as e:
        print(f"Error: {e}")
        return {"response": "Señor, he detectado un fallo menor en mis circuitos de red. Por favor intente de nuevo."}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
