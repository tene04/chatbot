from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core import ChatBot  

app = FastAPI(title="RAG+LLM ChatBot API")

# Modelo para recibir la petición JSON
class QueryRequest(BaseModel):
    query: str

# Instanciamos y inicializamos el chatbot una sola vez
bot = ChatBot()
if not bot.initialize():
    raise RuntimeError("Failed to initialize ChatBot")

@app.post("/ask")
async def ask_chatbot(request: QueryRequest):
    try:
        response = bot.ask(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_document")
async def add_document():
    success = bot.add_document()
    if success:
        return {"message": "Document added successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to add document")
