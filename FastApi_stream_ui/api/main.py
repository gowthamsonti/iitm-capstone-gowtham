from fastapi import FastAPI
from .pipeline_model import Question, Answer
from  .pipeline_fapi_askllm import ask_llm
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="Capsstone API")

async def stream_answer(question: Question):
    answer = await ask_llm(question)
    for word in answer.text.split(" "):
        yield word +" "
        await asyncio.sleep(0.05)

#@app.post("/ask", response_model=Answer)
#async def ask(q:Question)->Answer:
#    return StreamingResponse(stream_answer(q),media_type="text/plain")

@app.post("/ask")
async def ask(q: Question):
    return StreamingResponse(stream_answer(q), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status":"ok"}