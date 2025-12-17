# main.py
"""
CAG + RAG Chatbot FastAPI Server
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.cag_rag_chain import get_chain

app = FastAPI(title="CAG + RAG Chatbot")

# CAG-RAG 체인 인스턴스
cag_rag_chain = get_chain()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    cache_hit: Optional[bool] = None
    source: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """CAG → RAG 체인으로 질문에 답변"""
    inputs = {"question": req.question}

    result = cag_rag_chain.invoke(inputs)

    return ChatResponse(
        answer=result.get("answer", ""),
        cache_hit=result.get("cache_hit"),
        source=result.get("source"),
    )
