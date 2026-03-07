from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .graph import build_graph

app = FastAPI(title="MultiAgentBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

memory_store: Dict[str, Dict[str, Any]] = {}
workflow = build_graph()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[str]


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    base = str(request.base_url).rstrip("/")
    return {
        "status": "ok",
        "message": "Service is healthy. Open the chatbot UI at '/' or '/chat'.",
        "ui_url": f"{base}/",
        "chat_alias_url": f"{base}/chat",
        "api_docs_url": f"{base}/docs",
    }


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found"}


@app.get("/chat")
def chat_page():
    return index()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    persisted_memory = memory_store.get(session_id, {})

    state = {
        "session_id": session_id,
        "user_message": req.message,
        "memory": persisted_memory,
        "collected_outputs": [],
        "trace": [],
    }

    result = workflow.invoke(state)
    memory_store[session_id] = result.get("memory", {})

    return ChatResponse(
        session_id=session_id,
        answer=result.get("final_answer", "No answer generated."),
        trace=result.get("trace", []),
    )
