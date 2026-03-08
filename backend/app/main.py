from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI
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
    structured_output: Dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    persisted_memory = memory_store.get(session_id, {})
    persisted_memory.setdefault("chat_history", [])

    state = {
        "session_id": session_id,
        "user_message": req.message,
        "memory": persisted_memory,
        "collected_outputs": [],
        "trace": [],
    }

    result = workflow.invoke(state)
    updated_memory = result.get("memory", {})
    history = updated_memory.setdefault("chat_history", [])
    history.append(
        {
            "user": req.message,
            "assistant": result.get("final_answer", "No answer generated."),
        }
    )
    updated_memory["chat_history"] = history[-12:]
    memory_store[session_id] = updated_memory

    return ChatResponse(
        session_id=session_id,
        answer=result.get("final_answer", "No answer generated."),
        structured_output=result.get("structured_output", {"mode": "fallback", "sections": []}),
    )
