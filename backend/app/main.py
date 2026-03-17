from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from dotenv import load_dotenv

# ── Load .env from project root (MultiAgentBot-main/.env) ──────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # MultiAgentBot-main/
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .graph import build_graph

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="MultiAgentBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (frontend) ─────────────────────────────────────────────────
STATIC_DIR = BASE_DIR / "frontend"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── In-memory session store ─────────────────────────────────────────────────
memory_store: Dict[str, Dict[str, Any]] = {}
workflow = build_graph()


# ── Schemas ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[str]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug")
def debug() -> dict:
    """Check env vars and paths — remove this in production."""
    return {
        "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "not set"),
        "BASE_DIR": str(BASE_DIR),
        "STATIC_DIR": str(STATIC_DIR),
        "STATIC_DIR_exists": STATIC_DIR.exists(),
        "frontend_index_exists": (STATIC_DIR / "index.html").exists(),
        "env_file_exists": (BASE_DIR / ".env").exists(),
    }


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found. Place index.html in the frontend/ folder."}


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