from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]  
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

#  LangSmith tracing 
LANGSMITH_ENABLED = (
    os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    and bool(os.getenv("LANGCHAIN_API_KEY"))
)

if LANGSMITH_ENABLED:
    try:
        from langsmith import Client as LangSmithClient
        from langchain_core.tracers.langchain import LangChainTracer
        langsmith_client = LangSmithClient()
        LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "MultiAgentBot")
        print(f"[LangSmith] Tracing enabled → project: '{LANGSMITH_PROJECT}'")
    except ImportError:
        LANGSMITH_ENABLED = False
        langsmith_client = None
        print("[LangSmith] langsmith package not installed. Run: pip install langsmith")
else:
    langsmith_client = None
    LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "MultiAgentBot")
    print("[LangSmith] Tracing disabled. Add LANGCHAIN_TRACING_V2=true to .env to enable.")

from .graph import build_graph

#  App setup
app = FastAPI(title="MultiAgentBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = BASE_DIR / "frontend"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#  In-memory session store 
memory_store: Dict[str, Dict[str, Any]] = {}
workflow = build_graph()


# Schemas 
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[str]
    run_id: str | None = None


#  Routes 
@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "langsmith": "enabled" if LANGSMITH_ENABLED else "disabled",
    }


@app.get("/debug")
def debug() -> dict:
    return {
        "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "not set"),
        "LANGSMITH_ENABLED": LANGSMITH_ENABLED,
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "not set"),
        "LANGCHAIN_API_KEY_set": bool(os.getenv("LANGCHAIN_API_KEY")),
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
    run_id = str(uuid4())
    persisted_memory = memory_store.get(session_id, {})

    state = {
        "session_id": session_id,
        "user_message": req.message,
        "memory": persisted_memory,
        "collected_outputs": [],
        "trace": [],
        "retry_count": 0,
        "max_retries": 2,
        "failed_agents": [],
    }

    
    invoke_config: Dict[str, Any] = {
        "run_name": f"chat:{session_id[:8]}",
        "tags": ["multiagentbot", "chat"],
        "metadata": {
            "session_id": session_id,
            "run_id": run_id,
            "user_message_preview": req.message[:120],
        },
    }

    if LANGSMITH_ENABLED:
        from langchain_core.tracers.langchain import LangChainTracer
        tracer = LangChainTracer(project_name=LANGSMITH_PROJECT)
        invoke_config["callbacks"] = [tracer]

    result = workflow.invoke(state, config=invoke_config)
    memory_store[session_id] = result.get("memory", {})

    return ChatResponse(
        session_id=session_id,
        answer=result.get("final_answer", "No answer generated."),
        trace=result.get("trace", []),
        run_id=run_id if LANGSMITH_ENABLED else None,
    )