from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


AgentName = Literal[
    "general",
    "research",
    "coding",
    "travel",
    "budget",
    "critic",
]


class ChatState(TypedDict, total=False):
    session_id: str
    user_message: str
    normalized_message: str
    intent: str
    task_queue: List[AgentName]
    active_agent: Optional[AgentName]
    memory: Dict[str, Any]
    collected_outputs: List[Dict[str, str]]
    final_answer: str
    critic_feedback: str
    needs_regeneration: bool
    trace: List[str]