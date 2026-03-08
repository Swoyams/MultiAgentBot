from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List

import requests

from .state import AgentName, ChatState


ALLOWED_AGENTS: List[AgentName] = ["research", "coding", "travel", "budget"]


def _append_trace(state: ChatState, message: str) -> None:
    trace = state.setdefault("trace", [])
    trace.append(message)


def _recent_chat_history(memory: Dict[str, Any], max_turns: int = 5) -> str:
    history = memory.get("chat_history", [])[-max_turns:]
    if not history:
        return "(no prior chat history)"

    lines: List[str] = []
    for turn in history:
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")
        lines.append(f"User: {user}\nAssistant: {assistant}")
    return "\n\n".join(lines)


def _is_memory_recall_query(text: str) -> bool:
    recall_markers = ["remember", "earlier", "previous", "last chat", "last message", "what did i ask"]
    return any(marker in text for marker in recall_markers)


def _wants_detailed_output(text: str) -> bool:
    detail_markers = [
        "show code",
        "full code",
        "code snippet",
        "explain",
        "detailed",
        "step by step",
        "with trace",
        "debug details",
    ]
    return any(marker in text for marker in detail_markers)


def _build_memory_recall_answer(memory: Dict[str, Any]) -> str:
    history = memory.get("chat_history", [])
    if not history:
        return "I do not have any previous chat history in this session yet."

    recent = history[-3:]
    items = [f"{idx+1}. {turn.get('user', '').strip()}" for idx, turn in enumerate(recent)]
    return "Here are your most recent requests from this session:\n" + "\n".join(items)


def _invoke_llm(system_prompt: str, user_prompt: str, fallback: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return fallback

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=20,
        )
        if not response.ok:
            return fallback

        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() or fallback
    except requests.RequestException:
        return fallback


def preprocess_node(state: ChatState) -> ChatState:
    message = state["user_message"].strip()
    state["normalized_message"] = re.sub(r"\s+", " ", message.lower())
    _append_trace(state, "preprocess_node")
    return state


def memory_agent_node(state: ChatState) -> ChatState:
    memory = state.setdefault("memory", {})
    text = state.get("normalized_message", "")

    location_match = re.search(r"(?:to|visit|travel to)\s+([a-zA-Z\s]+)", text)
    if location_match:
        memory["preferred_location"] = location_match.group(1).strip().title()

    budget_match = re.search(r"(?:budget|under|around)\s+\$?(\d+[\d,]*)", text)
    if budget_match:
        memory["budget_range"] = budget_match.group(1).replace(",", "")

    lang_match = re.search(r"(?:python|javascript|typescript|java|go|rust)", text)
    if lang_match:
        memory["coding_language"] = lang_match.group(0).lower()

    interest_hits = [word for word in ["beach", "mountains", "ai", "museum", "food"] if word in text]
    if interest_hits:
        memory["interests"] = sorted(set(memory.get("interests", []) + interest_hits))

    _append_trace(state, "memory_agent_node")
    return state


def _heuristic_route(text: str) -> List[AgentName]:
    queue: List[AgentName] = []
    if any(key in text for key in ["research", "latest", "news", "what is", "find", "search"]):
        queue.append("research")
    if any(key in text for key in ["code", "bug", "python", "function", "api", "build"]):
        queue.append("coding")
    if any(key in text for key in ["travel", "trip", "itinerary", "flight", "hotel", "vacation"]):
        queue.append("travel")
    if any(key in text for key in ["budget", "cost", "convert", "currency", "optimize", "price"]):
        queue.append("budget")
    return queue or ["research"]


def supervisor_agent_node(state: ChatState) -> ChatState:
    text = state.get("normalized_message", "")
    memory = state.get("memory", {})
    history_context = _recent_chat_history(memory)

    routing_prompt = (
        "You are an orchestration supervisor for a multi-agent assistant. "
        "Choose from agents: research, coding, travel, budget. "
        "Return strict JSON with keys: agents (array), mode (general_search_only|full), reason (short). "
        "Use mode general_search_only if user clearly asks only search/research or general info and does not ask coding/travel/budget deliverables."
    )

    fallback_json = json.dumps({"agents": _heuristic_route(text), "mode": "full", "reason": "fallback"})
    raw = _invoke_llm(
        routing_prompt,
        f"Current user request:\n{state.get('user_message', '')}\n\nRecent chat history:\n{history_context}",
        fallback_json,
    )

    parsed: Dict[str, Any]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = json.loads(fallback_json)

    queue = [a for a in parsed.get("agents", []) if a in ALLOWED_AGENTS]
    if not queue:
        queue = _heuristic_route(text)

    mode = parsed.get("mode", "full")
    if mode == "general_search_only":
        queue = ["research"]

    state["intent"] = ",".join(queue)
    state["requested_agents"] = queue.copy()
    state["task_queue"] = queue
    _append_trace(state, f"supervisor_agent_node -> {queue}")
    return state


def dispatcher_node(state: ChatState) -> ChatState:
    queue = state.get("task_queue", [])
    state["active_agent"] = queue.pop(0) if queue else None
    state["task_queue"] = queue
    _append_trace(state, f"dispatcher_node -> {state['active_agent']}")
    return state


def research_agent_node(state: ChatState) -> ChatState:
    query = state.get("user_message", "")
    history_context = _recent_chat_history(state.get("memory", {}))
    wiki_output = "I could not find online context right now."
    try:
        response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(query[:80]),
            timeout=5,
            headers={"User-Agent": "MultiAgentBot/1.0"},
        )
        if response.ok:
            data = response.json()
            wiki_output = data.get("extract", wiki_output)
    except requests.RequestException:
        wiki_output = "Internet lookup was unavailable, so I used model reasoning only."

    content = _invoke_llm(
        "You are a research assistant. Provide a concise factual answer in max 6 bullet points.",
        f"Question: {query}\n\nContext:\n{wiki_output}\n\nRecent chat history:\n{history_context}",
        wiki_output,
    )

    state.setdefault("collected_outputs", []).append({"agent": "research", "content": content})
    _append_trace(state, "research_agent_node")
    return state


def coding_agent_node(state: ChatState) -> ChatState:
    language = state.get("memory", {}).get("coding_language", "python")
    prompt = state.get("user_message", "")
    history_context = _recent_chat_history(state.get("memory", {}))
    detailed = _wants_detailed_output(state.get("normalized_message", ""))
    coding_system_prompt = (
        "You are a senior software engineer. Return: approach, code, and test cases."
        if detailed
        else "You are a senior software engineer. Provide a concise solution summary only. "
        "Do not include code blocks, long explanations, or examples unless explicitly requested."
    )
    content = _invoke_llm(
        coding_system_prompt,
        f"Language: {language}\nTask: {prompt}\n\nRecent chat history:\n{history_context}",
        "I can help solve this coding task. If you want full code and explanation, ask: 'show code with steps'.",
    )
    state.setdefault("collected_outputs", []).append({"agent": "coding", "content": content})
    _append_trace(state, "coding_agent_node")
    return state


def travel_planner_agent_node(state: ChatState) -> ChatState:
    memory = state.get("memory", {})
    history_context = _recent_chat_history(memory)
    destination = memory.get("preferred_location", "the destination")
    budget = memory.get("budget_range", "flexible")
    interests = ", ".join(memory.get("interests", ["local highlights"]))

    fallback = (
        f"3-day itinerary for {destination}:\n"
        f"Day 1: Arrival and city orientation\n"
        f"Day 2: Core sights and food tour\n"
        f"Day 3: Leisure activities and departure\n"
        f"Budget target: {budget}."
    )

    content = _invoke_llm(
        "You are a travel planner. Return a practical day-wise itinerary with estimated costs.",
        f"User request: {state.get('user_message', '')}\nDestination hint: {destination}\nInterests: {interests}\nBudget: {budget}\n\nRecent chat history:\n{history_context}",
        fallback,
    )

    state.setdefault("collected_outputs", []).append({"agent": "travel", "content": content})
    _append_trace(state, "travel_planner_agent_node")
    return state


def _extract_numbers(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def budget_agent_node(state: ChatState) -> ChatState:
    text = state.get("normalized_message", "")
    history_context = _recent_chat_history(state.get("memory", {}))
    nums = _extract_numbers(text)

    if "convert" in text and nums:
        usd = nums[0]
        calc = (
            f"Currency conversion from ${usd:.2f}: "
            f"EUR {usd * 0.92:.2f}, INR {usd * 83.0:.2f}, GBP {usd * 0.78:.2f}."
        )
    elif "split" in text and len(nums) >= 2:
        total, people = nums[0], max(nums[1], 1)
        calc = f"Split optimization: total {total:.2f} / {people:.0f} people = {total/people:.2f} each."
    elif nums:
        calc = f"Detected values {nums} with combined total {math.fsum(nums):.2f}."
    else:
        calc = "No explicit numbers detected; ask user for amounts/currency." 

    content = _invoke_llm(
        "You are a finance assistant. Produce a short structured budget response.",
        f"Request: {state.get('user_message', '')}\nNumerical analysis: {calc}\n\nRecent chat history:\n{history_context}",
        calc,
    )

    state.setdefault("collected_outputs", []).append({"agent": "budget", "content": content})
    _append_trace(state, "budget_agent_node")
    return state


def critic_validator_agent_node(state: ChatState) -> ChatState:
    outputs = state.get("collected_outputs", [])
    needs_regeneration = False
    feedback: List[str] = []

    if not outputs:
        needs_regeneration = True
        feedback.append("No agent output found.")

    for item in outputs:
        if len(item.get("content", "").strip()) < 20:
            needs_regeneration = True
            feedback.append(f"{item['agent']} output too short.")

    state["needs_regeneration"] = needs_regeneration
    state["critic_feedback"] = " ".join(feedback) if feedback else "Validation passed."
    _append_trace(state, "critic_validator_agent_node")
    return state


def formatter_node(state: ChatState) -> ChatState:
    outputs = state.get("collected_outputs", [])
    requested_agents = state.get("requested_agents", [])
    memory = state.get("memory", {})

    if _is_memory_recall_query(state.get("normalized_message", "")):
        recall_answer = _build_memory_recall_answer(memory)
        state["final_answer"] = recall_answer
        state["structured_output"] = {"mode": "memory", "sections": [{"type": "memory", "content": recall_answer}]}
        _append_trace(state, "formatter_node")
        return state

    visible_outputs = [item for item in outputs if not requested_agents or item.get("agent") in requested_agents]
    if not visible_outputs:
        state["final_answer"] = _generate_fallback_answer(state.get("user_message", ""))
        state["structured_output"] = {"mode": "fallback", "sections": [{"type": "answer", "content": state["final_answer"]}]}
        _append_trace(state, "formatter_node")
        return state

    if requested_agents == ["research"]:
        answer = visible_outputs[0].get("content", "").strip()
        state["final_answer"] = answer
        state["structured_output"] = {"mode": "general_search_only", "sections": [{"type": "research", "content": answer}]}
        _append_trace(state, "formatter_node")
        return state

    synthesis_input = "\n\n".join(f"[{item['agent']}]\n{item['content']}" for item in visible_outputs)
    fallback = "\n\n".join(item["content"].strip() for item in visible_outputs if item.get("content"))
    detailed = _wants_detailed_output(state.get("normalized_message", ""))

    formatter_prompt = (
        "You are the final response formatter. Create a clean, user-facing answer only. "
        "Do not mention internal agents, routing, chain-of-thought, or hidden processing."
        if detailed
        else "You are the final response formatter. Create a short, clean final answer only. "
        "Do not include code blocks, internal traces, technical internals, or long explanations unless explicitly requested by the user."
    )

    clean_answer = _invoke_llm(
        formatter_prompt,
        synthesis_input,
        fallback,
    )

    state["final_answer"] = clean_answer
    state["structured_output"] = {
        "mode": "full",
        "sections": [{"type": item.get("agent", "unknown"), "content": item.get("content", "")} for item in visible_outputs],
    }
    _append_trace(state, "formatter_node")
    return state


def _generate_fallback_answer(question: str) -> str:
    q_lower = question.lower().strip()
    if any(greeting in q_lower for greeting in ["hello", "hi", "hey"]):
        return "Hello! I can help with research, coding, travel, and budget planning."
    return "Please share a bit more detail, and I will provide a focused answer."


def route_agent(state: ChatState) -> str:
    return state.get("active_agent") or "critic"


def route_after_agent(state: ChatState) -> str:
    return "dispatch" if state.get("task_queue") else "critic"
