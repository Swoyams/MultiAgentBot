from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

import requests

from .state import AgentName, ChatState


def _append_trace(state: ChatState, message: str) -> None:
    trace = state.setdefault("trace", [])
    trace.append(message)


def _openai_chat(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call OpenAI Responses API if OPENAI_API_KEY is available.

    Returns generated text on success, otherwise None so nodes can fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=20,
        )
        if not response.ok:
            return None

        payload = response.json()
        return payload.get("output_text")
    except requests.RequestException:
        return None


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


def supervisor_agent_node(state: ChatState) -> ChatState:
    text = state.get("normalized_message", "")
    queue: List[AgentName] = []

    if any(key in text for key in ["research", "latest", "news", "what is", "find", "document"]):
        queue.append("research")
    if any(key in text for key in ["code", "bug", "python", "function", "api", "build"]):
        queue.append("coding")
    if any(key in text for key in ["travel", "trip", "itinerary", "flight", "hotel", "vacation"]):
        queue.append("travel")
    if any(key in text for key in ["budget", "cost", "convert", "currency", "optimize", "price"]):
        queue.append("budget")

    if not queue:
        queue = ["research", "coding"]

    state["intent"] = ",".join(queue)
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

    llm_output = _openai_chat(
        "You are the Research Agent. Give concise factual context, include caveats when uncertain.",
        query,
    )
    if llm_output:
        output = llm_output
    else:
        output = "I could not find online context right now."
        try:
            response = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(query[:80]),
                timeout=5,
                headers={"User-Agent": "MultiAgentBot/1.0"},
            )
            if response.ok:
                data = response.json()
                output = data.get("extract", output)
        except requests.RequestException:
            output = "Internet lookup was unavailable, so I used internal reasoning only."

    state.setdefault("collected_outputs", []).append({"agent": "research", "content": output})
    _append_trace(state, "research_agent_node")
    return state


def coding_agent_node(state: ChatState) -> ChatState:
    language = state.get("memory", {}).get("coding_language", "python")
    prompt = state.get("user_message", "")

    llm_output = _openai_chat(
        f"You are the Coding Agent. Produce practical {language} guidance and a minimal working snippet.",
        prompt,
    )
    if llm_output:
        content = llm_output
    else:
        content = (
            f"Developer Agent ({language}) suggests: break the task into modules, add input validation, "
            f"write tests, and document APIs.\n\nStarter snippet:\n"
        )
        if language == "python":
            content += "def solve(user_input: str) -> str:\n    return user_input.strip()\n"
        elif language in {"javascript", "typescript"}:
            content += "function solve(userInput) { return userInput.trim(); }\n"
        else:
            content += f"// Build a minimal reusable function in {language}.\n"

        content += f"\nTask context: {prompt[:200]}"

    state.setdefault("collected_outputs", []).append({"agent": "coding", "content": content})
    _append_trace(state, "coding_agent_node")
    return state


def travel_planner_agent_node(state: ChatState) -> ChatState:
    memory = state.get("memory", {})
    destination = memory.get("preferred_location", "your destination")
    interests = ", ".join(memory.get("interests", ["local highlights"]))
    budget = memory.get("budget_range", "flexible")

    prompt = (
        f"Create a 3-day itinerary for {destination}. Interests: {interests}. "
        f"Budget preference: {budget}. Keep output practical and concise."
    )
    llm_output = _openai_chat("You are the Travel Planner Agent.", prompt)
    if llm_output:
        plan = llm_output
    else:
        plan = (
            f"3-day itinerary for {destination}:\n"
            f"Day 1: Arrival + city orientation around {interests}.\n"
            f"Day 2: Signature attractions + food tour.\n"
            f"Day 3: Free exploration + departure.\n"
            f"Target budget profile: {budget}."
        )

    state.setdefault("collected_outputs", []).append({"agent": "travel", "content": plan})
    _append_trace(state, "travel_planner_agent_node")
    return state


def _extract_numbers(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def budget_agent_node(state: ChatState) -> ChatState:
    text = state.get("normalized_message", "")
    nums = _extract_numbers(text)
    content = "Budget Agent: Provide numeric values to calculate totals, split plans, or currency conversion."

    rates = {"usd_to_eur": 0.92, "usd_to_inr": 83.0, "usd_to_gbp": 0.78}
    if "convert" in text and nums:
        usd = nums[0]
        content = (
            f"Currency conversion from ${usd:.2f}: EUR {usd * rates['usd_to_eur']:.2f}, "
            f"INR {usd * rates['usd_to_inr']:.2f}, GBP {usd * rates['usd_to_gbp']:.2f}."
        )
    elif "split" in text and len(nums) >= 2:
        total, people = nums[0], max(nums[1], 1)
        content = f"Split optimization: total {total:.2f} / {people:.0f} people = {total/people:.2f} each."
    elif nums:
        total = math.fsum(nums)
        content = f"Budget breakdown signal: detected values {nums} with combined total {total:.2f}."

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

    if any("nan" in item.get("content", "").lower() for item in outputs):
        needs_regeneration = True
        feedback.append("Detected invalid numeric output.")

    state["needs_regeneration"] = needs_regeneration
    state["critic_feedback"] = " ".join(feedback) if feedback else "Validation passed."
    _append_trace(state, "critic_validator_agent_node")
    return state


def formatter_node(state: ChatState) -> ChatState:
    sections = [f"### {item['agent'].title()} Agent\n{item['content']}" for item in state.get("collected_outputs", [])]
    critique = state.get("critic_feedback", "Validation passed.")
    memory = state.get("memory", {})
    memory_lines = "\n".join(f"- {k}: {v}" for k, v in memory.items()) or "- No preferences stored yet."

    state["final_answer"] = (
        "\n\n".join(sections)
        + "\n\n### Memory Snapshot\n"
        + memory_lines
        + "\n\n### Critic Review\n"
        + critique
    )
    _append_trace(state, "formatter_node")
    return state


def route_agent(state: ChatState) -> str:
    return state.get("active_agent") or "critic"


def route_after_agent(state: ChatState) -> str:
    return "dispatch" if state.get("task_queue") else "critic"
