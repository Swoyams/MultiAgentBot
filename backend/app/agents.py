from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List

import requests

from .state import AgentName, ChatState

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
except Exception:
    OPENAI_AVAILABLE = False
    client = None


def _append_trace(state: ChatState, message: str) -> None:
    trace = state.setdefault("trace", [])
    trace.append(message)


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

    # ── Coding: only trigger on explicit coding intent, not general questions ──
    # Require action words like "write", "build", "fix", "debug", "create" alongside code keywords
    coding_action = any(word in text for word in [
        "write a", "write me", "build a", "build me", "create a", "create me",
        "fix ", "debug ", "implement ", "code for", "function to", "script to",
        "how to code", "how to build", "how to implement", "help me code",
        "help me build", "show me code", "give me code", "generate code"
    ])
    coding_topic = any(word in text for word in [
        "code", "bug", "function", "api", "class", "script", "program",
        "algorithm", "snippet", "module", "library"
    ])
    # Language alone (e.g. "python" in "what is python") should NOT trigger coding
    explicit_language = any(
        f"{lang} " in text or text.endswith(lang)
        for lang in ["python", "javascript", "typescript", "java", "golang", "rust"]
    )

    if (coding_action and coding_topic) or (coding_action and explicit_language):
        queue.append("coding")

    # ── Research: factual questions, lookups, news ──
    if any(key in text for key in [
        "research", "latest", "news", "what is", "what are", "who is", "who was",
        "when did", "when was", "where is", "why is", "why does", "how does",
        "how did", "explain", "tell me about", "find", "look up", "document",
        "capital of", "meaning of", "definition of", "history of", "facts about"
    ]):
        if "coding" not in queue:   # don't double-route "how to code X"
            queue.append("research")

    # ── Travel ──
    if any(key in text for key in [
        "travel", "trip", "itinerary", "flight", "hotel", "vacation",
        "visit", "tourism", "tour", "destination", "places to"
    ]):
        queue.append("travel")

    # ── Budget ──
    if any(key in text for key in [
        "budget", "cost", "convert", "currency", "optimize", "price",
        "how much", "split", "per person", "exchange rate", "usd", "eur", "inr"
    ]):
        queue.append("budget")

    # ── Fallback: ask OpenAI to route, or default to general ──
    if not queue:
        if OPENAI_AVAILABLE and client:
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a routing assistant. Given a user message, decide which agent should handle it.\n"
                                "Reply with ONLY one word from this list:\n"
                                "- 'general'  → greetings, opinions, simple chat, personal questions\n"
                                "- 'research' → factual questions, definitions, history, news, explanations\n"
                                "- 'coding'   → ONLY if the user explicitly wants code written or debugged\n"
                                "- 'travel'   → trip planning, destinations, hotels\n"
                                "- 'budget'   → money, costs, currency conversion\n\n"
                                "IMPORTANT: A question like 'what is X' or 'capital of Y' is 'research', NOT 'coding'."
                            )
                        },
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                    max_tokens=5
                )
                decision = resp.choices[0].message.content.strip().lower() if resp.choices else "general"
                for agent in ["research", "coding", "travel", "budget", "general"]:
                    if agent in decision:
                        queue = [agent]
                        break
                if not queue:
                    queue = ["general"]
            except Exception:
                queue = ["general"]
        else:
            queue = ["general"]

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


def general_agent_node(state: ChatState) -> ChatState:
    query = state.get("user_message", "").strip()

    if OPENAI_AVAILABLE and client:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly conversational assistant. "
                            "Answer greetings, opinions, and simple questions in plain, natural language. "
                            "Keep responses concise — 1 to 3 sentences unless more detail is clearly needed. "
                            "Do NOT write code. Do NOT use bullet points or headers for simple answers. "
                            "Just respond naturally like a helpful person would."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=200
            )
            output = response.choices[0].message.content if response.choices else "I'm not sure how to help with that."
        except Exception as e:
            output = f"Error: {str(e)}"
    else:
        output = _generate_fallback_answer(query)

    state.setdefault("collected_outputs", []).append({"agent": "general", "content": output})
    _append_trace(state, "general_agent_node")
    return state


def research_agent_node(state: ChatState) -> ChatState:
    query = state.get("user_message", "").strip()
    output = ""

    if OPENAI_AVAILABLE and client:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledgeable research assistant. "
                            "Answer questions accurately and concisely in plain prose. "
                            "For simple factual questions (e.g. capitals, dates, definitions), "
                            "give a direct 1–3 sentence answer — no code, no bullet lists, no unnecessary padding. "
                            "For complex topics, provide a clear structured explanation in paragraphs. "
                            "Never write code unless the question is explicitly about programming syntax."
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.5,
                max_tokens=400
            )
            if response.choices:
                output = response.choices[0].message.content
            else:
                output = f"Unable to retrieve information about '{query}'."
        except Exception as e:
            output = f"Error accessing OpenAI API: {str(e)}"
    else:
        try:
            # Strip any agent-mode prefix before searching Wikipedia
            search_query = re.sub(r"^(research|find|look up)\s+", "", query, flags=re.IGNORECASE).strip()
            response = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(search_query[:80]),
                timeout=5,
                headers={"User-Agent": "MultiAgentBot/1.0"},
            )
            if response.ok:
                data = response.json()
                extract = data.get("extract", "")
                output = extract if extract and len(extract.strip()) > 10 else f"No detailed information found for '{query}'."
            else:
                output = f"Could not find information about '{query}'."
        except requests.RequestException:
            output = f"Unable to retrieve online information for '{query}'."
        except Exception:
            output = f"Error retrieving information about '{query}'."

    if not output:
        output = f"Processed query: {query}"

    state.setdefault("collected_outputs", []).append({"agent": "research", "content": output})
    _append_trace(state, "research_agent_node")
    return state


def coding_agent_node(state: ChatState) -> ChatState:
    language = state.get("memory", {}).get("coding_language", "python")
    prompt = state.get("user_message", "")

    if OPENAI_AVAILABLE and client:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an expert software developer specializing in {language}. "
                            "Provide clean, well-commented code solutions. "
                            "Structure your response as: "
                            "1) A brief one-sentence explanation of your approach, "
                            "2) The code block, "
                            "3) A short explanation of key parts only if non-obvious. "
                            "Do not over-explain. Do not add unnecessary boilerplate. "
                            "Be direct and practical."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=800
            )
            if response.choices:
                content = response.choices[0].message.content
            else:
                content = "Unable to generate code solution."
        except Exception as e:
            content = f"Error generating code solution: {str(e)}"
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
    query = state.get("user_message", "").strip()
    memory = state.get("memory", {})
    destination = memory.get("preferred_location", "your destination")
    interests = ", ".join(memory.get("interests", ["local highlights"]))
    budget = memory.get("budget_range", "flexible")

    if OPENAI_AVAILABLE and client:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert travel planner. "
                            "Create practical, personalized travel itineraries and advice. "
                            "Be specific with place names, timings, and tips. "
                            "Format itineraries clearly by day. Keep recommendations realistic and helpful. "
                            "Do not write code or unrelated content."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=600
            )
            plan = response.choices[0].message.content if response.choices else _fallback_travel(destination, interests, budget)
        except Exception:
            plan = _fallback_travel(destination, interests, budget)
    else:
        plan = _fallback_travel(destination, interests, budget)

    state.setdefault("collected_outputs", []).append({"agent": "travel", "content": plan})
    _append_trace(state, "travel_planner_agent_node")
    return state


def _fallback_travel(destination: str, interests: str, budget: str) -> str:
    return (
        f"3-day itinerary for {destination}:\n"
        f"Day 1: Arrival + city orientation around {interests}.\n"
        f"Day 2: Signature attractions + food tour.\n"
        f"Day 3: Free exploration + departure.\n"
        f"Target budget profile: {budget}."
    )


def _extract_numbers(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def budget_agent_node(state: ChatState) -> ChatState:
    text = state.get("normalized_message", "")
    nums = _extract_numbers(text)
    content = "Please provide numeric values so I can calculate totals, split costs, or convert currencies."

    rates = {"usd_to_eur": 0.92, "usd_to_inr": 83.0, "usd_to_gbp": 0.78}
    if "convert" in text and nums:
        usd = nums[0]
        content = (
            f"${usd:.2f} converts to:\n"
            f"• EUR: €{usd * rates['usd_to_eur']:.2f}\n"
            f"• INR: ₹{usd * rates['usd_to_inr']:.2f}\n"
            f"• GBP: £{usd * rates['usd_to_gbp']:.2f}"
        )
    elif "split" in text and len(nums) >= 2:
        total, people = nums[0], max(nums[1], 1)
        content = f"${total:.2f} split among {people:.0f} people = ${total/people:.2f} each."
    elif nums:
        total = math.fsum(nums)
        content = f"Detected values: {nums} — combined total: ${total:.2f}."

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
    outputs = state.get("collected_outputs", [])
    user_message = state.get("user_message", "")

    if outputs:
        answer_parts = [item.get("content", "").strip() for item in outputs if item.get("content", "").strip()]
        state["final_answer"] = "\n\n".join(answer_parts) if answer_parts else _generate_fallback_answer(user_message)
    else:
        state["final_answer"] = _generate_fallback_answer(user_message)

    _append_trace(state, "formatter_node")
    return state


def _generate_fallback_answer(question: str) -> str:
    q_lower = question.lower().strip()

    if any(greeting in q_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        return "Hello! I'm MultiAgentBot. How can I help you today?"

    if any(h in q_lower for h in ["help", "what can you do"]):
        return (
            "I have specialized agents for:\n"
            "- **Research**: Facts, news, explanations\n"
            "- **Coding**: Code solutions in any language\n"
            "- **Travel**: Trip planning and itineraries\n"
            "- **Budget**: Currency conversion and cost splitting\n\n"
            "Just ask me anything!"
        )

    return f"I've received your question: '{question}'. Try asking about research topics, coding problems, travel plans, or budget calculations."


def route_agent(state: ChatState) -> str:
    return state.get("active_agent") or "critic"


def route_after_agent(state: ChatState) -> str:
    return "dispatch" if state.get("task_queue") else "critic"