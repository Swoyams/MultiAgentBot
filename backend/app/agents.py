from __future__ import annotations

import math
import os
import re
import time as _time
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

# ── Pipeline timing ──────────────────────────────────────────────────────────
# Note: _pipeline_start is a global — safe for single-threaded dev use.
# For production multi-user deployments, move timing into state dict instead.
_pipeline_start: float = 0.0

# ── Live exchange rate cache ─────────────────────────────────────────────────
_rate_cache: Dict[str, Any] = {}
_rate_cache_ts: float = 0.0
_CACHE_TTL: int = 3600  # refresh every 1 hour

# Currency symbols → ISO codes
_SYMBOL_MAP = {
    "$": "USD", "₹": "INR", "€": "EUR", "£": "GBP",
    "¥": "JPY", "¢": "USD", "₩": "KRW", "₺": "TRY",
}

# Currency name aliases
_NAME_MAP = {
    "usd": "USD", "dollar": "USD", "dollars": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "inr": "INR", "rupee": "INR", "rupees": "INR",
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "jpy": "JPY", "yen": "JPY",
    "aed": "AED", "dirham": "AED", "dirhams": "AED",
    "cad": "CAD", "aud": "AUD", "sgd": "SGD", "chf": "CHF",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _append_trace(state: ChatState, message: str) -> None:
    global _pipeline_start
    trace = state.setdefault("trace", [])
    elapsed_ms = round((_time.perf_counter() - _pipeline_start) * 1000)
    trace.append(f"{message} · {elapsed_ms}ms")


def _extract_numbers(text: str) -> List[float]:
    """Extract numbers from text, handling comma-formatted values like 40,000."""
    cleaned = re.sub(r'[₹$€£¥₩₺]', '', text)
    matches = re.findall(r'\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?', cleaned)
    results = []
    for m in matches:
        try:
            results.append(float(m.replace(',', '')))
        except ValueError:
            pass
    return results


def _get_live_rates(base: str = "USD") -> Dict[str, float]:
    """Fetch live exchange rates from open.er-api.com. Caches for 1 hour."""
    global _rate_cache, _rate_cache_ts
    now = _time.time()
    cache_key = f"rates_{base}"

    if cache_key in _rate_cache and (now - _rate_cache_ts) < _CACHE_TTL:
        return _rate_cache[cache_key]

    try:
        resp = requests.get(
            f"https://open.er-api.com/v6/latest/{base}",
            timeout=4,
            headers={"User-Agent": "MultiAgentBot/1.0"},
        )
        if resp.ok:
            data = resp.json()
            if data.get("result") == "success":
                rates = data.get("rates", {})
                _rate_cache[cache_key] = rates
                _rate_cache_ts = now
                return rates
    except Exception:
        pass

    # Hardcoded fallback if API unreachable
    return {
        "USD": 1.0, "EUR": 0.92, "INR": 84.0, "GBP": 0.78,
        "JPY": 149.0, "AED": 3.67, "CAD": 1.36, "AUD": 1.53,
        "SGD": 1.34, "CHF": 0.89, "KRW": 1325.0, "TRY": 32.0,
    }


def _detect_currency(text: str) -> str:
    """Detect source currency from message. Defaults to USD."""
    for symbol, code in _SYMBOL_MAP.items():
        if symbol in text:
            return code
    for name, code in _NAME_MAP.items():
        if name in text:
            return code
    return "USD"


def _detect_target_currencies(text: str) -> List[str]:
    """Detect which currencies the user wants to convert TO."""
    targets = []
    for name, code in _NAME_MAP.items():
        if name in text and code not in targets:
            targets.append(code)
    if not targets:
        targets = ["EUR", "INR", "GBP", "JPY"]
    return targets


def _calc_tip(total: float, pct: float, people: int = 1) -> str:
    tip = total * pct / 100
    grand = total + tip
    lines = [f"Bill: ${total:.2f} | Tip ({pct:.0f}%): ${tip:.2f} | Total: ${grand:.2f}"]
    if people > 1:
        lines.append(f"Per person ({people}): ${grand/people:.2f}")
    return "\n".join(lines)


def _calc_savings(goal: float, months: float) -> str:
    monthly = goal / max(months, 1)
    weekly = monthly / 4.33
    return (
        f"To save ${goal:,.2f} in {months:.0f} months:\n"
        f"• Monthly: ${monthly:,.2f}\n"
        f"• Weekly:  ${weekly:,.2f}"
    )


def _calc_percentage(pct: float, amount: float) -> str:
    result = amount * pct / 100
    return f"{pct:.0f}% of ${amount:.2f} = ${result:.2f}"


def _parse_budget_summary(travel_output: str) -> Dict[str, Any]:
    """Extract structured cost data from travel agent's ---BUDGET SUMMARY--- block."""
    result: Dict[str, Any] = {}
    block_match = re.search(
        r'---BUDGET SUMMARY---\s*(.*?)\s*---END BUDGET SUMMARY---',
        travel_output, re.DOTALL | re.IGNORECASE
    )
    if not block_match:
        return result

    block = block_match.group(1)
    for line in block.strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            num_match = re.search(r'[\d,]+(?:\.\d+)?', val.replace(',', ''))
            if num_match:
                try:
                    result[key] = float(num_match.group().replace(',', ''))
                except ValueError:
                    pass
            sym_match = re.search(r'[₹$€£¥]', val)
            if sym_match and 'currency' not in result:
                result['currency'] = sym_match.group()

    return result


def _fallback_travel(destination: str, interests: str, budget: str) -> str:
    return (
        f"3-day itinerary for {destination}:\n"
        f"Day 1: Arrival + city orientation around {interests}.\n"
        f"Day 2: Signature attractions + food tour.\n"
        f"Day 3: Free exploration + departure.\n"
        f"Target budget profile: {budget}."
    )


def _generate_fallback_answer(question: str) -> str:
    q_lower = question.lower().strip()
    if any(g in q_lower for g in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
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


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def preprocess_node(state: ChatState) -> ChatState:
    global _pipeline_start
    _pipeline_start = _time.perf_counter()
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

    budget_match = re.search(
        r'(?:budget|under|around|of|worth)\s*[:\s]*[₹$€£]?\s*(\d[\d,]*)' 
        r'|[₹$€£]\s*(\d[\d,]*)'
        r'|(\d[\d,]*)\s*(?:rupees?|inr|usd|eur|gbp)',
        text, re.IGNORECASE
    )
    if budget_match:
        raw = next(g for g in budget_match.groups() if g)
        memory["budget_range"] = raw.replace(",", "")

    lang_match = re.search(r"(?:python|javascript|typescript|java|go|rust)", text)
    if lang_match:
        memory["coding_language"] = lang_match.group(0).lower()

    interest_hits = [w for w in ["beach", "mountains", "ai", "museum", "food"] if w in text]
    if interest_hits:
        memory["interests"] = sorted(set(memory.get("interests", []) + interest_hits))

    _append_trace(state, "memory_agent_node")
    return state


def supervisor_agent_node(state: ChatState) -> ChatState:
    text = state.get("normalized_message", "")
    queue: List[AgentName] = []

    # ── Coding: require explicit action word + coding topic ──────────────────
    # Use space-padded text to avoid substring matches (e.g. "search" in "research")
    padded = f" {text} "

    coding_action = any(word in padded for word in [
        " write a ", " write me ", " build a ", " build me ", " create a ", " create me ",
        " fix ", " debug ", " implement ", " code for ", " function to ", " script to ",
        " how to code ", " how to build ", " how to implement ", " help me code ",
        " help me build ", " show me code ", " give me code ", " generate code ",
        " program to ", " program that ", " code to ", " code that ",
        " find all ", " print all ", " list all ", " calculate ", " compute ",
        " sort ", " check if ", " count ", " find prime ", " fibonacci ",
        " factorial ", " palindrome ", " reverse ", " traverse ",
    ])
    coding_topic = any(word in padded for word in [
        " code ", " bug ", " function ", " api ", " class ", " script ", " program ",
        " algorithm ", " snippet ", " module ", " library ", " loop ", " array ",
        " integer ", " prime ", " fibonacci ", " recursion ", " output ",
    ])
    explicit_language = any(
        f" {lang} " in padded or text.endswith(f" {lang}")
        for lang in ["python", "javascript", "typescript", "java", "golang", "rust"]
    )
    # Catch purely algorithmic/math requests even without action words
    pure_algo = any(f" {word} " in padded or text.endswith(word) for word in [
        "prime numbers", "fibonacci sequence", "factorial of", "palindrome check",
        "bubble sort", "binary search", "linked list", "hash map",
        "recursion", "dynamic programming",
    ])

    if (coding_action and coding_topic) or (coding_action and explicit_language) or pure_algo:
        queue.append("coding")

    # ── Research ─────────────────────────────────────────────────────────────
    if any(key in text for key in [
        "research", "latest", "news", "what is", "what are", "who is", "who was",
        "when did", "when was", "where is", "why is", "why does", "how does",
        "how did", "explain", "tell me about", "look up", "document",
        "capital of", "meaning of", "definition of", "history of", "facts about"
        # Note: "find" removed — too ambiguous, catches "find prime numbers" etc.
    ]):
        if "coding" not in queue:
            queue.append("research")

    # ── Travel ───────────────────────────────────────────────────────────────
    if any(key in text for key in [
        "travel", "trip", "itinerary", "flight", "hotel", "vacation",
        "visit", "tourism", "tour", "destination", "places to"
    ]):
        queue.append("travel")

    # ── Budget: only explicit financial operations, not travel budget mentions
    budget_action = any(key in text for key in [
        "convert", "currency", "exchange rate", "usd", "eur", "inr", "gbp",
        "split", "per person", "tip ", "gratuity",
        "remaining budget", "how much left", "how much will", "how much does",
        "total cost", "cost breakdown", "price breakdown",
    ])
    budget_standalone = (
        any(key in text for key in ["budget", "cost", "price", "how much", "optimize"])
        and "travel" not in queue
        and "trip" not in text
        and "itinerary" not in text
        and "visit" not in text
    )
    if budget_action or budget_standalone:
        queue.append("budget")

    # ── Fallback: ask OpenAI or default to general ───────────────────────────
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
                                "Reply with ONLY one word:\n"
                                "- 'general'  → greetings, opinions, simple chat\n"
                                "- 'research' → factual questions, definitions, history, news\n"
                                "- 'coding'   → ONLY if user explicitly wants code written or debugged\n"
                                "- 'travel'   → trip planning, destinations, hotels\n"
                                "- 'budget'   → money, costs, currency conversion\n\n"
                                "IMPORTANT: 'what is X' or 'capital of Y' is 'research', NOT 'coding'."
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
                            "Answer greetings, opinions, and simple questions in plain natural language. "
                            "Keep responses concise — 1 to 3 sentences unless more detail is needed. "
                            "Do NOT write code. Do NOT use bullet points for simple answers. "
                            "Respond naturally like a helpful person would."
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
                            "Answer questions accurately and in detail. Follow these rules:\n\n"
                            "1. For simple factual questions (capitals, dates, one-line facts): "
                            "give a direct 1–3 sentence answer — no padding.\n\n"
                            "2. For complex topics, people, events, concepts, or anything requiring "
                            "explanation: write a thorough response of around 400–500 words. "
                            "Structure it with clear paragraphs. Cover: what it is, background/history, "
                            "how it works or why it matters, key facts, and real-world impact or examples.\n\n"
                            "3. Never write code unless the question is explicitly about programming syntax.\n\n"
                            "4. Be specific — use real names, dates, numbers, and examples. "
                            "Never be vague or generic."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.5,
                max_tokens=700
            )
            output = response.choices[0].message.content if response.choices else f"Unable to retrieve information about '{query}'."
        except Exception as e:
            output = f"Error accessing OpenAI API: {str(e)}"
    else:
        try:
            search_query = re.sub(r"^(research|find|look up)\s+", "", query, flags=re.IGNORECASE).strip()
            response = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(search_query[:80]),
                timeout=5,
                headers={"User-Agent": "MultiAgentBot/1.0"},
            )
            if response.ok:
                data = response.json()
                extract = data.get("extract", "")
                output = extract if extract and len(extract.strip()) > 10 else f"No information found for '{query}'."
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
                            "Do not over-explain. Do not add unnecessary boilerplate. Be direct and practical."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            content = response.choices[0].message.content if response.choices else "Unable to generate code solution."
        except Exception as e:
            content = f"Error generating code solution: {str(e)}"
    else:
        content = f"Developer Agent ({language}) suggests: break the task into modules, add input validation, write tests.\n\n"
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
            # Include budget context in the prompt if available
            budget_context = ""
            if budget and budget != "flexible":
                budget_context = f" The user's total budget is {budget}."

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert travel planner who creates highly detailed, vivid itineraries. "
                            "For each day, provide a RICH breakdown with:\n"
                            "- Morning, Afternoon, and Evening sections\n"
                            "- Specific landmark/restaurant/hotel names (real places)\n"
                            "- Approximate timings (e.g. 8:00 AM - 10:00 AM)\n"
                            "- Short descriptions of what makes each place special\n"
                            "- Practical tips (best time to visit, what to order, how to get there)\n"
                            "- Estimated cost per activity where relevant\n\n"
                            "Format EXACTLY like this for each day:\n"
                            "Day N: [Theme/Title for the day]\n"
                            "Morning: [activity with time, description, tip]\n"
                            "Afternoon: [activity with time, description, tip]\n"
                            "Evening: [activity with time, description, tip]\n"
                            "Stay: [hotel recommendation with reason]\n"
                            "Local Tip: [one practical insider tip for the day]\n\n"
                            "Be specific, vivid and practical — not generic. "
                            "Name real streets, dishes, neighborhoods. "
                            "IMPORTANT: If the user mentions a budget, end with EXACTLY:\n\n"
                            "---BUDGET SUMMARY---\n"
                            "Total Budget: [amount with currency]\n"
                            "Accommodation: [amount]\n"
                            "Food: [amount]\n"
                            "Transport: [amount]\n"
                            "Activities: [amount]\n"
                            "Miscellaneous: [amount]\n"
                            "Total Estimated Spend: [sum of all above]\n"
                            "Remaining Budget: [Total Budget minus Total Estimated Spend]\n"
                            "---END BUDGET SUMMARY---\n\n"
                            "All amounts must be in the same currency the user specified. "
                            "Make sure the arithmetic is correct."
                        )
                    },
                    {"role": "user", "content": query + budget_context}
                ],
                temperature=0.75,
                max_tokens=1600
            )
            plan = response.choices[0].message.content if response.choices else _fallback_travel(destination, interests, budget)
        except Exception:
            plan = _fallback_travel(destination, interests, budget)
    else:
        plan = _fallback_travel(destination, interests, budget)

    state.setdefault("collected_outputs", []).append({"agent": "travel", "content": plan})
    _append_trace(state, "travel_planner_agent_node")
    return state


def budget_agent_node(state: ChatState) -> ChatState:
    query = state.get("user_message", "").strip()
    text = state.get("normalized_message", "")
    nums = _extract_numbers(text)
    content = "Please provide an amount so I can help with conversion, splitting, tips, or savings calculations."

    # Check if travel agent already ran — use its output as context
    collected = state.get("collected_outputs", [])
    travel_output = next((o["content"] for o in collected if o["agent"] == "travel"), None)

    # If travel output has structured budget block, verify math and reformat
    if travel_output:
        summary = _parse_budget_summary(travel_output)
        if summary:
            sym = summary.get("currency", "₹")
            total_budget = summary.get("total budget", 0)
            line_items = {k: v for k, v in summary.items()
                         if k not in ("total budget", "total estimated spend", "remaining budget", "currency")}
            recalculated_spend = math.fsum(line_items.values()) if line_items else summary.get("total estimated spend", 0)
            recalculated_remaining = total_budget - recalculated_spend

            lines = ["**Budget Breakdown**\n"]
            for item, amount in line_items.items():
                lines.append(f"• {item.title()}: {sym}{amount:,.0f}")
            lines.append(f"\n**Total Estimated Spend:** {sym}{recalculated_spend:,.0f}")
            lines.append(f"**Total Budget:** {sym}{total_budget:,.0f}")
            lines.append(f"**Remaining Budget:** {sym}{recalculated_remaining:,.0f}")

            if recalculated_remaining < 0:
                lines.append(f"\n⚠️ Over budget by {sym}{abs(recalculated_remaining):,.0f}")
            elif recalculated_remaining < total_budget * 0.1:
                lines.append(f"\n⚠️ Less than 10% budget remaining — plan carefully!")
            else:
                lines.append(f"\n✅ {sym}{recalculated_remaining:,.0f} buffer remaining")

            content = "\n".join(lines)
            state.setdefault("collected_outputs", []).append({"agent": "budget", "content": content})
            _append_trace(state, "budget_agent_node")
            return state

    # OpenAI path
    if OPENAI_AVAILABLE and client:
        try:
            if travel_output:
                user_prompt = (
                    f"The user asked: {query}\n\n"
                    f"Travel plan:\n{travel_output}\n\n"
                    f"Extract ALL cost line items, sum correctly, show: each item, total spend, total budget, remaining budget. "
                    f"Only use numbers explicitly stated — never invent figures."
                )
            else:
                user_prompt = query

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise budget and finance assistant. "
                            "Handle: currency conversions (approximate current market rates), "
                            "bill splitting, tip calculations, savings goals, percentage calculations, "
                            "and budget breakdowns. "
                            "When given a travel plan, extract every cost line item and sum correctly. "
                            "Double-check your arithmetic. Use only stated numbers — never invent figures. "
                            "Be concise and direct."
                        )
                    },
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=400
            )
            if response.choices:
                content = response.choices[0].message.content
                state.setdefault("collected_outputs", []).append({"agent": "budget", "content": content})
                _append_trace(state, "budget_agent_node")
                return state
        except Exception:
            pass

    # Rule-based fallback
    budget_pattern = re.search(
        r'(?:budget|₹|rs\.?|inr|usd|\$|€|£)\s*[\s:]?\s*(\d[\d,]*(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    budget_num = float(budget_pattern.group(1).replace(',', '')) if budget_pattern else (nums[0] if nums else None)

    convert_keywords = ["convert", "exchange", "change", "worth", "how much is", "rate"]
    split_keywords   = ["split", "divide", "share", "each", "per person", "among"]
    tip_keywords     = ["tip", "gratuity", "service charge"]
    save_keywords    = ["save", "saving", "goal", "months", "years", "weekly", "monthly"]
    pct_keywords     = ["percent", "%", "percentage", "of"]

    if any(k in text for k in convert_keywords) and nums:
        amount = nums[0]
        source = _detect_currency(text)
        rates = _get_live_rates(source)
        targets = [t for t in _detect_target_currencies(text) if t != source]
        if not targets:
            targets = [t for t in ["USD", "EUR", "INR", "GBP"] if t != source]
        symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "AED": "د.إ"}
        lines = [f"{source} {amount:,.2f} converts to (live rates):"]
        for t in targets[:6]:
            if t in rates:
                lines.append(f"• {t}: {symbols.get(t,'')}{amount * rates[t]:,.2f}")
        content = "\n".join(lines)

    elif any(k in text for k in tip_keywords) and nums:
        total = nums[0]
        pct = nums[1] if len(nums) > 1 else 15.0
        people = int(nums[2]) if len(nums) > 2 else 1
        content = _calc_tip(total, pct, people)

    elif any(k in text for k in save_keywords) and len(nums) >= 2:
        content = _calc_savings(nums[0], nums[1])

    elif any(k in text for k in pct_keywords) and len(nums) >= 2:
        content = _calc_percentage(nums[0], nums[1])

    elif any(k in text for k in split_keywords) and len(nums) >= 2:
        total, people = nums[0], max(nums[1], 1)
        content = f"${total:,.2f} split among {people:.0f} people = ${total/people:,.2f} each."

    elif nums:
        content = f"Numbers detected: {[round(n,2) for n in nums]} → Total: ${math.fsum(nums):,.2f}"

    state.setdefault("collected_outputs", []).append({"agent": "budget", "content": content})
    _append_trace(state, "budget_agent_node")
    return state


def critic_validator_agent_node(state: ChatState) -> ChatState:
    outputs = state.get("collected_outputs", [])
    needs_regeneration = False
    feedback: List[str] = []
    failed_agents: List[str] = []

    if not outputs:
        needs_regeneration = True
        feedback.append("No agent output found.")

    for item in outputs:
        agent = item.get("agent", "unknown")
        content = item.get("content", "").strip()

        if len(content) < 20:
            needs_regeneration = True
            failed_agents.append(agent)
            feedback.append(f"{agent} output too short ({len(content)} chars).")
        elif "nan" in content.lower():
            needs_regeneration = True
            failed_agents.append(agent)
            feedback.append(f"{agent} produced invalid numeric output.")
        elif content.lower().startswith("error:") or "traceback" in content.lower():
            needs_regeneration = True
            failed_agents.append(agent)
            feedback.append(f"{agent} returned an error response.")
        elif content.startswith("I've received your question"):
            needs_regeneration = True
            failed_agents.append(agent)
            feedback.append(f"{agent} returned a fallback non-answer.")

    state["needs_regeneration"] = needs_regeneration
    state["failed_agents"] = list(set(failed_agents))
    state["critic_feedback"] = " | ".join(feedback) if feedback else "Validation passed."
    _append_trace(state, f"critic_validator_agent_node → {'RETRY' if needs_regeneration else 'OK'}")
    return state


def retry_node(state: ChatState) -> ChatState:
    """Re-queues failed agents for another attempt."""
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    failed = state.get("failed_agents", [])
    feedback = state.get("critic_feedback", "")

    good_outputs = [o for o in state.get("collected_outputs", []) if o.get("agent") not in failed]
    state["collected_outputs"] = good_outputs
    state["task_queue"] = failed if failed else state.get("requested_agents", [])
    state["needs_regeneration"] = False
    state["failed_agents"] = []

    _append_trace(state, f"retry_node → attempt {retry_count} | re-queuing: {state['task_queue']} | reason: {feedback}")
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


# ── Routing functions ─────────────────────────────────────────────────────────

def route_agent(state: ChatState) -> str:
    return state.get("active_agent") or "critic"


def route_after_agent(state: ChatState) -> str:
    return "dispatch" if state.get("task_queue") else "critic"


def route_after_critic(state: ChatState) -> str:
    needs_regeneration = state.get("needs_regeneration", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if needs_regeneration and retry_count < max_retries:
        return "retry"
    return "formatter"