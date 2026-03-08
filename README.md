# MultiAgentBot

A full-stack **LangGraph** multi-agent chatbot with a central orchestrator and specialized agents:

1. Supervisor/Orchestrator Agent
2. Research Agent
3. Coding Agent
4. Travel Planner Agent
5. Budget Agent

## Key behavior

- Uses **OpenAI API** (via `langchain-openai`) for orchestration and response generation.
- Produces **clean user-facing output** only (no internal traces/agent routing in UI output).
- Returns a **structured output payload** from the API.
- If user asks for **general search only**, response is limited to research output only.
- Maintains **session memory** (recent chat history + extracted preferences) and can answer recall-style prompts.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
uvicorn backend.app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## API

`POST /api/chat`

Request:

```json
{
  "message": "Only do a general search: what is machine learning?",
  "session_id": null
}
```

Response:

```json
{
  "session_id": "...",
  "answer": "...",
  "structured_output": {
    "mode": "general_search_only",
    "sections": [
      {"type": "research", "content": "..."}
    ]
  }
}
```

## Tests

```bash
pytest -q
```
