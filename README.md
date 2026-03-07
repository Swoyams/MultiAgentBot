# MultiAgentBot

A full-stack chatbot built with **LangGraph** that combines normal graph nodes and specialized agents:

1. **Supervisor Agent (mandatory brain)**
2. **Research Agent**
3. **Coding Agent**
4. **Travel Planner Agent**
5. **Budget/Calculator Agent**
6. **Memory Agent**
7. **Critic/Validator Agent**

---

## Project Folder Structure

```text
MultiAgentBot/
├── backend/
│   └── app/
│       ├── __init__.py          # Python package marker
│       ├── agents.py            # All agent + normal node logic
│       ├── graph.py             # LangGraph workflow wiring
│       ├── main.py              # FastAPI app + API endpoints
│       └── state.py             # Shared ChatState schema/types
├── frontend/
│   ├── index.html               # Chat UI page
│   ├── script.js                # Frontend chat behavior + API calls
│   └── style.css                # UI styling
├── tests/
│   └── test_api.py              # Basic API tests
├── .gitignore
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Architecture (High Level)

- **Normal nodes**: `preprocess`, `dispatch`, `formatter`
- **Agents**: `memory`, `supervisor`, `research`, `coding`, `travel`, `budget`, `critic`
- **Frontend**: Chat-like single page UI
- **Backend**: FastAPI with `/api/chat` and `/health`

---

## How to Run (Step by Step)

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd MultiAgentBot
```

### 2) Create a virtual environment

```bash
python -m venv .venv
```

### 3) Activate the virtual environment

**Linux/macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```


### 4.1) (Optional) Enable OpenAI-powered agent responses

If you want Research/Coding/Travel agents to use OpenAI instead of only local fallback logic, set:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o-mini"  # optional
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

### 5) Start backend server

```bash
uvicorn backend.app.main:app --reload
```

### 6) Open the app in browser

Go to:

```text
http://127.0.0.1:8000
```

### 7) Test the API quickly (optional)

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat endpoint example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan a 3-day trip to Paris under 1200 USD and convert 100 USD"}'
```

---

## How to Run Tests

```bash
pytest -q
```

---

## Notes

- Research agent uses Wikipedia summary API for lightweight web fetching.
- Memory is session-based in-memory storage.
- Critic agent validates output quality and flags issues.
- If dependency installation fails in restricted environments (proxy/index limits), run in a normal internet-enabled Python environment.
