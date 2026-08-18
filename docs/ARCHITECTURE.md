# Architecture

Educational Employee Onboarding AI Agent. Everything is synthetic and local.

```text
User
  │
  ▼
Streamlit UI  (streamlit_app.py)
  │  HTTP (requests)
  ▼
FastAPI        (app/main.py + app/api/routes.py)
  │
  ▼
Onboarding Agent  (app/agent/onboarding_agent.py)   ── explicit bind_tools loop
  │
  ├── get_employee_profile ── app/employees.py ── data/employees.json
  │
  ├── search_company_policy ── app/rag/langchain_rag.py
  │        │
  │        ├── retriever ── app/rag/langchain_retriever.py ── Chroma (chroma_db/)
  │        └── LLM ─────── app/services/langchain_client.py ── Gemini
  │
  └── create_hr_ticket ── app/database/tickets.py ── SQLite (data/tickets.db)
```

## Layers

- **FastAPI layer** (`app/api/routes.py`, `app/main.py`) — HTTP endpoints, Pydantic
  request/response models, safe error translation (no stack traces to clients).
- **Model client** — two side-by-side clients over the *same* Gemini key:
  - `model_client.py` (root) — raw google-genai SDK (used by manual RAG).
  - `app/services/langchain_client.py` — LangChain `ChatGoogleGenerativeAI`.
  The API key is read only via `model_client._get_api_key()` and is never logged.
- **RAG** — ingestion (`app/rag/ingestion.py`, reusing root loader/chunker/vector_store),
  a LangChain retriever over the existing Chroma collection
  (`app/rag/langchain_retriever.py`), and two RAG pipelines that produce grounded
  answers with **real citations** from retrieved metadata:
  - `app/rag/langchain_rag.py` (application path)
  - `app/rag/manual_rag.py` (kept for comparison)
- **Chroma** — the single persistent vector DB in `chroma_db/`. LangChain is only
  the orchestration layer; it does not create a second database.
- **Tools** (`app/agent/tools.py`) — three deterministic functions plus LangChain
  tool schemas: profile lookup, policy search, ticket creation.
- **Agent** (`app/agent/onboarding_agent.py`) — one tool-calling agent. Explicit
  loop with `tools_used` tracking, duplicate-call suppression, tool-call logging,
  and a max-iteration cap. No LangGraph.
- **SQLite** (`app/database/tickets.py`) — ticket persistence with SQLite-generated
  ids and deterministic duplicate suppression.
- **Streamlit** (`streamlit_app.py`) — thin UI; talks to FastAPI over HTTP only.
- **Evaluation** (`evaluation/`) — 30-case dataset + deterministic runner writing
  `results.json` with pass/fail and failure categories.

## Key design rules

- Sources are extracted from retrieved document metadata, never invented by the LLM.
- Missing evidence → the fixed refusal message; the agent never fabricates policy.
- Tickets are created only on explicit user request; the id always comes from SQLite.
- **Metadata filtering is relevance personalization, not authorization / security.**
- Policy version + effective date are preserved on chunks; when versions conflict the
  prompt prefers the most recent effective date. The current synthetic documents are
  all version 1.0, so this is metadata-preserving only (documented limitation).
