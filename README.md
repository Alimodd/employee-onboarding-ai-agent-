# Employee Onboarding AI Agent

An **educational** Employee Onboarding assistant that answers HR/onboarding
questions grounded in company policy documents, personalizes answers by employee
profile, and can open HR tickets — built as a single, understandable, local MVP.

> Synthetic data only. Not a production HR system. See **Known limitations**.

## Core stack

Python · FastAPI · Pydantic · Google Gemini (LLM + embeddings) · LangChain ·
Chroma (vector DB) · SQLite · Streamlit · Pytest

## Architecture

```text
User
  │
  ▼
Streamlit  ──HTTP──►  FastAPI  ──►  Onboarding Agent
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
             Profile Tool       Policy Tool        Ticket Tool
             (employees.json)        │                 │
                                     ▼                 ▼
                                    RAG              SQLite
                                     ▼
                                   Chroma  ◄── policy chunks + metadata
```

Two RAG pipelines exist side by side for learning: `app/rag/langchain_rag.py`
(application path) and `app/rag/manual_rag.py` (raw, for comparison). Full detail
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the study index in
[docs/LEARNING_MAP.md](docs/LEARNING_MAP.md).

## Setup

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2. Dependencies
pip install -r requirements.txt

# 3. Environment: copy the example and add your key (never commit .env)
copy .env.example .env            # Windows
#   then set API_key=<your Gemini key>
```

## Run

```bash
# Ingest documents into Chroma (safe to re-run; deterministic ids + upsert)
python ingest.py
#   or, once the API is running:  POST /documents/ingest

# Start the API
uvicorn app.main:app --reload      # http://localhost:8000  (/docs for Swagger)

# Start the UI (separate terminal)
streamlit run streamlit_app.py

# Tests (no live LLM calls; everything is mocked)
pytest -q

# Evaluation (LIVE: needs API key + ingested Chroma)
python -m evaluation.run_eval      # writes evaluation/results.json
```

## Example requests

```bash
# Generic policy question (cited)
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"What is the sick-leave procedure?"}'

# Employee-specific (profile-aware retrieval)
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"employee_id":101,"message":"Can I work remotely three days per week?"}'

# Unknown information (refuses instead of inventing)
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"How many stock options do I get?"}'

# HR ticket (only on explicit request; id from SQLite)
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"employee_id":101,"message":"Please open an HR ticket about parental leave."}'
```

`/chat` returns: `answer`, `sources` (real filenames from retrieval),
`tools_used`, and `ticket_id` (or `null`).

## Known limitations

- **Synthetic data** — fictional company, employees, and policies only.
- **Educational MVP**, local execution; not production-ready.
- **No authentication / authorization.** Metadata filtering personalizes
  relevance — it is **not** a security boundary.
- No real HR-system integration; a single-agent architecture.
- Policy versioning is metadata-only (all synthetic docs are v1.0).
- Retrieval quality depends on the documents; model wording can vary between runs.

## Repo notes

- `company_docs/`, `chroma_db/`, `data/tickets.db`, `.env`, and `*.docx` are
  gitignored (local only). Ingest locally to rebuild `chroma_db/`.
- The legacy Day 1-7 prompt-only app remains in the root `main.py` for reference;
  `app/main.py` is the current agent-based API.
