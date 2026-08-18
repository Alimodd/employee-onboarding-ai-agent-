# Learning Map (Days 14-30)

A map from each learning day to the code that implements it. This is a study
index, not a tutorial. For each component: files, key functions, input/output,
external dependency, stored data, main failure cases.

---

## Day 14 - Citations + starter evaluation
- **Files:** `app/rag/langchain_rag.py`, `app/rag/manual_rag.py`, `evaluation/dataset.json`
- **Key funcs:** `extract_sources()`, `rag_answer()`
- **In:** question → **Out:** answer + real `sources` from retrieved metadata
- **External:** Gemini (LLM) + Chroma
- **Stored:** none (dataset is static)
- **Fails:** blank question (ValueError); provider errors propagate

## Day 15 - LangChain model wrapper
- **Files:** `app/services/langchain_client.py` (raw SDK stays in `model_client.py`)
- **Key funcs:** `get_chat_model()`, `invoke_model()`, `_normalize_text()`
- **In:** prompt string → **Out:** normalized answer string
- **External:** `langchain-google-genai` → Gemini
- **Stored:** none
- **Fails:** missing key (ConfigurationError); provider error (ProviderRequestError); empty output (EmptyModelResponseError)

## Day 16 - LangChain documents + splitting
- **Files:** `app/rag/langchain_documents.py` (reuses root `document_loader`, `chunker`)
- **Key funcs:** `load_langchain_documents()`, `split_documents()`, `load_and_split()`
- **In:** company_docs files → **Out:** list of LangChain `Document` chunks with metadata
- **External:** `langchain-text-splitters`
- **Stored:** none (in memory)
- **Fails:** empty file skipped; missing metadata filled with "unknown"

## Day 17 - LangChain + Chroma retriever
- **Files:** `app/rag/langchain_retriever.py`
- **Key funcs:** `get_vectorstore()`, `get_retriever()`, `retrieve_documents()`, `build_profile_filter()`
- **In:** query (+ optional country/department/document_type) → **Out:** list of LangChain Documents
- **External:** Gemini embeddings + Chroma (reuses existing `company_policies` collection)
- **Stored:** none (read-only)
- **Fails:** blank query / non-positive top_k (ValueError); embedding/key errors propagate

## Day 18 - LangChain RAG pipeline
- **Files:** `app/rag/langchain_rag.py`
- **Key funcs:** `format_context()`, `build_rag_prompt()`, `rag_answer()`
- **In:** question → **Out:** `{answer, sources, retrieved_documents}`
- **External:** retriever + LLM
- **Stored:** none
- **Fails:** no docs → refusal message; provider errors propagate

## Day 19 - Three agent tools
- **Files:** `app/agent/tools.py`, `app/employees.py`, `app/database/tickets.py`
- **Key funcs:** `get_employee_profile()`, `search_company_policy()`, `create_hr_ticket()` + `LC_TOOLS`, `TOOL_DISPATCH`
- **In:** typed args → **Out:** structured dicts (profile / answer+sources / ticket)
- **External:** RAG (policy tool), SQLite (ticket tool)
- **Stored:** tickets tool writes to SQLite
- **Fails:** unknown/invalid employee → error dict; empty ticket fields → error dict

## Day 20 - Tool-calling agent
- **Files:** `app/agent/onboarding_agent.py`
- **Key funcs:** `run_agent()`
- **In:** message (+ optional employee_id) → **Out:** `{answer, sources, tools_used, ticket_id}`
- **External:** LangChain chat model with `bind_tools`
- **Stored:** indirectly (ticket tool)
- **Fails:** missing key (ConfigurationError propagates)

## Day 21 - Debugging + traceability
- **Files:** `app/agent/onboarding_agent.py`
- **Key funcs:** tool-call logging inside `run_agent`, `_tool_signature()` (dedup), `MAX_AGENT_ITERATIONS`
- **In/Out:** same as Day 20 plus `tools_used`
- **Stored:** logs only (no secrets, no descriptions)
- **Fails:** tool exceptions caught → error result, never silently swallowed; iteration cap → safe fallback

## Day 22 - Final structure
- **Files:** the `app/` package (`api/`, `rag/`, `agent/`, `database/`, `models/`, `services/`)
- Root modules (`model_client`, `vector_store`, `retriever`, `chunker`, `document_loader`) reused, not duplicated.

## Day 23 - Final FastAPI
- **Files:** `app/main.py`, `app/api/routes.py`, `app/models/schemas.py`, `app/rag/ingestion.py`
- **Endpoints:** `GET /health`, `POST /documents/ingest`, `POST /chat`, `GET /employees/{id}`, `GET /tickets`
- **In/Out:** Pydantic models; **Stored:** ingestion → Chroma, tickets → SQLite
- **Fails:** provider/config errors → safe HTTP codes (no stack traces)

## Day 24 - Profile-aware retrieval
- **Files:** `app/rag/langchain_retriever.py` (`build_profile_filter`), used by `search_company_policy` + agent
- **Behavior:** country/department match that value OR "All"; another country's exclusive policy is excluded
- **Note:** filtering is relevance personalization, NOT authorization

## Day 25 - SQLite ticket persistence
- **Files:** `app/database/tickets.py`
- **Key funcs:** `init_db()`, `create_ticket()`, `get_all_tickets()`, `_find_duplicate()`
- **In:** employee_id, topic, description → **Out:** ticket dict with SQLite-generated id
- **External:** `sqlite3`; **Stored:** `data/tickets.db` (persists across restarts)
- **Fails:** blank fields / DB errors → TicketError; duplicate open ticket → returns existing

## Day 26 - Streamlit frontend
- **Files:** `streamlit_app.py`
- **In:** employee id + message → **Out:** rendered answer/sources/tools/ticket
- **External:** FastAPI over HTTP (never imports agent code)
- **Fails:** backend down / timeout / 422 handled with user messages

## Day 27 - End-to-end evaluation
- **Files:** `evaluation/dataset.json` (30 cases), `evaluation/run_eval.py`
- **Key funcs:** `evaluate_case()`, `main()`
- **In:** dataset → **Out:** `results.json` (pass/fail + failure_category + notes)
- **External:** live agent (Gemini); **Stored:** `evaluation/results.json`
- **Fails:** per-case exceptions recorded as `api_failure`

## Day 28 - Failure handling + hardening
- **Files:** across `app/` (routes error mapping, tools error dicts, tickets TicketError,
  retriever empty-collection handling, model client exception hierarchy)
- **Behavior:** safe structured errors, no stack traces to clients, no secrets in logs

## Day 29 - README + demo docs
- **Files:** `README.md`, `docs/DEMO.md`, `docs/ARCHITECTURE.md`

## Day 30 - Implementation review artifacts
- **Files:** this `docs/LEARNING_MAP.md` (Days 14-30 mapped to code)
