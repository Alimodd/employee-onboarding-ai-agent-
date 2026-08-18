# Demo (2-3 minutes)

Prerequisites: `.env` with a valid `API_key`, dependencies installed, and the
company documents ingested at least once.

```bash
# terminal 1 - backend
uvicorn app.main:app --reload

# terminal 2 - one-time ingestion (safe to re-run)
curl -X POST http://localhost:8000/documents/ingest
```

## 1. Health
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 2. Policy question (cited)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the general sick-leave procedure?"}'
# answer + "sources":["sick_leave_policy.txt"] + tools_used:["search_company_policy"]
```

## 3. Employee-specific question (profile-aware)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"employee_id":101,"message":"Can I work remotely three days per week?"}'
# tools_used includes get_employee_profile AND search_company_policy
# sources: ["remote_work_policy.txt"]
```

## 4. Unknown information (refusal — no invention)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"employee_id":101,"message":"How many stock options do I receive?"}'
# answer: "I could not find this information in the available company documents."
# sources: []   ticket_id: null
```

## 5. Ticket creation (explicit request)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"employee_id":101,"message":"Please open an HR ticket because I cannot find the parental leave policy."}'
# ticket_id: <number from SQLite>
```

## 6. Ticket appears in SQLite
```bash
curl http://localhost:8000/tickets
# [{"ticket_id":1,"employee_id":101,"topic":"...","status":"open",...}]
```

## 7. Employee profile
```bash
curl http://localhost:8000/employees/101   # 200 profile
curl http://localhost:8000/employees/9999  # 404
```

## 8. Streamlit UI
```bash
streamlit run streamlit_app.py
```
Enter an employee ID, ask a policy question, watch the answer, sources, and
tools-used render; request a ticket and see the confirmation.
