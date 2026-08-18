"""API tests for the agent-based FastAPI app (agent mocked, temp ticket DB)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.database import tickets
from app.main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Redirect the ticket DB so tests never touch the real file.
    monkeypatch.setattr(tickets, "TICKETS_DB_PATH", tmp_path / "tickets.db")
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_success(client, monkeypatch):
    monkeypatch.setattr(routes, "run_agent", lambda message, employee_id=None: {
        "answer": "You get 20 days.",
        "sources": ["annual_leave_policy.txt"],
        "tools_used": ["search_company_policy"],
        "ticket_id": None,
    })
    r = client.post("/chat", json={"message": "How much annual leave?", "employee_id": 101})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] == ["annual_leave_policy.txt"]
    assert body["tools_used"] == ["search_company_policy"]
    assert body["ticket_id"] is None


def test_chat_rejects_blank_message(client):
    r = client.post("/chat", json={"message": "   "})
    assert r.status_code == 422


def test_chat_requires_message(client):
    r = client.post("/chat", json={"employee_id": 101})
    assert r.status_code == 422


def test_get_employee_found(client):
    r = client.get("/employees/101")
    assert r.status_code == 200
    assert r.json()["name"] == "Anke De Vries"


def test_get_employee_not_found(client):
    r = client.get("/employees/9999")
    assert r.status_code == 404


def test_tickets_endpoint_empty(client):
    r = client.get("/tickets")
    assert r.status_code == 200
    assert r.json() == []


def test_tickets_endpoint_after_insert(client):
    tickets.create_ticket(101, "Parental leave", "need info")
    r = client.get("/tickets")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["employee_id"] == 101
