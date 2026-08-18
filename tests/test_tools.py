"""Tests for the three agent tools (LLM/RAG mocked)."""

from __future__ import annotations

import pytest

from app.agent import tools
from app.database import tickets


# --------------------------------------------------------------------------- #
# Employee profile tool
# --------------------------------------------------------------------------- #
def test_profile_tool_found():
    result = tools.get_employee_profile(101)
    assert result["found"] is True
    assert result["profile"]["department"] == "Engineering"


def test_profile_tool_missing():
    result = tools.get_employee_profile(9999)
    assert result["found"] is False
    assert "not found" in result["error"].lower()


def test_profile_tool_malformed():
    result = tools.get_employee_profile("abc")
    assert result["found"] is False


# --------------------------------------------------------------------------- #
# Policy search tool (rag_answer mocked)
# --------------------------------------------------------------------------- #
def test_policy_tool_returns_sources(monkeypatch):
    def fake_rag(query, top_k=3, country=None, department=None):
        return {
            "answer": "You get 20 days.",
            "sources": ["annual_leave_policy.txt"],
            "retrieved_documents": [
                {"text": "20 working days...", "metadata": {"source": "annual_leave_policy.txt", "section": "Entitlement"}}
            ],
        }

    monkeypatch.setattr(tools, "rag_answer", fake_rag)
    result = tools.search_company_policy("annual leave", country="Belgium")
    assert result["sources"] == ["annual_leave_policy.txt"]
    assert result["evidence"][0]["source"] == "annual_leave_policy.txt"
    assert result["answer"]


def test_policy_tool_empty_query():
    result = tools.search_company_policy("   ")
    assert result["sources"] == []
    assert "error" in result


# --------------------------------------------------------------------------- #
# Ticket tool (temp DB, employee validated)
# --------------------------------------------------------------------------- #
def test_ticket_tool_creates(monkeypatch, tmp_path):
    monkeypatch.setattr(tickets, "TICKETS_DB_PATH", tmp_path / "t.db")
    result = tools.create_hr_ticket(101, "Parental leave", "Need info")
    assert result["ticket_id"] == 1
    assert result["status"] == "open"


def test_ticket_tool_rejects_unknown_employee(monkeypatch, tmp_path):
    monkeypatch.setattr(tickets, "TICKETS_DB_PATH", tmp_path / "t.db")
    result = tools.create_hr_ticket(9999, "Topic", "desc")
    assert "error" in result
    assert result.get("ticket_id") is None


def test_ticket_tool_invalid_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(tickets, "TICKETS_DB_PATH", tmp_path / "t.db")
    result = tools.create_hr_ticket(101, "", "desc")
    assert "error" in result
