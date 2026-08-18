"""Tests for the onboarding agent routing (chat model + tools mocked).

A scripted fake chat model replaces the real LLM so these tests are fast,
deterministic, and free. Tool execution is redirected to simple fakes.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent import onboarding_agent


class FakeChat:
    """A chat model that returns a pre-scripted sequence of AIMessages."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self._i = 0

    def bind_tools(self, tools):
        return self  # ignore tools; we drive behaviour via the script

    def invoke(self, messages):
        msg = self._scripted[self._i]
        self._i += 1
        return msg


def _use_fake(monkeypatch, scripted, dispatch=None):
    monkeypatch.setattr(onboarding_agent, "get_chat_model", lambda: FakeChat(scripted))
    if dispatch is not None:
        monkeypatch.setattr(onboarding_agent, "TOOL_DISPATCH", dispatch)


def _tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])


# --------------------------------------------------------------------------- #
# Policy-only routing
# --------------------------------------------------------------------------- #
def test_policy_only_routing(monkeypatch):
    scripted = [
        _tool_call("search_company_policy_tool", {"query": "remote work"}),
        AIMessage(content="You may work remotely up to two days per week."),
    ]
    dispatch = {
        "search_company_policy_tool": lambda **kw: {
            "answer": "two days", "sources": ["remote_work_policy.txt"], "evidence": []
        }
    }
    _use_fake(monkeypatch, scripted, dispatch)

    result = onboarding_agent.run_agent("How many remote days can I work?")
    assert "search_company_policy" in result["tools_used"]
    assert result["sources"] == ["remote_work_policy.txt"]
    assert result["ticket_id"] is None


# --------------------------------------------------------------------------- #
# Profile + policy routing (employee_id supplied)
# --------------------------------------------------------------------------- #
def test_profile_and_policy_routing(monkeypatch):
    scripted = [
        _tool_call("search_company_policy_tool", {"query": "remote work"}),
        AIMessage(content="Based on your Belgium/Engineering profile: two days."),
    ]
    dispatch = {
        "search_company_policy_tool": lambda **kw: {
            "answer": "two days", "sources": ["remote_work_policy.txt"], "evidence": []
        }
    }
    _use_fake(monkeypatch, scripted, dispatch)

    result = onboarding_agent.run_agent("Can I work remotely three days?", employee_id=101)
    assert "get_employee_profile" in result["tools_used"]
    assert "search_company_policy" in result["tools_used"]
    assert result["sources"] == ["remote_work_policy.txt"]


# --------------------------------------------------------------------------- #
# Explicit ticket creation
# --------------------------------------------------------------------------- #
def test_explicit_ticket_creation(monkeypatch):
    scripted = [
        _tool_call("create_hr_ticket_tool",
                   {"employee_id": 101, "topic": "Parental leave", "description": "not found"}),
        AIMessage(content="I have created ticket #7 for you."),
    ]
    dispatch = {
        "create_hr_ticket_tool": lambda **kw: {
            "ticket_id": 7, "status": "open", "duplicate": False
        }
    }
    _use_fake(monkeypatch, scripted, dispatch)

    result = onboarding_agent.run_agent("Please open an HR ticket about parental leave.")
    assert result["ticket_id"] == 7
    assert "create_hr_ticket" in result["tools_used"]


# --------------------------------------------------------------------------- #
# No accidental ticket / no forced tool call
# --------------------------------------------------------------------------- #
def test_no_accidental_ticket(monkeypatch):
    scripted = [AIMessage(content="Hello! How can I help you today?")]
    _use_fake(monkeypatch, scripted, dispatch={})
    result = onboarding_agent.run_agent("Hi there")
    assert result["ticket_id"] is None
    assert "create_hr_ticket" not in result["tools_used"]


# --------------------------------------------------------------------------- #
# Identical repeated tool calls are de-duplicated
# --------------------------------------------------------------------------- #
def test_duplicate_tool_calls_skipped(monkeypatch):
    calls = {"n": 0}

    def counting(**kw):
        calls["n"] += 1
        return {"answer": "x", "sources": ["annual_leave_policy.txt"], "evidence": []}

    scripted = [
        _tool_call("search_company_policy_tool", {"query": "leave"}),
        _tool_call("search_company_policy_tool", {"query": "leave"}),  # identical
        AIMessage(content="20 days."),
    ]
    _use_fake(monkeypatch, scripted, {"search_company_policy_tool": counting})
    onboarding_agent.run_agent("leave days?")
    assert calls["n"] == 1  # executed once, second identical call skipped
