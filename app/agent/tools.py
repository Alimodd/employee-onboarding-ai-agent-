"""Day 19 - the three core agent tools.

Each tool is a plain, deterministic Python function (easy to test without any
LLM) plus a thin LangChain ``@tool`` wrapper used only to describe the tool to
the model. The agent orchestrator dispatches to the plain functions so it always
receives structured dicts (real sources, real ticket ids), never model text.

Tools:
    get_employee_profile(employee_id)
    search_company_policy(query, department=None, country=None)
    create_hr_ticket(employee_id, topic, description)
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from app import employees
from app.config import DEFAULT_TOP_K
from app.database import tickets
from app.rag.langchain_rag import rag_answer

logger = logging.getLogger("tools")


# --------------------------------------------------------------------------- #
# Plain functions (the real implementations)
# --------------------------------------------------------------------------- #
def get_employee_profile(employee_id: int) -> dict:
    """Look up a fictional employee profile by id.

    Returns ``{"found": True, "profile": {...}}`` or
    ``{"found": False, "error": "..."}``. Never raises for a normal miss.
    """
    try:
        profile = employees.get_employee(employee_id)
    except employees.EmployeeNotFoundError:
        return {"found": False, "error": f"Employee {employee_id} not found."}
    except ValueError:
        return {"found": False, "error": f"Invalid employee id: {employee_id!r}."}
    return {"found": True, "profile": profile}


def search_company_policy(
    query: str,
    department: str | None = None,
    country: str | None = None,
) -> dict:
    """Search company policy documents and return a grounded answer + sources.

    Uses the LangChain/Chroma RAG layer. Returns
    ``{"answer": str, "sources": [...], "evidence": [...]}``. If nothing relevant
    is found, ``answer`` is the standard not-found message and ``sources`` is [].
    """
    if not query or not query.strip():
        return {"answer": "", "sources": [], "evidence": [], "error": "empty query"}

    result = rag_answer(query, top_k=DEFAULT_TOP_K, country=country, department=department)
    # Keep only compact evidence for the model; full chunks stay internal.
    evidence = [
        {
            "source": d["metadata"].get("source"),
            "section": d["metadata"].get("section"),
            "text": d["text"][:300],
        }
        for d in result.get("retrieved_documents", [])
    ]
    return {"answer": result["answer"], "sources": result["sources"], "evidence": evidence}


def create_hr_ticket(employee_id: int, topic: str, description: str) -> dict:
    """Create an HR ticket in SQLite after validating the employee exists.

    Returns the created (or duplicate) ticket dict, or ``{"error": "..."}``.
    The ticket id always comes from SQLite.
    """
    profile = get_employee_profile(employee_id)
    if not profile["found"]:
        return {"error": profile["error"]}
    try:
        return tickets.create_ticket(int(employee_id), topic, description)
    except tickets.TicketError as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# LangChain tool wrappers (schema/description for the model to choose from)
# --------------------------------------------------------------------------- #
@tool
def get_employee_profile_tool(employee_id: int) -> dict:
    """Look up an employee's profile (name, department, role, country, start date)
    by their numeric employee_id. Use this when the question depends on who the
    employee is or where they work."""
    return get_employee_profile(employee_id)


@tool
def search_company_policy_tool(
    query: str, department: str | None = None, country: str | None = None
) -> dict:
    """Search official company policy documents for an answer. Always use this for
    any company-policy question (leave, remote work, sick leave, benefits, IT
    security, code of conduct, onboarding). Optionally pass the employee's
    department and country to prefer applicable policies."""
    return search_company_policy(query, department=department, country=country)


@tool
def create_hr_ticket_tool(employee_id: int, topic: str, description: str) -> dict:
    """Create an HR support ticket for an employee. ONLY call this when the user
    explicitly asks to open/create a ticket or contact HR. Never create a ticket
    just because information was missing."""
    return create_hr_ticket(employee_id, topic, description)


# Schemas the model sees, and the dispatch table the orchestrator executes.
LC_TOOLS = [get_employee_profile_tool, search_company_policy_tool, create_hr_ticket_tool]

TOOL_DISPATCH = {
    "get_employee_profile_tool": get_employee_profile,
    "search_company_policy_tool": search_company_policy,
    "create_hr_ticket_tool": create_hr_ticket,
}
