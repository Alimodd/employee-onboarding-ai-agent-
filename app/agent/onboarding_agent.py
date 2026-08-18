"""Days 20-21 - the single Employee Onboarding tool-calling agent.

An explicit, easy-to-trace loop (no LangGraph): the model is given the three
tools via ``bind_tools``; when it requests a tool call we dispatch to the plain
Python function, feed the result back, and repeat until the model produces a
final answer or we hit the iteration cap.

Returns a structured result:
    {"answer": str, "sources": [...], "tools_used": [...], "ticket_id": int|None}

Sources come from the policy-search tool's real retrieved metadata; ticket_id
comes from SQLite - neither is invented by the model.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.tools import LC_TOOLS, TOOL_DISPATCH, get_employee_profile
from app.config import MAX_AGENT_ITERATIONS, NOT_FOUND_MESSAGE
from app.services.langchain_client import _normalize_text, get_chat_model

logger = logging.getLogger("agent")

# Friendly names reported in tools_used (drop the "_tool" suffix).
_FRIENDLY = {
    "get_employee_profile_tool": "get_employee_profile",
    "search_company_policy_tool": "search_company_policy",
    "create_hr_ticket_tool": "create_hr_ticket",
}

SYSTEM_PROMPT = f"""\
You are the Employee Onboarding assistant for a company. Follow these rules:

- For any company-policy claim (leave, remote work, sick leave, benefits, IT
  security, code of conduct, onboarding), you MUST use the search_company_policy
  tool. Never state a company policy from your own knowledge.
- When a question depends on a specific employee, use their profile (country and
  department) to search for the policies that apply to them.
- Never fabricate a company policy. If the policy tool cannot find the answer,
  tell the user: "{NOT_FOUND_MESSAGE}" You may add that they can ask HR for help.
- Only create an HR ticket when the user explicitly asks to open a ticket or
  contact HR. Do NOT create a ticket merely because information is missing.
- Do not call the same tool twice with identical arguments.
- Keep answers concise and base every policy statement on retrieved sources.
"""


def _tool_signature(name: str, args: dict) -> tuple:
    """A hashable signature to detect identical repeated tool calls."""
    return (name, tuple(sorted((k, str(v)) for k, v in (args or {}).items())))


def run_agent(message: str, employee_id: int | None = None) -> dict:
    """Run the onboarding agent for one user message.

    Input: the user message and an optional employee_id for profile-aware answers.
    Output: {"answer", "sources", "tools_used", "ticket_id"}.
    Calls: the LangChain chat model (tool-calling) + the three tools.
    Fails: missing API key -> model_client.ConfigurationError (propagated).
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty, non-whitespace string.")

    llm = get_chat_model().bind_tools(LC_TOOLS)

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    tools_used: list[str] = []
    sources: list[str] = []
    ticket_id: int | None = None

    # Profile-aware context: if we know the employee, resolve them once up front
    # and hand the model their country/department to steer policy search.
    emp_country = emp_department = None
    if employee_id is not None:
        profile_result = get_employee_profile(employee_id)
        tools_used.append("get_employee_profile")
        if profile_result["found"]:
            p = profile_result["profile"]
            emp_country, emp_department = p.get("country"), p.get("department")
            messages.append(
                SystemMessage(
                    content=(
                        f"Employee context: id={p['employee_id']}, "
                        f"department={emp_department}, country={emp_country}, "
                        f"role={p.get('role')}. Use department and country when "
                        "searching policies for this employee."
                    )
                )
            )
        else:
            messages.append(
                SystemMessage(content=f"Note: {profile_result['error']}")
            )

    messages.append(HumanMessage(content=message))

    seen_calls: dict[tuple, str] = {}

    for iteration in range(MAX_AGENT_ITERATIONS):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            answer = _normalize_text(ai.content).strip() or NOT_FOUND_MESSAGE
            return _result(answer, sources, tools_used, ticket_id)

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            friendly = _FRIENDLY.get(name, name)

            # Default department/country for the policy tool from employee context.
            if name == "search_company_policy_tool":
                args.setdefault("country", emp_country)
                args.setdefault("department", emp_department)

            signature = _tool_signature(name, args)
            if signature in seen_calls:
                messages.append(
                    ToolMessage(
                        content="Duplicate call skipped: " + seen_calls[signature],
                        tool_call_id=call["id"],
                    )
                )
                continue

            func = TOOL_DISPATCH.get(name)
            if func is None:
                messages.append(
                    ToolMessage(content=f"Unknown tool: {name}", tool_call_id=call["id"])
                )
                continue

            start = time.perf_counter()
            try:
                result = func(**args)
                ok = "error" not in result
            except Exception as exc:  # noqa: BLE001 - surface, never swallow silently
                logger.exception("Tool %s raised an exception", friendly)
                result = {"error": f"Tool '{friendly}' failed."}
                ok = False
            duration_ms = (time.perf_counter() - start) * 1000

            # Day 21 tool-call logging (non-sensitive args only; no secrets).
            logger.info(
                "tool=%s ok=%s duration_ms=%.1f args=%s",
                friendly, ok, duration_ms, {k: args.get(k) for k in args if k != "description"},
            )
            if friendly not in tools_used:
                tools_used.append(friendly)

            # Capture structured, trustworthy fields from tool results.
            if name == "search_company_policy_tool":
                for src in result.get("sources", []):
                    if src not in sources:
                        sources.append(src)
            if name == "create_hr_ticket_tool" and "ticket_id" in result:
                ticket_id = result["ticket_id"]

            seen_calls[signature] = str(result)[:200]
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    # Iteration cap reached without a plain answer: give a safe fallback.
    logger.warning("Agent hit iteration cap (%d).", MAX_AGENT_ITERATIONS)
    return _result(
        "I could not complete the request within the allowed steps. "
        "Please try rephrasing.",
        sources, tools_used, ticket_id,
    )


def _result(answer, sources, tools_used, ticket_id) -> dict:
    return {
        "answer": answer,
        "sources": sources,
        "tools_used": tools_used,
        "ticket_id": ticket_id,
    }
