"""Day 27 - evaluation runner for the Employee Onboarding AI Agent.

Runs every case in ``dataset.json`` through the real agent and records what
actually happened, using deterministic checks (no LLM-as-judge). Results are
written to ``evaluation/results.json``.

Requires a valid API key AND an ingested Chroma collection, because it exercises
the full live pipeline. Run:

    python -m evaluation.run_eval
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.agent.onboarding_agent import run_agent
from app.config import NOT_FOUND_MESSAGE, configure_logging

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

logger = logging.getLogger("eval")


def _is_refusal(answer: str) -> bool:
    return NOT_FOUND_MESSAGE.lower() in (answer or "").lower()


def evaluate_case(case: dict) -> dict:
    """Run one case and return a result record with pass/fail + failure category."""
    result = run_agent(case["question"], employee_id=case.get("employee_id"))
    answer = result["answer"]
    sources = result["sources"]
    tools_used = result["tools_used"]
    ticket_id = result["ticket_id"]

    expected_action = case["expected_action"]
    expected_sources = case.get("expected_sources", [])
    expected_tools = case.get("expected_tools", [])

    passed = True
    failure = None
    notes = ""

    if expected_action == "refuse":
        if not _is_refusal(answer):
            passed, failure = False, "refusal_failure"
            notes = "Expected a refusal but the agent answered."
    elif expected_action == "create_ticket":
        if ticket_id is None:
            passed, failure = False, "tool_execution_failure"
            notes = "Expected a ticket to be created."
    elif expected_action == "profile_lookup":
        if "get_employee_profile" not in tools_used:
            passed, failure = False, "tool_selection_failure"
            notes = "Expected a profile lookup."
    else:  # answer / profile_and_policy
        if _is_refusal(answer):
            passed, failure = False, "grounding_failure"
            notes = "Agent refused a question that should be answerable."
        elif expected_sources and not set(expected_sources) & set(sources):
            passed, failure = False, "citation_failure"
            notes = f"Expected one of {expected_sources}, got {sources}."
        elif set(expected_tools) - set(tools_used):
            passed, failure = False, "tool_selection_failure"
            notes = f"Missing expected tools: {set(expected_tools) - set(tools_used)}."

    # A ticket must never be created unless the case expects it.
    if expected_action != "create_ticket" and ticket_id is not None:
        passed, failure = False, "unexpected_ticket_creation"
        notes = f"Unexpected ticket #{ticket_id} created."

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_action": expected_action,
        "actual_answer": answer,
        "actual_sources": sources,
        "actual_tools": tools_used,
        "ticket_id": ticket_id,
        "passed": passed,
        "failure_category": failure,
        "notes": notes,
    }


def main() -> None:
    configure_logging()
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    logger.info("Evaluation started (%d cases).", len(cases))

    results = []
    for case in cases:
        try:
            results.append(evaluate_case(case))
        except Exception as exc:  # noqa: BLE001 - record API failures, keep going
            logger.exception("Case %s crashed", case["id"])
            results.append({
                "id": case["id"], "category": case["category"],
                "passed": False, "failure_category": "api_failure",
                "notes": str(exc),
            })

    passed = sum(1 for r in results if r.get("passed"))
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Evaluation completed: %d/%d passed.", passed, len(results))
    print(f"Passed {passed}/{len(results)}. Results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
