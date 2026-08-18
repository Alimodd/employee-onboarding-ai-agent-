"""Fictional employee directory backed by a local JSON file.

No database is introduced for employees (the assignment asks to keep this
simple). The file is loaded once and cached in memory.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.config import EMPLOYEES_PATH


class EmployeeNotFoundError(Exception):
    """Raised when an employee id does not exist."""


@lru_cache(maxsize=1)
def _load_all() -> dict[int, dict]:
    """Load employees.json into a dict keyed by employee_id (cached)."""
    raw = json.loads(EMPLOYEES_PATH.read_text(encoding="utf-8"))
    return {int(e["employee_id"]): e for e in raw}


def list_employees() -> list[dict]:
    """Return all fictional employees."""
    return list(_load_all().values())


def get_employee(employee_id: int) -> dict:
    """Return one employee profile.

    Input: an integer employee id.
    Output: the profile dict.
    Fails: non-int id -> ValueError; unknown id -> EmployeeNotFoundError.
    """
    try:
        key = int(employee_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid employee_id: {employee_id!r}") from exc

    employees = _load_all()
    if key not in employees:
        raise EmployeeNotFoundError(f"Employee {key} not found.")
    return employees[key]
