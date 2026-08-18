"""Tests for the fictional employee directory."""

from __future__ import annotations

import pytest

from app import employees


def test_existing_employee_returned():
    emp = employees.get_employee(101)
    assert emp["name"] == "Anke De Vries"
    assert emp["country"] == "Belgium"
    assert emp["department"] == "Engineering"


def test_missing_employee_raises():
    with pytest.raises(employees.EmployeeNotFoundError):
        employees.get_employee(9999)


def test_malformed_employee_id_raises():
    with pytest.raises(ValueError):
        employees.get_employee("not-a-number")


def test_list_employees_nonempty():
    all_emps = employees.list_employees()
    assert len(all_emps) >= 5
    assert {"Belgium", "Netherlands", "Germany"} <= {e["country"] for e in all_emps}
