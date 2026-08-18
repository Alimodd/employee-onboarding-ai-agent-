"""Tests for SQLite HR ticket persistence and duplicate protection."""

from __future__ import annotations

import pytest

from app.database import tickets


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "tickets.db"


def test_insert_and_retrieve(tmp_db):
    tickets.init_db(tmp_db)
    created = tickets.create_ticket(101, "Parental leave", "Need info", tmp_db)
    assert created["ticket_id"] == 1
    assert created["status"] == "open"
    assert created["duplicate"] is False

    all_tickets = tickets.get_all_tickets(tmp_db)
    assert len(all_tickets) == 1
    assert all_tickets[0]["employee_id"] == 101


def test_ticket_id_comes_from_sqlite(tmp_db):
    t1 = tickets.create_ticket(101, "Topic A", "desc", tmp_db)
    t2 = tickets.create_ticket(102, "Topic B", "desc", tmp_db)
    assert (t1["ticket_id"], t2["ticket_id"]) == (1, 2)


def test_duplicate_open_ticket_is_suppressed(tmp_db):
    tickets.create_ticket(101, "Parental leave", "Need info", tmp_db)
    dup = tickets.create_ticket(101, "  parental LEAVE ", "again", tmp_db)
    assert dup["duplicate"] is True
    assert dup["ticket_id"] == 1
    assert len(tickets.get_all_tickets(tmp_db)) == 1


def test_different_topic_not_duplicate(tmp_db):
    tickets.create_ticket(101, "Parental leave", "x", tmp_db)
    other = tickets.create_ticket(101, "Payroll question", "y", tmp_db)
    assert other["duplicate"] is False
    assert len(tickets.get_all_tickets(tmp_db)) == 2


def test_empty_topic_rejected(tmp_db):
    with pytest.raises(tickets.TicketError):
        tickets.create_ticket(101, "   ", "desc", tmp_db)


def test_get_all_on_missing_db_returns_empty(tmp_path):
    assert tickets.get_all_tickets(tmp_path / "nope.db") == []
