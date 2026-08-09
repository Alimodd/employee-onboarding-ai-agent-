"""Tests for Day 12 semantic retrieval (no LLM involved).

Embeddings and Chroma are mocked so the tests are deterministic, need no API
key, and never rebuild the real vector database. A tiny in-memory fake stands in
for the Chroma collection and records how it was queried.
"""

from __future__ import annotations

import types

import pytest

import retriever


# --------------------------------------------------------------------------- #
# A minimal fake Chroma collection
# --------------------------------------------------------------------------- #
class FakeCollection:
    """Records query calls and returns canned, deterministic results."""

    def __init__(self, rows, count=None):
        # rows: list of (id, text, metadata)
        self._rows = rows
        self._count = len(rows) if count is None else count
        self.last_where = "unset"
        self.last_n_results = None

    def count(self):
        return self._count

    def query(self, query_embeddings, n_results, where, include):
        self.last_where = where
        self.last_n_results = n_results
        # Apply the where filter ourselves so filter tests are meaningful.
        rows = self._rows
        if where:
            clauses = where["$and"] if "$and" in where else [where]
            for clause in clauses:
                (field, value), = clause.items()
                rows = [r for r in rows if r[2].get(field) == value]
        rows = rows[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[0.1 * (i + 1) for i in range(len(rows))]],
        }


LEAVE = ("NS-HR-001::chunk-1", "Employees receive 20 working days of annual leave.",
         {"source": "annual_leave_policy.txt", "title": "Annual Leave Policy",
          "country": "Belgium", "department": "All", "document_type": "HR Policy"})
REMOTE = ("NS-HR-002::chunk-1", "Employees may work remotely two days per week.",
          {"source": "remote_work_policy.txt", "title": "Remote Work Policy",
           "country": "Belgium", "department": "All", "document_type": "HR Policy"})
SECURITY = ("NS-IT-001::chunk-1", "Use strong passwords and lock your screen.",
            {"source": "it_security_policy.txt", "title": "IT Security Policy",
             "country": "All", "department": "All", "document_type": "IT Policy"})

ALL_ROWS = [REMOTE, LEAVE, SECURITY]


@pytest.fixture
def fake_db(monkeypatch):
    """Patch retrieval to use the fake collection and a dummy embedding."""
    collection = FakeCollection(ALL_ROWS)
    monkeypatch.setattr(retriever, "get_collection", lambda: collection)
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.1, 0.2, 0.3]])
    return collection


# --------------------------------------------------------------------------- #
# 1. direct question returns chunks with text + metadata
# --------------------------------------------------------------------------- #
def test_direct_question_returns_relevant_chunks(fake_db):
    results = retriever.search_documents("How many remote work days?", top_k=3)
    assert results, "expected at least one chunk"
    first = results[0]
    assert "text" in first and first["text"]
    assert "metadata" in first and first["metadata"].get("source")
    assert "id" in first
    assert "distance" in first  # similarity/distance is exposed


# --------------------------------------------------------------------------- #
# 2. paraphrased question still runs retrieval (embedding is what maps it)
# --------------------------------------------------------------------------- #
def test_paraphrased_question_still_retrieves(fake_db):
    results = retriever.search_documents("time off to go on holiday", top_k=3)
    assert len(results) > 0


# --------------------------------------------------------------------------- #
# 3. top_k controls the number of results
# --------------------------------------------------------------------------- #
def test_top_k_controls_result_count(fake_db):
    assert len(retriever.search_documents("policy", top_k=1)) == 1
    assert len(retriever.search_documents("policy", top_k=3)) == 3
    # Fake collection recorded the requested n_results.
    assert fake_db.last_n_results == 3


# --------------------------------------------------------------------------- #
# 4. empty query is rejected safely
# --------------------------------------------------------------------------- #
def test_empty_query_rejected():
    with pytest.raises(ValueError):
        retriever.search_documents("   ")


def test_non_positive_top_k_rejected(fake_db):
    with pytest.raises(ValueError):
        retriever.search_documents("anything", top_k=0)


# --------------------------------------------------------------------------- #
# 5. unrelated query does not crash (still returns a list)
# --------------------------------------------------------------------------- #
def test_unrelated_query_does_not_crash(fake_db):
    results = retriever.search_documents("best pizza topping in the world", top_k=3)
    assert isinstance(results, list)


# --------------------------------------------------------------------------- #
# 6. metadata filtering works
# --------------------------------------------------------------------------- #
def test_single_metadata_filter(fake_db):
    results = retriever.search_documents("passwords", document_type="IT Policy", top_k=3)
    assert all(r["metadata"]["document_type"] == "IT Policy" for r in results)
    assert fake_db.last_where == {"document_type": "IT Policy"}


def test_multiple_metadata_filters_use_and(fake_db):
    retriever.search_documents("leave", country="Belgium", department="All", top_k=3)
    assert "$and" in fake_db.last_where


def test_no_filter_passes_none_where(fake_db):
    retriever.search_documents("anything", top_k=2)
    assert fake_db.last_where is None


# --------------------------------------------------------------------------- #
# 7. retrieval uses the existing DB, and empty DB returns [] (no rebuild)
# --------------------------------------------------------------------------- #
def test_empty_collection_returns_empty(monkeypatch):
    empty = FakeCollection([], count=0)
    monkeypatch.setattr(retriever, "get_collection", lambda: empty)
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.1]])
    assert retriever.search_documents("anything") == []
