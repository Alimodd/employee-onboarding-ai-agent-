"""Tests for LangChain document loading/splitting and profile filters (no API)."""

from __future__ import annotations

from app.rag.langchain_documents import METADATA_KEYS, load_and_split
from app.rag.langchain_retriever import build_profile_filter


def test_documents_split_into_chunks():
    chunks = load_and_split()
    assert len(chunks) > 7  # several chunks per document


def test_metadata_survives_splitting():
    chunks = load_and_split()
    for chunk in chunks[:20]:
        for key in ("source", "title", "section", "country", "department"):
            assert key in chunk.metadata
        # Every guaranteed key is present (consistent structure).
        assert set(METADATA_KEYS).issubset(chunk.metadata.keys())


def test_source_matches_filename():
    chunks = load_and_split()
    assert any(c.metadata["source"].endswith(".txt") for c in chunks)


# --------------------------------------------------------------------------- #
# Profile-aware filter (Day 24)
# --------------------------------------------------------------------------- #
def test_filter_country_uses_or_all():
    where = build_profile_filter(country="Belgium")
    assert where == {"$or": [{"country": "Belgium"}, {"country": "All"}]}


def test_filter_country_and_department_combined():
    where = build_profile_filter(country="Belgium", department="Engineering")
    assert "$and" in where
    assert len(where["$and"]) == 2


def test_no_filter_returns_none():
    assert build_profile_filter() is None
