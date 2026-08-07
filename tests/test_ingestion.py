"""Tests for the Day 10 + Day 11 ingestion pipeline.

Covers loading, chunking, metadata, edge cases, and Chroma storage. The
embedding API is mocked (deterministic, no quota, no key needed) and Chroma is
written to a temporary folder so the test never touches the real ``chroma_db/``.
"""

from __future__ import annotations

import types

import pytest

import model_client
import vector_store
from chunker import (
    DEFAULT_CHUNK_SIZE,
    Chunk,
    chunk_document,
    chunk_documents,
    chunk_text,
    split_into_sections,
)
from document_loader import Document, load_documents


# --------------------------------------------------------------------------- #
# Day 10: loading
# --------------------------------------------------------------------------- #
def test_documents_can_be_loaded():
    docs = load_documents()
    assert len(docs) == 7
    assert all(isinstance(d, Document) for d in docs)


def test_loaded_documents_have_text_and_metadata():
    docs = load_documents()
    for doc in docs:
        assert doc.text.strip()
        # Missing metadata is filled with "unknown", never left absent/None.
        for key in ("document_id", "title", "source_filename"):
            assert doc.metadata.get(key)


# --------------------------------------------------------------------------- #
# Day 10: chunking + metadata
# --------------------------------------------------------------------------- #
def test_documents_produce_chunks():
    chunks = chunk_documents(load_documents())
    assert len(chunks) > 7  # at least one chunk per document, usually several


def test_chunks_contain_metadata():
    chunks = chunk_documents(load_documents())
    for chunk in chunks:
        for key in ("document_id", "title", "source", "section", "chunk_id"):
            assert chunk.metadata.get(key)
        assert chunk.metadata["chunk_id"] == chunk.id


def test_chunk_ids_are_unique():
    chunks = chunk_documents(load_documents())
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_document_is_handled_safely():
    empty = Document(text="   ", metadata={"document_id": "X"})
    assert empty.is_empty
    assert chunk_document(empty) == []


def test_very_short_document_becomes_one_chunk():
    short = Document(text="Short policy.", metadata={"document_id": "S"})
    chunks = chunk_document(short)
    assert len(chunks) == 1
    assert chunks[0].text == "Short policy."


def test_chunk_text_respects_size_and_overlap():
    text = "abcdefghij" * 10  # 100 chars
    pieces = chunk_text(text, chunk_size=40, overlap=10)
    assert all(len(p) <= 40 for p in pieces)
    assert len(pieces) > 1


def test_chunk_text_empty_returns_empty():
    assert chunk_text("   ") == []


def test_split_into_sections_detects_headings():
    text = "Overview\n--------\nBody text here.\n\nRules\n-----\nMore text."
    sections = split_into_sections(text)
    names = [name for name, _ in sections]
    assert "Overview" in names and "Rules" in names


# --------------------------------------------------------------------------- #
# Day 11: Chroma storage (embedding API mocked, temp DB)
# --------------------------------------------------------------------------- #
def _fake_embed_client(dim=8):
    """Fake google-genai client returning constant-length vectors."""

    def embed_content(model, contents):
        items = [
            types.SimpleNamespace(values=[float(len(t) % 7)] * dim) for t in contents
        ]
        return types.SimpleNamespace(embeddings=items)

    models = types.SimpleNamespace(embed_content=embed_content)
    return types.SimpleNamespace(models=models)


@pytest.fixture
def mocked_embeddings(monkeypatch):
    monkeypatch.setenv(model_client.API_KEY_ENV, "dummy-key")
    monkeypatch.setattr(
        model_client, "_build_client", lambda key: _fake_embed_client()
    )


def _sample_chunks():
    return [
        Chunk(id="D::chunk-0", text="First chunk.", metadata={"title": "T", "section": "S"}),
        Chunk(id="D::chunk-1", text="Second chunk.", metadata={"title": "T", "section": "S"}),
    ]


def test_chroma_ingestion_succeeds(monkeypatch, tmp_path, mocked_embeddings):
    monkeypatch.setattr(vector_store, "CHROMA_PATH", tmp_path / "chroma_db")
    collection = vector_store.get_collection(tmp_path / "chroma_db")
    stored = vector_store.store_chunks(_sample_chunks(), collection)
    assert stored == 2
    assert collection.count() == 2


def test_stored_chunk_count_is_sensible(monkeypatch, tmp_path, mocked_embeddings):
    collection = vector_store.get_collection(tmp_path / "chroma_db")
    chunks = chunk_documents(load_documents())
    stored = vector_store.store_chunks(chunks, collection)
    assert stored == len(chunks)
    assert collection.count() == len(chunks)


def test_repeated_ingestion_does_not_duplicate(monkeypatch, tmp_path, mocked_embeddings):
    path = tmp_path / "chroma_db"
    collection = vector_store.get_collection(path)
    chunks = _sample_chunks()

    vector_store.store_chunks(chunks, collection)
    first_count = collection.count()

    # Re-run the exact same ingestion.
    vector_store.store_chunks(chunks, collection)
    second_count = collection.count()

    # Deterministic IDs + upsert => count stays the same, no duplicates.
    assert first_count == 2
    assert second_count == 2
