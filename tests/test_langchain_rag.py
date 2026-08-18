"""Tests for the LangChain RAG pipeline (retriever + LLM mocked)."""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument

from app.config import NOT_FOUND_MESSAGE
from app.rag import langchain_rag


def _docs():
    return [
        LCDocument(page_content="Employees may work remotely up to two days per week.",
                   metadata={"source": "remote_work_policy.txt", "section": "Remote Days"}),
        LCDocument(page_content="Standard working hours are 09:00 to 17:00.",
                   metadata={"source": "remote_work_policy.txt", "section": "Hours"}),
    ]


def test_format_context_labels_sources():
    ctx = langchain_rag.format_context(_docs())
    assert "[Source: remote_work_policy.txt" in ctx
    assert "two days per week" in ctx


def test_extract_sources_deduplicates():
    assert langchain_rag.extract_sources(_docs()) == ["remote_work_policy.txt"]


def test_rag_answer_returns_real_sources(monkeypatch):
    monkeypatch.setattr(langchain_rag, "retrieve_documents", lambda *a, **k: _docs())
    monkeypatch.setattr(langchain_rag, "invoke_model", lambda prompt: "You may work remotely two days per week.")
    result = langchain_rag.rag_answer("How many remote days?")
    assert result["sources"] == ["remote_work_policy.txt"]
    assert "remotely" in result["answer"]
    assert len(result["retrieved_documents"]) == 2


def test_rag_answer_refuses_when_no_docs(monkeypatch):
    monkeypatch.setattr(langchain_rag, "retrieve_documents", lambda *a, **k: [])
    # invoke_model must NOT be called when there is nothing retrieved.
    monkeypatch.setattr(langchain_rag, "invoke_model", lambda p: (_ for _ in ()).throw(AssertionError("should not call LLM")))
    result = langchain_rag.rag_answer("unknown thing")
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["sources"] == []


def test_rag_answer_refusal_has_no_sources(monkeypatch):
    monkeypatch.setattr(langchain_rag, "retrieve_documents", lambda *a, **k: _docs())
    monkeypatch.setattr(langchain_rag, "invoke_model", lambda p: NOT_FOUND_MESSAGE)
    result = langchain_rag.rag_answer("something not covered")
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["sources"] == []  # no sources attached to a refusal
