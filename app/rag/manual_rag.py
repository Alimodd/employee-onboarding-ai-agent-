"""Day 13/14 - manual RAG (no LangChain), kept for comparison with the LangChain
version in ``langchain_rag.py``.

    question
    -> retriever.search_documents (Chroma, raw)
    -> format context
    -> grounded prompt
    -> model_client.generate_response (raw Gemini SDK)
    -> answer + real sources from retrieved metadata

Same behaviour and same citation guarantee as the LangChain pipeline, but built
from the raw building blocks so a learner can compare the two side by side.
"""

from __future__ import annotations

import logging

import model_client
from app.config import DEFAULT_TOP_K, NOT_FOUND_MESSAGE
from retriever import search_documents

logger = logging.getLogger("manual_rag")

_PROMPT_TEMPLATE = """\
You are an Employee Onboarding assistant. Answer using ONLY the company policy
context below. Do not use outside knowledge and do not invent policies. If the
context is insufficient, reply with exactly: "{not_found}". Keep it concise.

COMPANY CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def format_context(results: list[dict]) -> str:
    """Join raw retrieval hits into a readable, source-labelled context string."""
    blocks = []
    for hit in results:
        meta = hit.get("metadata", {})
        source = meta.get("source", meta.get("source_filename", "unknown"))
        section = meta.get("section", "")
        header = f"[Source: {source}" + (f" | Section: {section}]" if section else "]")
        blocks.append(f"{header}\n{hit['text'].strip()}")
    return "\n\n".join(blocks)


def extract_sources(results: list[dict]) -> list[str]:
    """Return de-duplicated source filenames in retrieval order."""
    seen: list[str] = []
    for hit in results:
        meta = hit.get("metadata", {})
        source = meta.get("source", meta.get("source_filename"))
        if source and source not in seen:
            seen.append(source)
    return seen


def rag_answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Manual-RAG answer with real citations.

    Output: ``{"answer": str, "sources": list[str], "retrieved_documents": list}``.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty, non-whitespace string.")

    results = search_documents(question, top_k=top_k)
    if not results:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "retrieved_documents": []}

    prompt = _PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_MESSAGE,
        context=format_context(results),
        question=question,
    )
    answer = model_client.generate_response(prompt)
    sources = [] if answer.strip() == NOT_FOUND_MESSAGE else extract_sources(results)

    return {"answer": answer, "sources": sources, "retrieved_documents": results}
