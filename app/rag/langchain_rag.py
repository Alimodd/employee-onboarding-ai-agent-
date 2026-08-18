"""Day 18 (+ Day 14 citations) - the LangChain RAG pipeline.

Flow:
    question
    -> retrieve LangChain Documents (Chroma)
    -> format context (clearly separated, labelled by real source)
    -> grounded prompt (answer ONLY from context, else refuse)
    -> LLM
    -> answer
    -> sources extracted from the RETRIEVED metadata (never invented by the LLM)
    -> structured result

Sources always come from the metadata of the documents we actually retrieved, so
the LLM cannot fabricate citations.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document as LCDocument

from app.config import DEFAULT_TOP_K, NOT_FOUND_MESSAGE
from app.rag.langchain_retriever import retrieve_documents
from app.services.langchain_client import invoke_model

logger = logging.getLogger("langchain_rag")

_PROMPT_TEMPLATE = """\
You are an Employee Onboarding assistant. Answer the employee's question using \
ONLY the company policy context provided below.

Rules:
- Use ONLY the provided context. Do not use outside or general knowledge.
- Do not invent company policies, numbers, or rules that are not in the context.
- If the context does not contain enough information to answer, reply with \
exactly: "{not_found}"
- If several context blocks describe different versions of the same policy, \
prefer the one with the most recent effective date.
- Keep the answer concise and factual.

COMPANY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:"""


def format_context(documents: list[LCDocument]) -> str:
    """Join retrieved documents into one readable, clearly separated context.

    Each block is labelled with its real source filename, e.g.::

        [Source: remote_work_policy.txt | Section: Remote Days]
        Employees may work remotely up to two days per week...
    """
    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", doc.metadata.get("source_filename", "unknown"))
        section = doc.metadata.get("section", "")
        header = f"[Source: {source}" + (f" | Section: {section}]" if section else "]")
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def extract_sources(documents: list[LCDocument]) -> list[str]:
    """Return de-duplicated source filenames, preserving retrieval order."""
    seen: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", doc.metadata.get("source_filename"))
        if source and source not in seen:
            seen.append(source)
    return seen


def build_rag_prompt(question: str, context: str) -> str:
    """Combine instructions + retrieved context + question into one prompt."""
    return _PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_MESSAGE, context=context, question=question
    )


def rag_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    country: str | None = None,
    department: str | None = None,
) -> dict:
    """Answer a question with grounded RAG and real citations.

    Input: a non-empty question, optional profile filters.
    Output: ``{"answer": str, "sources": list[str], "retrieved_documents": list}``.
        ``retrieved_documents`` is internal/debug detail (text + metadata).
    Calls: ``retrieve_documents`` then ``invoke_model``.
    Stores: nothing.
    Fails: blank question -> ValueError; provider/key errors propagate.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty, non-whitespace string.")

    documents = retrieve_documents(
        question, top_k=top_k, country=country, department=department
    )

    # No evidence at all -> refuse instead of inventing an answer.
    if not documents:
        logger.info("RAG: no documents retrieved; refusing.")
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "retrieved_documents": []}

    context = format_context(documents)
    prompt = build_rag_prompt(question, context)
    answer = invoke_model(prompt)

    # If the model refused, do not attach sources (nothing was actually used).
    sources = [] if answer.strip() == NOT_FOUND_MESSAGE else extract_sources(documents)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_documents": [
            {"text": d.page_content, "metadata": d.metadata} for d in documents
        ],
    }
