"""Day 17 (+ Day 24) - LangChain retriever over the existing Chroma collection.

Chroma stays the real vector database. LangChain is only the orchestration layer
on top of it: we point ``langchain_chroma.Chroma`` at the SAME persistent client
and collection that the root ingestion already fills, using the SAME embedding
model, so no second database is created.

    query -> embedding (Gemini) -> Chroma similarity search -> LangChain Documents

Profile-aware retrieval (Day 24): a country/department filter matches that value
OR the generic value ``"All"``, so company-wide policies are always eligible
while another country's/department's exclusive policy is not wrongly returned.

NOTE: metadata filtering here is relevance personalization, NOT authorization.
"""

from __future__ import annotations

import logging

import chromadb
from langchain_core.documents import Document as LCDocument

import model_client
from app.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    LC_EMBED_MODEL,
)

logger = logging.getLogger("langchain_retriever")


def _get_embeddings():
    """LangChain embeddings bound to our key, using the ingestion embed model."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = model_client._get_api_key()
    return GoogleGenerativeAIEmbeddings(model=LC_EMBED_MODEL, google_api_key=api_key)


def get_vectorstore():
    """Open the existing persistent Chroma collection as a LangChain vector store."""
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
    )


def build_profile_filter(
    country: str | None = None,
    department: str | None = None,
    document_type: str | None = None,
) -> dict | None:
    """Build a Chroma ``where`` filter with OR-``All`` semantics for country/dept.

    ``country="Belgium"`` matches documents whose country is ``"Belgium"`` OR
    ``"All"``. Returns ``None`` when no filter is requested.
    """
    clauses: list[dict] = []
    if country:
        clauses.append({"$or": [{"country": country}, {"country": "All"}]})
    if department:
        clauses.append({"$or": [{"department": department}, {"department": "All"}]})
    if document_type:
        clauses.append({"document_type": document_type})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def get_retriever(top_k: int = DEFAULT_TOP_K, where: dict | None = None):
    """Return a LangChain retriever over the existing Chroma collection."""
    search_kwargs: dict = {"k": top_k}
    if where:
        search_kwargs["filter"] = where
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def retrieve_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    country: str | None = None,
    department: str | None = None,
    document_type: str | None = None,
) -> list[LCDocument]:
    """Return the ``top_k`` most relevant LangChain Documents for ``query``.

    Input: a non-empty query, optional profile filters.
    Output: list of LangChain Documents (metadata preserved); ``[]`` if the
        collection is empty or nothing matches.
    Calls: Gemini embeddings + Chroma similarity search.
    Stores: nothing (read-only).
    Fails: blank query / non-positive top_k -> ValueError; embedding/key errors
        propagate from ``model_client``.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    where = build_profile_filter(country, department, document_type)
    retriever = get_retriever(top_k=top_k, where=where)
    logger.info("Retrieval executed (top_k=%d, filtered=%s)", top_k, bool(where))
    docs = retriever.invoke(query)
    logger.info("Retrieval returned %d documents", len(docs))
    return docs
