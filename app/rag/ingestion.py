"""Ingestion entry point used by the API (Day 23 POST /documents/ingest).

Reuses the existing, tested root pipeline (loader -> chunker -> Chroma upsert)
instead of duplicating it. Re-running is safe: chunk ids are deterministic and
stored with ``upsert``, so chunks are overwritten, not duplicated.
"""

from __future__ import annotations

import logging

from app.config import COLLECTION_NAME
from chunker import chunk_documents
from document_loader import load_documents
from vector_store import get_collection, store_chunks

logger = logging.getLogger("ingestion")


def ingest_documents() -> dict:
    """Load, chunk, embed, and upsert all company documents into Chroma.

    Output: ``{"documents": int, "chunks": int, "collection": str, "stored": int}``.
    Calls: root ``load_documents`` / ``chunk_documents`` / ``store_chunks``.
    Fails: embedding/key errors propagate from ``model_client``.
    """
    logger.info("Ingestion started.")
    docs = load_documents()
    non_empty = [d for d in docs if not d.is_empty]
    chunks = chunk_documents(non_empty)

    stored = 0
    if chunks:
        collection = get_collection()
        stored = store_chunks(chunks, collection)

    logger.info(
        "Ingestion completed (documents=%d, chunks=%d, stored=%d).",
        len(non_empty), len(chunks), stored,
    )
    return {
        "documents": len(non_empty),
        "chunks": len(chunks),
        "collection": COLLECTION_NAME,
        "stored": stored,
    }
