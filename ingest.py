"""Day 11 - runnable ingestion entry point for the policy knowledge base.

Ties the whole Day 10 + Day 11 pipeline together and prints what happened:

    policy files -> loader -> documents -> chunker -> chunks + metadata
                 -> Gemini embeddings -> persistent Chroma collection

Run it manually with:

    python ingest.py

It requires a valid Gemini API key in the environment (``API_key`` in ``.env``)
because it calls the embedding API. Re-running is safe: chunk IDs are
deterministic and stored with ``upsert``, so chunks are overwritten, not
duplicated.
"""

from __future__ import annotations

from chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_documents
from document_loader import load_documents
from vector_store import (
    CHROMA_PATH,
    COLLECTION_NAME,
    get_collection,
    store_chunks,
)


def run_ingestion() -> None:
    """Load, chunk, embed, and store all policy documents; print a summary."""
    # 1. Load the source documents.
    docs = load_documents()
    non_empty = [d for d in docs if not d.is_empty]

    # 2. Split them into chunks with metadata.
    chunks = chunk_documents(non_empty)

    print("Ingestion pipeline")
    print("-" * 50)
    print(f"Documents loaded : {len(docs)} ({len(non_empty)} with text)")
    print(f"Chunks created   : {len(chunks)}")
    print(f"Chunk size/overlap: {DEFAULT_CHUNK_SIZE}/{DEFAULT_OVERLAP}")

    if not chunks:
        print("No chunks to store; nothing was written to Chroma.")
        return

    # 3. Embed the chunks and store them in the persistent Chroma collection.
    collection = get_collection()
    stored = store_chunks(chunks, collection)

    print(f"Collection name  : {COLLECTION_NAME}")
    print(f"Chunks stored    : {stored}")
    print(f"Items in Chroma  : {collection.count()}")
    print(f"Chroma DB path   : {CHROMA_PATH}")

    # 4. Show one stored chunk + its metadata so the result is inspectable.
    sample = collection.get(limit=1, include=["documents", "metadatas"])
    if sample["ids"]:
        print("\nExample stored chunk")
        print("-" * 50)
        print(f"id      : {sample['ids'][0]}")
        meta = sample["metadatas"][0]
        print(f"title   : {meta.get('title')}")
        print(f"section : {meta.get('section')}")
        print(f"source  : {meta.get('source')}")
        text = sample["documents"][0]
        print(f"text    : {text[:200]}...")


if __name__ == "__main__":
    run_ingestion()
