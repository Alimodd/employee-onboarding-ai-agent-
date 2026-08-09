"""Day 12 - semantic retrieval over the existing Chroma policy collection.

This is retrieval ONLY. There is no LLM and no generated answer here. Given a
user question, we turn it into an embedding (with the SAME model used at
ingestion time) and ask Chroma for the nearest stored chunks:

    query  ->  query embedding  ->  Chroma similarity search
           ->  Top-K nearest chunks (text + metadata + distance)

Chroma does the nearest-neighbour math internally; we do not implement it by
hand. Optional metadata filters (country / department / document_type) narrow
the search using Chroma's built-in ``where`` clause.
"""

from __future__ import annotations

from vector_store import embed_texts, get_collection

# Default number of chunks to pull back for a query.
DEFAULT_TOP_K = 3

# Metadata fields we allow as simple equality filters (all exist on the chunks).
FILTERABLE_FIELDS = ("country", "department", "document_type")


def _build_where(filters: dict) -> dict | None:
    """Turn ``{field: value}`` filters into a Chroma ``where`` clause.

    Returns ``None`` when there is nothing to filter. A single filter is a plain
    ``{field: value}`` dict; multiple filters are combined with Chroma's ``$and``.
    """
    active = {k: v for k, v in filters.items() if v is not None}
    if not active:
        return None
    if len(active) == 1:
        return active
    return {"$and": [{k: v} for k, v in active.items()]}


def search_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    country: str | None = None,
    department: str | None = None,
    document_type: str | None = None,
) -> list[dict]:
    """Return the ``top_k`` stored chunks most similar to ``query``.

    Input:
        query: a non-empty natural-language question.
        top_k: how many chunks to return (must be > 0).
        country / department / document_type: optional exact-match metadata
            filters, applied by Chroma before ranking by similarity.

    Output:
        A list of dicts, nearest first, each shaped like::

            {"id": str, "text": str, "metadata": dict, "distance": float}

        ``distance`` is Chroma's vector distance (smaller = more similar).
        Returns ``[]`` if the collection is empty or nothing matches the filter.

    Calls:
        ``vector_store.embed_texts`` (query embedding) and
        ``vector_store.get_collection`` (opens the existing persistent Chroma DB
        - it does NOT rebuild or re-ingest anything).

    Stores: nothing. This is a read-only query.

    Fails when:
        * ``query`` is blank or ``top_k <= 0`` (ValueError).
        * the embedding API / key fails (model_client errors propagate).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Reuse the already-ingested persistent collection; never rebuild it here.
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Embed the query with the SAME model used for the stored chunks.
    query_embedding = embed_texts([query])[0]

    where = _build_where(
        {"country": country, "department": department, "document_type": document_type}
    )

    # Never ask Chroma for more items than it holds.
    n_results = min(top_k, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,  # None means "no filter"
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns list-of-lists (one inner list per query); we sent one query.
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"id": cid, "text": text, "metadata": meta, "distance": dist}
        for cid, text, meta, dist in zip(ids, documents, metadatas, distances)
    ]


def _print_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval results for manual inspection."""
    print("=" * 78)
    print(f"QUERY: {query}")
    print(f"Top-{len(results)} results" if results else "No results")
    for rank, hit in enumerate(results, start=1):
        meta = hit["metadata"]
        preview = hit["text"].strip().replace("\n", " ")[:160]
        print("-" * 78)
        print(f"  rank     : {rank}")
        print(f"  distance : {hit['distance']:.4f}  (smaller = more similar)")
        print(f"  id       : {hit['id']}")
        print(f"  source   : {meta.get('source')}")
        print(f"  title    : {meta.get('title')}")
        print(f"  section  : {meta.get('section')}")
        print(f"  country  : {meta.get('country')}  |  department: {meta.get('department')}")
        print(f"  preview  : {preview}...")
    print()


if __name__ == "__main__":
    # Manual retrieval demo: direct questions, a paraphrase, an unrelated query,
    # and one filtered search. No LLM is involved anywhere.
    demos = [
        ("How many days can I work remotely?", {}),
        ("What should I do if I am sick?", {}),
        ("What are the IT security rules?", {}),
        ("time off to relax and go on holiday", {}),          # paraphrase of leave
        ("What is the best pizza topping?", {}),              # unrelated
        ("How many remote work days are allowed?", {"country": "Belgium"}),  # filtered
    ]
    for query, filters in demos:
        if filters:
            print(f"(filter: {filters})")
        _print_results(query, search_documents(query, top_k=3, **filters))
