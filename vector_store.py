"""Day 11 - embed chunks and store them in a local, persistent Chroma database.

This is the final step of the ingestion pipeline:

    list[Chunk]  ->  embeddings (Gemini)  ->  Chroma collection (on disk)

Key points, kept simple for learning:

* Embeddings are produced with the same Gemini setup the rest of the project
  already uses (``gemini-embedding-001`` via ``model_client``). We compute the
  vectors ourselves and hand them to Chroma, so Chroma does not need to download
  its own embedding model.
* The database is a ``PersistentClient`` pointed at ``chroma_db/``, so the data
  survives restarting the Python process.
* We store each chunk with a deterministic ID and use ``upsert``. Re-running the
  ingestion overwrites the same IDs instead of piling up duplicate copies.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

import model_client
from chunker import Chunk

# Where the persistent Chroma database lives (a folder next to this file).
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

# Logical name of the collection that holds the policy chunks.
COLLECTION_NAME = "company_policies"

# Same embedding model used by the Day 8 embedding experiment.
EMBED_MODEL = "gemini-embedding-001"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input string, using Gemini.

    Input:
        texts: a non-empty list of non-empty strings.

    Output:
        A list of float vectors aligned by index with ``texts``.

    Raises:
        ValueError: empty input, or an unexpected response shape.
        model_client.ConfigurationError: missing API key.
        model_client.ProviderRequestError: the embedding call failed.
        model_client.EmptyModelResponseError: the API returned no vectors.
    """
    if not texts or any(not t or not t.strip() for t in texts):
        raise ValueError("All texts must be non-empty, non-whitespace strings.")

    # Reuse the project's existing key loading and client construction.
    api_key = model_client._get_api_key()
    client = model_client._build_client(api_key)

    try:
        response = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    except Exception as exc:  # noqa: BLE001 - normalise into the known hierarchy
        raise model_client.ProviderRequestError("Embedding request failed.") from exc

    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise model_client.EmptyModelResponseError("Embedding API returned no data.")
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Unexpected embedding count: got {len(embeddings)}, expected {len(texts)}."
        )

    vectors: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if not values:
            raise ValueError("Unexpected embedding format: missing values.")
        vectors.append(list(values))
    return vectors


def get_collection(persist_path: Path = CHROMA_PATH):
    """Open (or create) the persistent Chroma collection for policy chunks."""
    client = chromadb.PersistentClient(path=str(persist_path))
    # get_or_create keeps re-runs safe: the collection is reused, not recreated.
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks: list[Chunk], collection) -> int:
    """Embed ``chunks`` and upsert them into the Chroma ``collection``.

    Input:
        chunks: the chunks to store (their ``.id`` becomes the Chroma ID).
        collection: a Chroma collection (from :func:`get_collection`).

    Output:
        The number of chunks stored.

    Notes:
        Uses ``upsert`` with deterministic chunk IDs, so re-running does not
        create duplicate entries - existing IDs are overwritten in place.
    """
    if not chunks:
        return 0

    ids = [c.id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)
