"""Day 10 - Step 2: split loaded documents into chunks with metadata.

This turns each :class:`document_loader.Document` into a list of small
overlapping text chunks, and attaches useful metadata to every chunk:

    Document (text + metadata)  ->  list[Chunk]  (chunk text + richer metadata)

Design choices kept deliberately simple and explicit for learning:

* Chunking is character-based with a small overlap. Character counting is easy
  to reason about and needs no tokenizer.
* We first split a document into its named sections (the policies use
  ``Heading`` followed by a line of dashes). This lets us record which section a
  chunk came from, which is exactly the kind of metadata that makes retrieval
  answers explainable later.
* Chunk IDs are deterministic (``<document_id>::chunk-<n>``) so re-running the
  pipeline overwrites the same IDs instead of creating duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass

from document_loader import Document, load_documents

# Reasonable defaults for short policy documents: ~500 characters per chunk with
# a small 80-character overlap so a sentence split across a boundary still shows
# up whole in one of the neighbouring chunks.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80


@dataclass
class Chunk:
    """One chunk of text plus everything we know about where it came from."""

    id: str
    text: str
    metadata: dict


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a policy document into ``(section_name, section_text)`` pairs.

    The documents mark sections with a heading line followed by a line of dashes,
    for example::

        Leave Entitlement
        -----------------
        Every employee receives ...

    Any text before the first such heading is returned under the name
    ``"Header"``. If the document has no dashed headings at all, the whole text
    is returned as a single ``"Body"`` section.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_name = "Header"
    current_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        # A heading is a non-empty line immediately followed by a dashes-only line.
        is_heading = (
            line.strip()
            and next_line.strip()
            and set(next_line.strip()) == {"-"}
        )
        if is_heading:
            # Close the previous section and start a new one.
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = line.strip()
            current_lines = []
            i += 2  # skip the heading line and its dashes underline
            continue
        current_lines.append(line)
        i += 1

    if current_lines:
        sections.append((current_name, current_lines))

    # Join line lists back into text and drop sections that are only whitespace.
    result: list[tuple[str, str]] = []
    for name, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            result.append((name, body))

    if not result:
        # No headings found (or nothing but whitespace): treat as one body.
        stripped = text.strip()
        return [("Body", stripped)] if stripped else []
    return result


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split a single string into overlapping character windows.

    Input:
        text: the text to split.
        chunk_size: maximum characters per chunk (must be > 0).
        overlap: characters shared between consecutive chunks (0 <= overlap < chunk_size).

    Output:
        A list of chunk strings. Empty/whitespace text returns ``[]``. Text
        shorter than ``chunk_size`` returns a single chunk (the "very short
        document" edge case).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Turn one :class:`Document` into a list of :class:`Chunk` with metadata.

    Input:
        doc: a loaded document.
        chunk_size / overlap: passed through to :func:`chunk_text`.

    Output:
        A list of chunks. An empty document yields ``[]`` (empty-document edge
        case). Every chunk's metadata carries the document metadata plus:
        ``section``, ``chunk_index`` (position within this document), and
        ``chunk_id``.
    """
    if doc.is_empty:
        return []

    document_id = doc.metadata.get("document_id", "unknown")
    chunks: list[Chunk] = []
    index = 0
    for section_name, section_text in split_into_sections(doc.text):
        for piece in chunk_text(section_text, chunk_size, overlap):
            chunk_id = f"{document_id}::chunk-{index}"
            metadata = {
                # Copy the source document metadata onto the chunk.
                **doc.metadata,
                # Chunk-specific metadata.
                "source": doc.metadata.get("source_filename", "unknown"),
                "section": section_name,
                "chunk_index": index,
                "chunk_id": chunk_id,
            }
            chunks.append(Chunk(id=chunk_id, text=piece, metadata=metadata))
            index += 1
    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a whole list of documents into a single flat list of chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    # Day 10 manual inspection: Document -> chunks -> metadata, printed clearly.
    docs = load_documents()
    chunks = chunk_documents(docs)

    print("Document -> chunks -> metadata inspection")
    print(f"Loaded documents: {len(docs)}")
    print(f"Total chunks:     {len(chunks)}")
    print(f"Chunk size / overlap: {DEFAULT_CHUNK_SIZE} / {DEFAULT_OVERLAP}\n")

    # Print the first few chunks so the transformation is visible.
    preview_count = min(4, len(chunks))
    for chunk in chunks[:preview_count]:
        print("=" * 72)
        print(f"chunk_id : {chunk.id}")
        print(f"title    : {chunk.metadata['title']}")
        print(f"section  : {chunk.metadata['section']}")
        print(f"source   : {chunk.metadata['source']}")
        print(f"length   : {len(chunk.text)} chars")
        print("-" * 72)
        print(chunk.text)
        print()
