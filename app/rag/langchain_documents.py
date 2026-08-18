"""Day 16 - load the company policies as LangChain Documents and split them.

Reuses the existing loaders instead of re-reading files a second way:
    document_loader.load_documents()   -> our Document objects (text + metadata)
    chunker.split_into_sections()      -> (section_name, section_text) pairs

We then wrap each section as a LangChain ``Document`` (so section metadata is
preserved) and run a ``RecursiveCharacterTextSplitter`` over them. Metadata is
carried onto every resulting chunk.

    policy files -> LangChain Documents (per section) -> split chunks (metadata kept)
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from chunker import split_into_sections
from document_loader import load_documents

# Metadata keys we guarantee on every chunk (consistent structure even if a
# particular document leaves some of them as "unknown").
METADATA_KEYS = (
    "source",
    "source_filename",
    "document_id",
    "title",
    "document_type",
    "section",
    "country",
    "department",
    "effective_date",
    "version",
)


def _base_metadata(doc_metadata: dict, section: str) -> dict:
    """Build a consistent metadata dict for a chunk from a document's metadata."""
    meta = {key: doc_metadata.get(key, "unknown") for key in METADATA_KEYS}
    # "source" mirrors the filename for readable citations.
    meta["source"] = doc_metadata.get("source_filename", "unknown")
    meta["section"] = section
    return meta


def load_langchain_documents() -> list[LCDocument]:
    """Return one LangChain Document per (non-empty) policy section.

    Output: list of ``langchain_core.documents.Document``; empty documents are
    skipped (empty-file edge case). Very short documents produce a single
    section. Missing metadata is filled with ``"unknown"``.
    """
    lc_docs: list[LCDocument] = []
    for doc in load_documents():
        if doc.is_empty:
            continue  # empty-file edge case: nothing to index
        for section_name, section_text in split_into_sections(doc.text):
            lc_docs.append(
                LCDocument(
                    page_content=section_text,
                    metadata=_base_metadata(doc.metadata, section_name),
                )
            )
    return lc_docs


def split_documents(
    documents: list[LCDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[LCDocument]:
    """Split LangChain Documents into overlapping chunks, preserving metadata.

    Input: LangChain Documents (e.g. from :func:`load_langchain_documents`).
    Output: smaller LangChain Documents; each keeps its parent's metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split() -> list[LCDocument]:
    """Convenience: load company policies and return split, metadata-rich chunks."""
    return split_documents(load_langchain_documents())


if __name__ == "__main__":
    chunks = load_and_split()
    print(f"LangChain chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print("-" * 60)
        print("metadata:", {k: chunk.metadata.get(k) for k in ("title", "section", "source")})
        print(chunk.page_content[:160], "...")
