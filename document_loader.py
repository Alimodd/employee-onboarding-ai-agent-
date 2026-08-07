"""Day 10 - Step 1: load the synthetic policy documents from disk.

This module reads the seven Northstar Analytics policy files listed in
``company_docs/manifest.json`` and returns them as simple ``Document`` objects
that pair the raw text with its metadata.

It intentionally does NO chunking, embedding, or retrieval. Its only job is:

    manifest.json + *.txt files  ->  list[Document]

Keeping this separate makes the data-flow easy to see and easy to test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Folder that holds the .txt policies and the manifest.
DOCS_DIR = Path(__file__).resolve().parent / "company_docs"
MANIFEST_PATH = DOCS_DIR / "manifest.json"

# Metadata keys we try to attach to every document. Missing keys are filled
# with "unknown" so downstream code (and Chroma) never sees ``None``.
METADATA_FIELDS = (
    "document_id",
    "title",
    "document_type",
    "version",
    "effective_date",
    "country",
    "department",
    "audience",
    "source_filename",
)


@dataclass
class Document:
    """One loaded policy document: its full text plus its metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True when the document has no usable (non-whitespace) text."""
        return not self.text or not self.text.strip()


def _clean_metadata(entry: dict) -> dict:
    """Return a metadata dict with every expected field present as a string.

    Missing or blank values become ``"unknown"``. This is the "missing
    metadata" edge case the assignment asks us to handle: we never crash and we
    never store ``None``.
    """
    meta: dict = {}
    for key in METADATA_FIELDS:
        value = entry.get(key)
        if value is None or not str(value).strip():
            meta[key] = "unknown"
        else:
            meta[key] = str(value).strip()
    return meta


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Load every document listed in the manifest.

    Input:
        docs_dir: folder containing ``manifest.json`` and the ``.txt`` files.

    Output:
        A list of :class:`Document`, one per manifest entry, each holding the
        file's text and its cleaned metadata. Files that are missing on disk or
        empty are still returned (with empty text) so the caller can decide what
        to do - we do not silently drop them.

    Raises:
        FileNotFoundError: if the manifest itself is missing.
    """
    manifest_path = docs_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    documents: list[Document] = []
    for entry in manifest.get("documents", []):
        metadata = _clean_metadata(entry)
        source = metadata["source_filename"]
        file_path = docs_dir / source

        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8")
        else:
            # Missing file edge case: keep the entry but with empty text.
            text = ""

        documents.append(Document(text=text, metadata=metadata))

    return documents


if __name__ == "__main__":
    # Manual inspection: show what was loaded and a short preview of each doc.
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}\n")
    for doc in docs:
        preview = doc.text.strip().replace("\n", " ")[:70]
        flag = "  [EMPTY]" if doc.is_empty else ""
        print(f"- {doc.metadata['document_id']}: {doc.metadata['title']}{flag}")
        print(f"    file: {doc.metadata['source_filename']}  chars: {len(doc.text)}")
        print(f"    preview: {preview}...\n")
