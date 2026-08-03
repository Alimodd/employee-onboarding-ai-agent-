"""Validation tests for the synthetic company knowledge base (Northstar Analytics).

These tests check structural integrity only: files exist, the manifest matches
the files, IDs are unique, required metadata is present, and the core HR facts do
not contradict the prompt-only FAQ policy. No embeddings, chunking, or retrieval
are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hr_faq

DOCS_DIR = Path(__file__).resolve().parent.parent / "company_docs"
MANIFEST = DOCS_DIR / "manifest.json"

EXPECTED_FILES = {
    "annual_leave_policy.txt",
    "remote_work_policy.txt",
    "sick_leave_policy.txt",
    "it_security_policy.txt",
    "code_of_conduct.txt",
    "employee_benefits_guide.txt",
    "onboarding_checklist.txt",
}

REQUIRED_META_FIELDS = {
    "document_id",
    "title",
    "source_filename",
    "document_type",
    "version",
    "effective_date",
    "country",
    "department",
    "audience",
}


def _load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_all_seven_files_exist():
    for name in EXPECTED_FILES:
        assert (DOCS_DIR / name).is_file(), f"missing {name}"


def test_no_document_is_empty():
    for name in EXPECTED_FILES:
        text = (DOCS_DIR / name).read_text(encoding="utf-8")
        assert text.strip(), f"{name} is empty"


def test_manifest_exists_and_lists_seven():
    manifest = _load_manifest()
    assert len(manifest["documents"]) == 7


def test_manifest_filenames_match_real_files():
    manifest = _load_manifest()
    listed = {d["source_filename"] for d in manifest["documents"]}
    assert listed == EXPECTED_FILES
    for d in manifest["documents"]:
        assert (DOCS_DIR / d["source_filename"]).is_file()


def test_document_ids_unique():
    manifest = _load_manifest()
    ids = [d["document_id"] for d in manifest["documents"]]
    assert len(ids) == len(set(ids))


def test_required_metadata_fields_present():
    manifest = _load_manifest()
    for d in manifest["documents"]:
        assert REQUIRED_META_FIELDS.issubset(d.keys())
        assert all(str(d[f]).strip() for f in REQUIRED_META_FIELDS)


def test_versions_and_dates_consistent():
    manifest = _load_manifest()
    for d in manifest["documents"]:
        assert d["version"] == "1.0"
        assert d["effective_date"] == "2026-01-01"


def test_documents_declared_synthetic():
    for name in EXPECTED_FILES:
        text = (DOCS_DIR / name).read_text(encoding="utf-8").lower()
        assert "synthetic" in text and "educational" in text


def test_core_facts_match_faq_policy():
    """The knowledge base must not contradict the Day-7 prompt-only policy."""
    leave = (DOCS_DIR / "annual_leave_policy.txt").read_text(encoding="utf-8").lower()
    remote = (DOCS_DIR / "remote_work_policy.txt").read_text(encoding="utf-8").lower()
    sick = (DOCS_DIR / "sick_leave_policy.txt").read_text(encoding="utf-8").lower()

    # Facts asserted by the FAQ policy.
    faq = hr_faq.HR_POLICY.lower()
    assert "20 working days" in faq and "20 working days" in leave
    assert "2 days per week" in faq and "2 days per week" in remote
    assert "before 09:00" in faq and "before 09:00" in sick
    # Working hours appear in the remote-work doc (which restates them).
    assert "09:00 to 17:00" in faq and "09:00 to 17:00" in remote
