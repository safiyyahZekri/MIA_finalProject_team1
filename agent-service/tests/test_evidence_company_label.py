"""EVIDENCE_COMPANY_LABEL shows the company a passage's document names in its
own page headers (retrieval-api company metadata) next to the passage, so a
page that never prints the company can still be tied to it."""

from fastapi.testclient import TestClient

from app import llm as llm_module
from app import main

EVIDENCE = {"document_id": "d", "page": 1, "metadata": {"company": "CTS  CORPORATION"}}


def test_the_label_names_the_company_only_when_switched_on(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "EVIDENCE_COMPANY_LABEL", False)
    assert llm_module.evidence_source_label(EVIDENCE) == "d p1"

    monkeypatch.setattr(llm_module.settings, "EVIDENCE_COMPANY_LABEL", True)
    assert llm_module.evidence_source_label(EVIDENCE) == "d p1 company=CTS CORPORATION"
    assert (
        llm_module.evidence_source_label({**EVIDENCE, "score": 0.5}, include_score=True)
        == "d p1 score=0.5 company=CTS CORPORATION"
    )


def test_passages_without_company_metadata_keep_their_label(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "EVIDENCE_COMPANY_LABEL", True)
    for evidence in (
        {"document_id": "d", "page": 2},
        {"document_id": "d", "page": 2, "metadata": {"company": None}},
        {"document_id": "d", "page": 2, "metadata": {"company": "  "}},
        {"document_id": "d", "page": 2, "metadata": "not a dict"},
    ):
        assert llm_module.evidence_source_label(evidence) == "d p2"


def test_the_grader_sees_the_company_label(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "EVIDENCE_COMPANY_LABEL", True)
    snippets = llm_module.extraction_snippets([{**EVIDENCE, "text": "Row 4: United States | 106,757"}])
    assert "company=CTS CORPORATION" in snippets


def test_health_reports_the_switch():
    config = TestClient(main.app).get("/health").json()["config"]
    assert config["evidence_company_label"] is False
