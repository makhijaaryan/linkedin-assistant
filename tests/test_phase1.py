import pandas as pd
import pytest
from pydantic import ValidationError

from ingest.dedupe import find_duplicates, signatures
from ingest.text import clean_text, mask_contacts
from ontology import ontology as O
from ontology.schema import ParsedJob, locate_evidence


def test_mask_contacts():
    text, n = mask_contacts("Mail jane.doe@x.com, call (480) 555-1234 or see https://x.com/jobs")
    assert n == 3
    assert "[EMAIL]" in text and "[PHONE]" in text and "[URL]" in text and "555" not in text


def test_clean_text_strips_html():
    assert clean_text('Data <span class="hl">Engineer</span> &amp; ML') == "Data Engineer & ML"


def test_dedupe_requires_same_title():
    body = "We are hiring. " + " ".join(f"word{i}" for i in range(200))
    df = pd.DataFrame({"doc_id": ["a", "b", "c"], "title": ["Sales Manager", "Sales Manager", "VP of Sales"],
                       "company": ["Acme"] * 3, "text": [body, body + " Apply now.", body]})
    cluster, _ = find_duplicates(df, ["title", "company"], signatures(df["text"], 128, 5, 573), 0.9)
    assert cluster[0] == cluster[1]      # same title + company + near-identical text
    assert cluster[2] != cluster[0]      # same boilerplate, different title -> not a duplicate


def test_schema_rejects_unknown_enum_and_locates_evidence():
    with pytest.raises(ValidationError):
        ParsedJob.model_validate({"doc_id": "j", "extractor": "x", "extractor_version": "0",
                                  "job": {"title": "t", "seniority": "Senior"}})
    job = ParsedJob.model_validate({
        "doc_id": "j", "extractor": "x", "extractor_version": "0", "job": {"title": "ML Engineer"},
        "requirements": [{"text": "3+ years Python", "requirement_type": "Experience", "importance": "Required",
                          "evidence": {"quote": "3+ years  of python", "confidence": 0.8}}]})
    doc = "Requirements: 3+ years of Python."
    ev = locate_evidence(doc, job.requirements[0].evidence)
    assert doc[ev.char_start:ev.char_end] == "3+ years of Python"


def test_ontology_matches_proposal():
    assert len(O.ENTITIES) == 11
    names = {r[0] for r in O.RELATIONS}
    proposal = {"HAS_SKILL", "HELD", "USED", "AT", "POSTED_BY", "FOR_ROLE", "HAS_REQUIREMENT", "TARGETS",
                "RELATED_TO", "TYPICAL_FOR", "SUPPORTED_BY", "EARNED", "BUILT", "AS_ROLE"}
    assert proposal <= names and names - proposal == O.ADDED_RELATIONS
