"""Offline tests for Phases 2-9 (no network, no Neo4j, no LLM calls)."""

import numpy as np
import pytest

from agents.explainer import Fact, template_explanation
from eval import stats
from eval.extraction_eval import matches, pred_entities, score_doc
from extraction.chunk import chunk_text
from extraction.distill import chunk_examples
from extraction.llm import JobChunk, coerce, parse_json, salvage
from graph.normalize import alias_table, clean, taxonomy
from ranking.cascade import minmax, mmr, rrf
from router.router import PrivacyError, Router


def test_chunks_cover_text_with_overlap():
    text = "\n".join(" ".join(f"w{i}_{j}" for j in range(12)) for i in range(100))   # 1,200 words
    chunks = chunk_text(text, 200, 500, 0.10)
    assert len(chunks) >= 3
    assert all(text[c.start:c.start + len(c.text)] == c.text for c in chunks)
    assert chunks[0].text.split()[0] == "w0_0" and chunks[-1].text.split()[-1] == "w99_11"
    assert chunks[1].start < chunks[0].start + len(chunks[0].text)                    # overlap


def test_parse_json_and_coerce():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    assert parse_json('{"a": [1, 2') is None
    c = coerce({"importance": "required", "min_years": "5+ years", "employment_type": "full-time", "x": "null"})
    assert c == {"importance": "Required", "min_years": 5.0, "employment_type": "Full-time", "x": None}


def test_salvage_keeps_valid_items():
    data = {"job": {"title": "Analyst"}, "requirements": [
        {"text": "SQL", "requirement_type": "Skill", "importance": "Required", "evidence": {"quote": "SQL", "confidence": 1}},
        {"text": "bad", "requirement_type": "Nope", "importance": "Required", "evidence": {"quote": "x", "confidence": 1}}]}
    out, dropped = salvage(data, "JobPosting")
    assert isinstance(out, JobChunk) and len(out.requirements) == 1 and dropped == 1


def test_router_blocks_private_data_on_hosted_provider():
    r = Router()
    with pytest.raises(PrivacyError):
        r.chat("extract_job", [{"role": "user", "content": "x"}], private=True, provider="asu", model="gpt-oss-120b")


def test_stats_sanity():
    a, b = np.arange(40, dtype=float), np.arange(40, dtype=float) - 1
    ci = stats.paired_bootstrap(a, b)
    assert ci["mean_diff"] == pytest.approx(1.0) and ci["ci_low"] == pytest.approx(1.0)
    assert stats.randomization_test(a, b)["p"] < 0.01
    assert stats.noninferiority(b, a, delta=3)["noninferior"]
    assert stats.holm({"x": 0.01, "y": 0.04})["y"] == pytest.approx(0.04)
    w = stats.wilson(19, 20)
    assert w["ci_low"] < 0.95 < w["ci_high"]
    m = stats.mcnemar([1, 1, 1, 0] * 10, [1, 0, 0, 0] * 10)
    assert m["a_only"] == 20 and m["b_only"] == 0 and m["p"] < 0.001


def test_extraction_matching_rule():
    assert matches("python", [], "python", [])
    assert matches("5 years of python", [(0, 17)], "5 years python experience", [(0, 25)])
    assert not matches("java", [(30, 34)], "python", [(0, 6)])
    text = "We need Python and SQL. Bachelor's degree required."
    parsed = {"job": {"title": "Unknown"}, "requirements": [
        {"text": "Python and SQL", "requirement_type": "Skill", "importance": "Required", "skills": ["Python", "SQL"],
         "evidence": {"quote": "We need Python and SQL", "char_start": 0, "char_end": 22}}]}
    gold = {"Skill": {"python": [(8, 14)], "sql": [(19, 22)]}, "Credential": {"bachelor s degree": [(24, 41)]}}
    sc = score_doc(gold, pred_entities(parsed, text))
    assert sc["Skill"]["tp"] == 2 and sc["Credential"]["fn"] == 1


def test_normalizer_aliases():
    assert clean("Python 3.8") == "python" and clean("MS Excel") == "microsoft excel"
    tax, alias = taxonomy(), alias_table()
    assert tax.iloc[alias["aws"]].canonical_name.startswith("Amazon Web Services")
    assert tax.iloc[alias["sql"]].canonical_name == "Structured query language SQL"


def test_rrf_and_mmr():
    s1, s2 = np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])
    f = rrf([s1, s2])
    assert f[1] > f[2] and f[0] > f[2]
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    assert mmr(np.array([1.0, 0.99, 0.5]), emb, k=2, lam=0.5) == [0, 2]     # skips the near-duplicate
    assert list(minmax(np.array([2.0, 4.0]))) == [0.0, 1.0]


def test_template_explanation_cites_every_sentence():
    facts = [Fact("met", 'The posting\'s required requirement "SQL" asks for SQL; the CV states "SQL".', ["a" * 16]),
             Fact("gap", 'The required "Tableau" asks for Tableau, which the CV does not mention.', ["b" * 16])]
    ex = template_explanation("c", "j", facts)
    assert [s["text"] for s in ex.sentences] == ["Matched skills: SQL.", "Missing required skills: Tableau."]
    assert all(s["cites"] for s in ex.sentences)


def test_distill_splits_items_by_chunk():
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))["llm"]
    text = " ".join(f"w{i}" for i in range(900)) + " Python"
    i = text.index("Python")
    parsed = {"job": {"title": "X"}, "requirements": [
        {"text": "Python", "requirement_type": "Skill", "importance": "Required", "skills": ["Python"],
         "evidence": {"quote": "Python", "confidence": 1, "char_start": i, "char_end": i + 6}}]}
    ex = chunk_examples(parsed, "JobPosting", text, cfg)
    import json
    targets = [json.loads(e["messages"][-1]["content"]) for e in ex]
    assert sum(len(t["requirements"]) for t in targets) == 1 and targets[-1]["requirements"]
    assert "char_start" not in json.dumps(targets)
