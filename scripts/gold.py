"""Claude's gold extraction labels (D30), written in the Label Studio export format eval.extraction_eval reads.

  python scripts/gold.py show <n>          # print public test doc n (0..119) with its doc_id
  python scripts/gold.py add <doc_id> < entities.txt   # one "Label | exact text" per line; every occurrence is marked
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

GOLD = Path("data/processed/annotation/gold_extraction.json")
LIMIT, SHOW_ALL = 6000, 7000     # long postings: only the first 6,000 chars are annotated (the rest is legal boilerplate)
LABELS = ["Skill", "Requirement", "Experience", "Credential", "Project", "Company", "Occupation", "Location"]
docs = [json.loads(x) for x in open("data/processed/test_sets/extraction_test.jsonl")]
src = pd.concat([pd.read_parquet(f"data/processed/{f}.parquet", columns=["doc_id", "source"])
                 for f in ("jobs", "resumes")]).set_index("doc_id")["source"]
public = [d for d in docs if src[d["doc_id"]] in ("kaggle_linkedin_postings", "kaggle_resume_dataset")]
gold = json.loads(GOLD.read_text()) if GOLD.exists() else []

def spans(text: str, ents: list) -> tuple[list, list]:
    """Mark every occurrence; a single-word entity must sit on word boundaries (so "R" doesn't match inside
    words and "Excel" doesn't match "Excellent")."""
    result, missing = [], []
    for lab, ent in ents:
        assert lab in LABELS, lab
        pat = r"\s+".join(map(re.escape, ent.split()))
        if len(ent.split()) == 1:   # only single words: multi-word phrases can't hide inside a word, and postings
            # often glue sentences together ("...architectureExperience"), which word boundaries would reject
            pat = (r"(?<![A-Za-z0-9])" if ent[:1].isalnum() else "") + pat + (r"(?![A-Za-z0-9])" if ent[-1:].isalnum() else "")
        hits = list(re.finditer(pat, text, re.IGNORECASE))
        if not hits:
            missing.append(f"{lab}: {ent}")
        for m in hits:
            result.append({"type": "labels", "from_name": "label", "to_name": "text",
                           "value": {"start": m.start(), "end": m.end(), "text": text[m.start():m.end()], "labels": [lab]}})
    return result, missing


if sys.argv[1] == "rebuild":      # recompute spans of every labeled doc from its distinct (label, text) pairs
    for t in gold:
        ents = sorted({(r["value"]["labels"][0], r["value"]["text"]) for r in t["annotations"][-1]["result"]})
        seen, uniq = set(), []
        for lab, e in ents:
            if (lab, e.lower()) not in seen:
                seen.add((lab, e.lower()))
                uniq.append((lab, e))
        res, _ = spans(t["data"]["text"], uniq)
        t["annotations"][-1]["result"] = [r for r in res if r["value"]["start"] < t["data"]["annotated_upto"]]
    GOLD.write_text(json.dumps(gold))
    print(f"rebuilt {len(gold)} docs, {sum(len(t['annotations'][-1]['result']) for t in gold)} spans")
elif sys.argv[1] == "show":
    d = public[int(sys.argv[2])]
    done = {t["data"]["doc_id"] for t in gold}
    print(f"[{sys.argv[2]}/{len(public)}] {d['doc_id']} {d['doc_type']} {'(DONE)' if d['doc_id'] in done else ''}")
    t = d["text"]
    print(t if len(t) <= SHOW_ALL else t[:LIMIT] + f"\n[... {len(t) - LIMIT} more chars not annotated ...]")
elif sys.argv[1] == "add":
    d = next(x for x in public if x["doc_id"] == sys.argv[2])
    ents = [tuple(x.strip() for x in line.split("|", 1)) for line in sys.stdin.read().splitlines() if "|" in line]
    result, missing = spans(d["text"], ents)
    text = d["text"]
    gold = [t for t in gold if t["data"]["doc_id"] != d["doc_id"]]
    upto = len(text) if len(text) <= SHOW_ALL else LIMIT
    result = [r for r in result if r["value"]["start"] < upto]
    gold.append({"data": {"doc_id": d["doc_id"], "doc_type": d["doc_type"], "text": text, "annotated_upto": upto},
                 "annotations": [{"result": result, "was_cancelled": False, "completed_by": "claude (D30)"}]})
    GOLD.write_text(json.dumps(gold))
    print(f"{d['doc_id']}: {len(result)} spans; {len(gold)} docs done" + (f"; NOT FOUND: {missing}" if missing else ""))
