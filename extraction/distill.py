"""R5 / RQ2 distillation data: teacher labels -> per-chunk chat examples for QLoRA (notebooks/qlora_extraction.ipynb).

  python -m extraction.distill --make-set 1500          # stratified train-split docs (public sources: hosted teacher)
  python -m extraction.run --set ids:data/processed/distill/teacher_ids.txt --extractor asu/gpt-oss-120b
  python -m extraction.distill --export asu__gpt-oss-120b
        -> data/processed/distill/{train,val}.jsonl and hand_correct_100.jsonl (fix these by hand; kept out of train)

Each merged teacher output is split back into its chunks: an item belongs to the chunk its evidence offset
falls in, so every training target is exactly what the model should answer for that chunk's prompt.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from extraction.chunk import chunk_text
from extraction.llm import HEAD_FIELD, LIST_FIELDS
from extraction.prompts import messages

OUT = Path("data/processed/distill")
SEED = 573


def make_set(n: int, cv_share: float = 0.4):
    jobs = pd.read_parquet("data/processed/jobs.parquet", columns=["doc_id", "split", "in_working_set", "is_canonical",
                                                                  "industry_group"])
    cvs = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "split", "in_working_set", "is_canonical",
                                                                    "source", "category"])
    j = jobs[(jobs["split"] == "train") & jobs["in_working_set"] & jobs["is_canonical"]]
    c = cvs[(cvs["split"] == "train") & cvs["in_working_set"] & cvs["is_canonical"]
            & (cvs["source"] == "kaggle_resume_dataset")]
    n_cv = min(int(n * cv_share), len(c))
    pick = lambda df, k, col: df.groupby(col, group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, round(k * len(g) / len(df)))), random_state=SEED))
    ids = pick(j, n - n_cv, "industry_group")["doc_id"].tolist() + pick(c, n_cv, "category")["doc_id"].tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "teacher_ids.txt").write_text("\n".join(ids))
    print(f"{len(ids)} teacher docs -> {OUT / 'teacher_ids.txt'}")


def strip_offsets(obj):
    if isinstance(obj, dict):
        return {k: strip_offsets(v) for k, v in obj.items() if k not in ("char_start", "char_end")}
    if isinstance(obj, list):
        return [strip_offsets(x) for x in obj]
    return obj


def chunk_examples(parsed: dict, doc_type: str, text: str, cfg: dict) -> list[dict]:
    chunks = chunk_text(text, cfg["chunk_words"][0], cfg["chunk_words"][1], cfg["chunk_overlap"])
    head = HEAD_FIELD[doc_type][0]
    out = []
    for c in chunks:
        lo, hi = c.start, c.start + len(c.text)
        target = {head: strip_offsets(parsed.get(head)) if c.index == 0 else None}
        for field in LIST_FIELDS[doc_type]:
            items = [x for x in parsed.get(field, [])
                     if x["evidence"].get("char_start") is not None and lo <= x["evidence"]["char_start"] < hi]
            target[field] = strip_offsets(items)
        part = f" (part {c.index + 1} of {len(chunks)})" if len(chunks) > 1 else ""
        msgs = messages(doc_type, c.text, part) + [{"role": "assistant", "content": json.dumps(target)}]
        out.append({"messages": msgs})
    return out


def export(teacher: str, n_hand: int = 100, val_share: float = 0.05):
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))["llm"]
    rows = [json.loads(x) for x in open(Path("data/processed/extractions") / teacher / "run0.jsonl") if x.strip()]
    from extraction.run import load_docs
    docs = load_docs(f"ids:{OUT / 'teacher_ids.txt'}").set_index("doc_id")
    # the teacher's run0 also holds its test-set and graph-pool extractions: keep teacher-set (train split) docs only
    rows = [r for r in rows if r["parsed"]["doc_id"] in docs.index]
    random.Random(SEED).shuffle(rows)
    hand, rest = rows[:n_hand], rows[n_hand:]
    n_val = max(1, int(len(rest) * val_share))
    for name, part in (("val", rest[:n_val]), ("train", rest[n_val:])):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in part:
                p = r["parsed"]
                dt = "JobPosting" if "job" in p else "CV"
                for ex in chunk_examples(p, dt, docs.at[p["doc_id"], "text"], cfg):
                    f.write(json.dumps(ex) + "\n")
    with open(OUT / "hand_correct_100.jsonl", "w") as f:
        for r in hand:
            f.write(json.dumps({"doc_id": r["parsed"]["doc_id"], "teacher": r["parsed"], "corrected": None}) + "\n")
    print(f"train/val/hand-correct written to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-set", type=int, default=0)
    ap.add_argument("--export", default="")
    a = ap.parse_args()
    if a.make_set:
        make_set(a.make_set)
    if a.export:
        export(a.export)
