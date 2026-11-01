"""Overnight progress tracker: one line per workstream plus a weighted overall percentage.

  python scripts/progress.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EX = ROOT / "data/processed/extractions"
ANN = ROOT / "data/processed/annotation"
RES = ROOT / "eval/results"
WEIGHTS = {"runs": 0.25, "labels": 0.30, "frontend": 0.20, "train_eval": 0.25}


def ids_in(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(x)["parsed"]["doc_id"] for x in open(path) if x.strip()}


def run_parts() -> dict:
    """{stream: (docs done, docs planned)} for every extraction stream of the overnight queue."""
    frozen = json.loads((ROOT / "eval/frozen/test_sets.json").read_text())
    src = pd.concat([pd.read_parquet(ROOT / f"data/processed/{f}.parquet", columns=["doc_id", "source"])
                     for f in ("jobs", "resumes")]).set_index("doc_id")["source"]
    public = {"kaggle_linkedin_postings", "kaggle_resume_dataset"}
    test = frozen["extraction_test"]
    test_pub = {d for d in test if src.get(d) in public}
    pool = set((ROOT / "data/processed/graph_pool.txt").read_text().split())
    teacher = set((ROOT / "data/processed/distill/teacher_ids_600.txt").read_text().split())
    parts = {}
    for m in ("asu__gpt-oss-120b", "asu__llama4-maverick-17b", "asu__gemma4-31b-it", "azure__gpt-5-mini"):
        parts[m.split("__")[1]] = (sum(len(ids_in(EX / m / f"run{k}.jsonl") & test_pub) for k in range(3)), 3 * len(test_pub))
    r0 = ids_in(EX / "asu__gpt-oss-120b/run0.jsonl")
    parts["graph-pool"] = (len(r0 & pool), len(pool))
    parts["teacher"] = (len(r0 & teacher), len(teacher))
    pub = set((ROOT / "data/processed/profiles_public.txt").read_text().split())
    parts["public profiles"] = (len(r0 & pub), len(pub))      # private profiles aren't needed for the eval (D36)
    spo = EX / "spo__asu__gpt-oss-120b/run0.jsonl"
    spo_done = {json.loads(x)["doc_id"] for x in open(spo) if x.strip()} if spo.exists() else set()
    spo_ids = set((ROOT / "data/processed/spo_ids.txt").read_text().split())
    parts["SPO"] = (len(spo_done & spo_ids), len(spo_ids))
    local = set((ROOT / "data/processed/test_local60.txt").read_text().split())   # local tier subset (D36)
    for m in ("ollama__llama3.1-8b", "ollama__qwen2.5-7b", "ollama__gemma2-9b"):
        parts[m.split("__")[1]] = (len(ids_in(EX / m / "run0.jsonl") & local), len(local))
    return parts


def runs() -> tuple[float, str]:
    parts = run_parts()
    done, total = sum(a for a, _ in parts.values()), sum(b for _, b in parts.values())
    return done / total, ", ".join(f"{k} {a}/{b}" for k, (a, b) in parts.items())


def labels() -> tuple[float, str]:
    parts = {}
    pool = pd.read_csv(ANN / "relevance_pool.csv", dtype=str).fillna("")
    src = pd.read_parquet(ROOT / "data/processed/resumes.parquet", columns=["doc_id", "source"]).set_index("doc_id")["source"]
    pub = pool[pool["profile_id"].map(src) == "kaggle_resume_dataset"]
    parts["relevance"] = ((pub.get("llm_label", pd.Series(dtype=str)) != "").sum(), len(pub))
    gold = ANN / "gold_extraction.json"
    parts["gold docs"] = (len(json.loads(gold.read_text())) if gold.exists() else 0, 120)
    for name, f, col, n in (("mentions", "skill_mentions_500.csv", "llm_gold_taxonomy_id", 500),
                            ("edges", "edge_sample_100.csv", "llm_is_correct", 100),
                            ("dedupe", "dedupe_handcheck_jobs.csv", "llm_is_duplicate", 200)):
        p = ANN / f
        k = 0
        if p.exists():
            df = pd.read_csv(p, dtype=str).fillna("")
            k = int((df[col] != "").sum()) if col in df else 0
        parts[name] = (min(k, n), n)
    w = {"relevance": 0.35, "gold docs": 0.35, "mentions": 0.12, "edges": 0.08, "dedupe": 0.10}
    pct = sum(w[k] * a / b for k, (a, b) in parts.items())
    return pct, ", ".join(f"{k} {a}/{b}" for k, (a, b) in parts.items())


FRONTEND = ["ui/components.py", "ui/theme.css", "ui/views/dashboard.py", "ui/views/match.py", "ui/views/profile.py",
            "ui/views/skill_gap.py", "ui/views/graph.py", "ui/views/assistant.py", "ui/views/models.py",
            "ui/views/research.py", "ui/views/candidates.py"]


def frontend() -> tuple[float, str]:
    done = [f for f in FRONTEND if (ROOT / f).exists()]
    tested = (ROOT / ".cache/ui_apptest_ok").exists()
    pct = (len(done) + (2 if tested else 0)) / (len(FRONTEND) + 2)
    return pct, f"{len(done)}/{len(FRONTEND)} files, app test {'passed' if tested else 'pending'}"


STEPS = {"graph rebuilt": ".cache/graph_rebuilt", "extraction eval": "eval/results/extraction_benchmark.json",
         "graph eval": "eval/results/graph_eval_llm.json", "ranking eval + LTR": "eval/results/ranking_eval_llm_label.json",
         "SPO ablation": "eval/results/spo_ablation.json", "QLoRA data": "data/processed/distill/train.jsonl",
         "QLoRA trained": ".cache/qlora_done", "explanation sheet": "data/processed/annotation/explanations_60.csv",
         "summary": ".cache/summary_final"}


def train_eval() -> tuple[float, str]:
    done = [k for k, f in STEPS.items() if (ROOT / f).exists()]
    return len(done) / len(STEPS), (", ".join(done) or "none yet") + f" ({len(done)}/{len(STEPS)})"


if __name__ == "__main__":
    rows = {"runs": runs(), "labels": labels(), "frontend": frontend(), "train_eval": train_eval()}
    overall = sum(WEIGHTS[k] * v[0] for k, v in rows.items())
    print(f"OVERALL {overall:.0%}")
    for k, (p, detail) in rows.items():
        print(f"  {k:<10} {p:>4.0%}  {detail}")
