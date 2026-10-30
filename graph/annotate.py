"""R3 gold sheets and scoring: 500 skill mentions (normalization accuracy) and 100 edges (edge precision).

  python -m graph.annotate --make     # writes data/processed/annotation/{skill_mentions_500,edge_sample_100}.csv
  python -m graph.annotate --score    # after filling gold_taxonomy_id / is_correct -> eval/results/graph_eval.json

Mentions come from test-split documents only. Each sheet row shows both normalizers' picks and the
top-5 candidates, so labeling is mostly choosing; `gold_taxonomy_id` = an id, or `none` when no
O*NET entry is the same skill (then a correct normalizer must also output none).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import stats
from graph.build import driver, load_extractions
from graph.normalize import Normalizer, taxonomy

ANN = Path("data/processed/annotation")
MENTIONS, EDGES = ANN / "skill_mentions_500.csv", ANN / "edge_sample_100.csv"
SEED = 573


def mentions_from(extractors: list[str]) -> pd.DataFrame:
    split = pd.read_csv("eval/frozen/splits.csv.gz").set_index("doc_id")["split"]
    rows = []
    for r in load_extractions(extractors):
        p = r["parsed"]
        if split.get(p["doc_id"]) != "test":
            continue
        if "job" in p:
            for q in p["requirements"]:
                rows += [{"doc_id": p["doc_id"], "mention": s, "context": q["evidence"]["quote"]} for s in q["skills"]]
        else:
            rows += [{"doc_id": p["doc_id"], "mention": s["name"], "context": s["evidence"]["quote"]} for s in p["skills"]]
    return pd.DataFrame(rows)


def make(extractors: list[str], n: int = 500, n_edges: int = 100):
    ANN.mkdir(parents=True, exist_ok=True)
    m = mentions_from(extractors).drop_duplicates(["doc_id", "mention"])
    m = m.sample(min(n, len(m)), random_state=SEED).reset_index(drop=True)
    tax = taxonomy()
    for method in ("classical", "modern"):
        res = Normalizer(method).normalize(m["mention"].tolist(), m["context"].tolist())
        m[f"{method}_id"] = [r.taxonomy_id or "none" for r in res]
        m[f"{method}_name"] = [r.canonical_name if r.taxonomy_id else "" for r in res]
    cands = Normalizer("modern").candidates(m["mention"].tolist(), k=5)
    m["candidates"] = [" || ".join(f"{tax.iloc[i].taxonomy_id} = {tax.iloc[i].canonical_name}" for i, _ in c)
                       for c in cands]
    m["gold_taxonomy_id"] = ""
    if MENTIONS.exists():
        raise SystemExit(f"{MENTIONS} exists; delete it first to regenerate (it may hold your labels)")
    m.to_csv(MENTIONS, index=False)
    with driver().session() as s:
        e = s.run("""MATCH (a)-[r]->(b) WHERE r.source = 'Extracted' AND type(r) <> 'SUPPORTED_BY'
                     WITH a, r, b, rand() AS k ORDER BY k LIMIT $n
                     MATCH (ev:Evidence) WHERE ev.evidence_id IN r.evidence_ids
                     RETURN r.edge_id AS edge_id, type(r) AS rel, labels(a)[0] AS from_type,
                            coalesce(a.title, a.text, a.name, a.headline, a.candidate_id, a.job_id) AS from_node,
                            labels(b)[0] AS to_type, coalesce(b.canonical_name, b.title, b.text, b.name, b.city) AS to_node,
                            r.surface AS surface, collect(ev.text)[0] AS evidence_text""", n=n_edges).data()
    edges = pd.DataFrame(e)
    edges["is_correct"] = ""
    edges.to_csv(EDGES, index=False)
    print(f"wrote {len(m)} mentions -> {MENTIONS}\nwrote {len(edges)} edges -> {EDGES}")


def score() -> dict:
    out = {}
    m = pd.read_csv(MENTIONS).dropna(subset=["gold_taxonomy_id"])
    m = m[m["gold_taxonomy_id"].astype(str).str.strip() != ""]
    if len(m):
        for method in ("classical", "modern"):
            ok = (m[f"{method}_id"].astype(str) == m["gold_taxonomy_id"].astype(str).str.strip()).to_numpy()
            out[f"normalization_{method}"] = stats.wilson(int(ok.sum()), len(ok))
            out[f"_{method}_correct"] = ok
        out["modern_vs_classical_mcnemar"] = stats.mcnemar(out.pop("_modern_correct"), out.pop("_classical_correct"))
    e = pd.read_csv(EDGES).dropna(subset=["is_correct"])
    e = e[e["is_correct"].astype(str).str.strip() != ""]
    if len(e):
        ok = e["is_correct"].astype(int).to_numpy()
        out["edge_precision"] = stats.wilson(int(ok.sum()), len(ok))
        out["edge_precision_by_type"] = {t: stats.wilson(int(g["is_correct"].astype(int).sum()), len(g))
                                         for t, g in e.groupby("rel")}
    out["targets"] = {"normalization_top1": 0.90, "edge_precision": 0.90}
    Path("eval/results/graph_eval.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--extractions", nargs="*", default=["groq__openai__gpt-oss-20b", "ollama__llama3.1-8b"])
    a = ap.parse_args()
    if a.make:
        make(a.extractions)
    if a.score:
        print(json.dumps(score(), indent=1, default=float))
