"""R3 gold sheets and scoring: 500 skill mentions (normalization accuracy) and 100 edges (edge precision).

  python -m graph.annotate --make     # writes data/processed/annotation/{skill_mentions_500,edge_sample_100}.csv
  python -m graph.annotate --score    # after filling gold_taxonomy_id / is_correct -> eval/results/graph_eval.json
  python -m graph.annotate --score --labels llm   # Claude's llm_* columns (D30) -> graph_eval_llm.json

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


def public_ids() -> set:
    """Kaggle (public) docs only: these sheets are labeled by Claude, who never reads resume_corpus CVs (D24, D30)."""
    import yaml
    public = set(yaml.safe_load(open("config.yaml"))["public_sources"])
    src = pd.concat([pd.read_parquet(f"data/processed/{f}.parquet", columns=["doc_id", "source"])
                     for f in ("jobs", "resumes")])
    return set(src.loc[src["source"].isin(public), "doc_id"])


def mentions_from(extractors: list[str]) -> pd.DataFrame:
    split = pd.read_csv("eval/frozen/splits.csv.gz").set_index("doc_id")["split"]
    public = public_ids()
    rows = []
    for r in load_extractions(extractors):
        p = r["parsed"]
        if split.get(p["doc_id"]) != "test" or p["doc_id"] not in public:
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
    m["gold_taxonomy_id"], m["llm_gold_taxonomy_id"] = "", ""
    if MENTIONS.exists():
        raise SystemExit(f"{MENTIONS} exists; delete it first to regenerate (it may hold your labels)")
    m.to_csv(MENTIONS, index=False)
    with driver().session() as s:
        e = s.run("""MATCH (a)-[r]->(b) WHERE r.source = 'Extracted' AND type(r) <> 'SUPPORTED_BY'
                     MATCH (ev:Evidence) WHERE ev.evidence_id IN r.evidence_ids AND ev.doc_id IN $public
                     WITH a, r, b, collect(ev) AS evs, rand() AS k ORDER BY k LIMIT $n
                     UNWIND evs AS ev
                     RETURN DISTINCT r.edge_id AS edge_id, type(r) AS rel, labels(a)[0] AS from_type,
                            coalesce(a.title, a.text, a.name, a.headline, a.candidate_id, a.job_id) AS from_node,
                            labels(b)[0] AS to_type, coalesce(b.canonical_name, b.title, b.text, b.name, b.city) AS to_node,
                            r.surface AS surface, collect(ev.text)[0] AS evidence_text""", n=n_edges,
                  public=sorted(public_ids())).data()
    edges = pd.DataFrame(e).drop_duplicates("edge_id")
    edges["is_correct"], edges["llm_is_correct"] = "", ""
    edges.to_csv(EDGES, index=False)
    print(f"wrote {len(m)} mentions -> {MENTIONS}\nwrote {len(edges)} edges -> {EDGES}")


def score(labels: str = "human") -> dict:
    """labels: "human" (gold_taxonomy_id / is_correct) or "llm" (Claude's llm_* columns, D30)."""
    gcol, ecol = ("gold_taxonomy_id", "is_correct") if labels == "human" else ("llm_gold_taxonomy_id", "llm_is_correct")
    out = {"labels": labels}
    m = pd.read_csv(MENTIONS, dtype=str).fillna("")
    m = m[m[gcol].str.strip() != ""].rename(columns={gcol: "_gold"})
    if len(m):
        for method in ("classical", "modern"):
            ok = (m[f"{method}_id"].astype(str) == m["_gold"].str.strip()).to_numpy()
            out[f"normalization_{method}"] = stats.wilson(int(ok.sum()), len(ok))
            out[f"_{method}_correct"] = ok
        out["modern_vs_classical_mcnemar"] = stats.mcnemar(out.pop("_modern_correct"), out.pop("_classical_correct"))
    e = pd.read_csv(EDGES, dtype=str).fillna("")
    e = e[e[ecol].str.strip() != ""]
    if len(e):
        ok = e[ecol].astype(int).to_numpy()
        out["edge_precision"] = stats.wilson(int(ok.sum()), len(ok))
        out["edge_precision_by_type"] = {t: stats.wilson(int(g[ecol].astype(int).sum()), len(g))
                                         for t, g in e.groupby("rel")}
    out["targets"] = {"normalization_top1": 0.90, "edge_precision": 0.90}
    tag = "" if labels == "human" else f"_{labels}"
    Path(f"eval/results/graph_eval{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--extractions", nargs="*", default=["asu__gpt-oss-120b", "ollama__llama3.1-8b"])
    ap.add_argument("--labels", default="human", choices=["human", "llm"])
    a = ap.parse_args()
    if a.make:
        make(a.extractions)
    if a.score:
        print(json.dumps(score(a.labels), indent=1, default=float))
