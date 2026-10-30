"""RQ3: ontology-guided vs. free-form SPO extraction, on the same documents and model.

  python -m eval.spo_eval --spo spo__groq__openai__gpt-oss-20b --ontology groq__openai__gpt-oss-20b

1. Duplicate-node rate: entity nodes that are the same concept under another surface form. Both graphs are
   scored with the same procedure: nodes are merged when they share a taxonomy id or their embeddings have
   cosine >= 0.9 (union-find). rate = 1 - concepts / nodes. Two-proportion z-test + bootstrap CI over docs.
2. Delta nDCG@10: LambdaMART with SPO-graph features in place of the ontology-graph features, on the same
   judged pairs (data/processed/ranking_features.parquet, written by eval.ranking_eval). Wilcoxon per profile.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import stats
from graph.normalize import Normalizer, clean
from ranking.graph_features import FEATURES as GRAPH_FEATURES
from ranking.ltr import ALL_FEATURES, cross_validated_scores

EXTRACTIONS = Path("data/processed/extractions")
SPO_FEATURES = ["s_overlap", "s_jaccard", "s_n_job_objects"]


def spo_nodes(rows: list[dict]) -> dict[str, set]:
    return {r["doc_id"]: {clean(t[k]) for t in r["triples"] for k in ("subject", "object") if clean(t[k])} for r in rows}


def ontology_nodes(rows: list[dict]) -> dict[str, set]:
    out = {}
    for r in rows:
        p = r["parsed"]
        names = set()
        if "job" in p:
            names |= {clean(s) for q in p["requirements"] for s in q.get("skills", [])}
            names |= {clean(p["job"].get(k) or "") for k in ("title", "company")}
        else:
            names |= {clean(s["name"]) for s in p.get("skills", [])}
            names |= {clean(x.get(k) or "") for x in p.get("experiences", []) for k in ("title", "company")}
            names |= {clean(c["name"]) for c in p.get("credentials", [])}
        out[p["doc_id"]] = {n for n in names if n}
    return out


def concepts(names: list[str], norm: Normalizer, cos: float = 0.9) -> dict[str, int]:
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_tax = {}
    for i, m in enumerate(norm.normalize(names, private=True)):
        if m.taxonomy_id:
            if m.taxonomy_id in by_tax:
                parent[find(i)] = find(by_tax[m.taxonomy_id])
            else:
                by_tax[m.taxonomy_id] = i
    model, _ = norm._embed_index()
    emb = model.encode(names, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    for start in range(0, len(names), 2000):
        with np.errstate(all="ignore"):
            sims = emb[start:start + 2000] @ emb.T
        for a, b in zip(*np.nonzero(sims >= cos)):
            a += start
            if a < b:
                parent[find(a)] = find(b)
    return {n: find(i) for i, n in enumerate(names)}


def dup_rate(doc_nodes: dict[str, set], norm: Normalizer) -> dict:
    names = sorted(set().union(*doc_nodes.values()))
    cid = concepts(names, norm)
    n_nodes, n_concepts = len(names), len(set(cid.values()))
    per_doc = [1 - len({cid[x] for x in v}) / len(v) if v else 0.0 for v in doc_nodes.values()]
    return {"nodes": n_nodes, "concepts": n_concepts, "duplicates": n_nodes - n_concepts,
            "rate": 1 - n_concepts / n_nodes if n_nodes else 0.0, "per_doc_mean": stats.bootstrap_ci(per_doc)}


def spo_pair_features(feats: pd.DataFrame, nodes: dict[str, set]) -> pd.DataFrame:
    f = feats.copy()
    rows = []
    for q, d in zip(f["query_id"], f["doc_id"]):
        c, j = nodes.get(q), nodes.get(d)
        if c is None or j is None:
            rows.append([np.nan] * 3)
            continue
        inter = len(c & j)
        rows.append([inter / len(j) if j else np.nan, inter / len(c | j) if c | j else np.nan, len(j)])
    f[SPO_FEATURES] = rows
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spo", required=True)
    ap.add_argument("--ontology", required=True)
    a = ap.parse_args()
    spo_rows = [json.loads(x) for x in open(EXTRACTIONS / a.spo / "run0.jsonl") if x.strip()]
    ont_rows = [json.loads(x) for x in open(EXTRACTIONS / a.ontology / "run0.jsonl") if x.strip()]
    common = {r["doc_id"] for r in spo_rows} & {r["parsed"]["doc_id"] for r in ont_rows}
    s_nodes = {k: v for k, v in spo_nodes(spo_rows).items() if k in common}
    o_nodes = {k: v for k, v in ontology_nodes(ont_rows).items() if k in common}
    norm = Normalizer("modern")
    res = {"docs": len(common), "spo": dup_rate(s_nodes, norm), "ontology": dup_rate(o_nodes, norm)}
    res["dup_rate_test"] = stats.two_proportion_z(res["spo"]["duplicates"], res["spo"]["nodes"],
                                                  res["ontology"]["duplicates"], res["ontology"]["nodes"])
    feats_path = Path("data/processed/ranking_features.parquet")
    if feats_path.exists():
        from ranx import Qrels, Run, evaluate
        df = pd.read_parquet(feats_path)
        df = spo_pair_features(df, s_nodes)
        sets = {"ontology_graph": ALL_FEATURES + ["f_ce"],
                "spo_graph": [f for f in ALL_FEATURES if f not in GRAPH_FEATURES] + SPO_FEATURES + ["f_ce"]}
        qrels = Qrels({q: {d: int(l) for d, l in zip(g["doc_id"], g["label"])} for q, g in df.groupby("query_id")})
        ndcg = {}
        for name, feats in sets.items():
            scores, sdf = cross_validated_scores(df, feats)
            run = {}
            for q, d, s in zip(sdf["query_id"], sdf["doc_id"], scores):
                run.setdefault(q, {})[d] = float(s)
            ndcg[name] = np.asarray(evaluate(qrels, Run(run), "ndcg@10", return_mean=False))
        res["ndcg@10"] = {k: float(v.mean()) for k, v in ndcg.items()}
        res["ndcg_test_ontology_vs_spo"] = {**stats.wilcoxon(ndcg["ontology_graph"], ndcg["spo_graph"]),
                                            **stats.paired_bootstrap(ndcg["ontology_graph"], ndcg["spo_graph"])}
    Path("eval/results/spo_ablation.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
