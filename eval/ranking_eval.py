"""R4/R5 ranking evaluation on the 40 judged profiles (pooled setting, D23) + pre-registered tests.

  python -m eval.ranking_eval                      # labels = human `label` column of relevance_pool.csv
  python -m eval.ranking_eval --labels llm_label   # secondary: LLM-judged labels (reported separately)

Every system ranks the same 20 pooled jobs per profile. The learned systems are LambdaMART models
scored out-of-fold (grouped by profile). Ablations remove one feature group from the full cascade.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ranx import Qrels, Run, compare, evaluate

from eval import stats
from ranking.cascade import Corpus, cross_scores, cv_query, minmax, mmr
from ranking.graph_features import FEATURES as GRAPH_FEATURES, GraphView
from ranking.live import save_models
from ranking.ltr import ALL_FEATURES, cross_validated_scores, fit, pair_features
from ranking.pool import POOL

RESULTS = Path("eval/results")
METRICS = ["ndcg@10", "mrr@10", "precision@5"]
FEATURE_SETS = {
    "ltr_full": ALL_FEATURES + ["f_ce"],
    "ablate_no_graph": [f for f in ALL_FEATURES if f not in GRAPH_FEATURES] + ["f_ce"],
    "ablate_no_rrf": [f for f in ALL_FEATURES if f != "f_rrf"] + ["f_ce"],
    "ablate_no_cross_encoder": ALL_FEATURES,
}


def build_features(pool: pd.DataFrame, use_graph: bool = True) -> pd.DataFrame:
    jobs, cvs = Corpus("jobs"), Corpus("cvs")
    graph = GraphView() if use_graph else None
    frames = []
    for pid, g in pool.groupby("profile_id", sort=False):
        q = cv_query(pid, cvs)
        s = jobs.first_stage(q)
        idx = np.array([jobs.pos[j] for j in g["job_id"]])
        f = pair_features(q, jobs, s, idx, graph)
        f["f_ce"] = minmax(cross_scores(q.text, [jobs.meta.iloc[i]["title"] + "\n" + jobs.text[i] for i in idx]))
        order = jobs.classical(q, s, idx, k=len(idx))
        rank = {jobs.meta.iloc[i]["doc_id"]: r for r, i in enumerate(order)}
        f["classical"] = [-rank[d] for d in f["doc_id"]]
        f["_emb_row"] = idx
        frames.append(f.merge(g[["job_id", "label"]].rename(columns={"job_id": "doc_id"}), on="doc_id"))
        print(f"  features {pid}")
    out = pd.concat(frames, ignore_index=True)
    out.attrs["dense"] = jobs.dense
    return out


def system_scores(df: pd.DataFrame, dense: np.ndarray) -> dict[str, pd.Series]:
    sys = {name: df[f"f_{name}"] for name in ("bm25", "tfidf", "jaccard", "dense", "rrf")}
    sys["classical"] = df["classical"]
    sys["cross_encoder"] = df["f_ce"]
    for name, feats in FEATURE_SETS.items():
        scores, sorted_df = cross_validated_scores(df, feats)
        sys[name] = pd.Series(scores, index=sorted_df.set_index(["query_id", "doc_id"]).index) \
            .reindex(df.set_index(["query_id", "doc_id"]).index).to_numpy()
        sys[name] = pd.Series(sys[name], index=df.index)
    # full cascade = LambdaMART (all features incl. cross-encoder) -> MMR diverse top 10
    casc = pd.Series(0.0, index=df.index)
    for _, g in df.groupby("query_id"):
        rel = sys["ltr_full"].loc[g.index].to_numpy()
        picked = mmr(rel, dense[g["_emb_row"].to_numpy()], k=10)
        for r, pos in enumerate(picked):
            casc.loc[g.index[pos]] = 100 - r
        rest = [i for i in range(len(g)) if i not in picked]
        for r, pos in enumerate(sorted(rest, key=lambda i: -rel[i])):
            casc.loc[g.index[pos]] = -r
    sys["cascade_mmr"] = casc
    return sys


def to_run(df: pd.DataFrame, scores: pd.Series, name: str) -> Run:
    d = {}
    for q, doc, s in zip(df["query_id"], df["doc_id"], scores):
        d.setdefault(q, {})[doc] = float(s) if np.isfinite(s) else -1e9
    return Run(d, name=name)


def analyze(df: pd.DataFrame, sys: dict, label_col: str) -> dict:
    qrels = Qrels({q: {d: int(l) for d, l in zip(g["doc_id"], g["label"])} for q, g in df.groupby("query_id")})
    runs = {name: to_run(df, s, name) for name, s in sys.items()}
    per_query = {name: {m: np.asarray(evaluate(qrels, r, m, return_mean=False)) for m in METRICS}
                 for name, r in runs.items()}
    means = {name: {m: float(v.mean()) for m, v in pq.items()} for name, pq in per_query.items()}
    res = {"labels": label_col, "n_profiles": int(df["query_id"].nunique()), "means": means}

    def test(a, b, m="ndcg@10"):
        x, y = per_query[a][m], per_query[b][m]
        return {**stats.paired_bootstrap(x, y), **stats.randomization_test(x, y),
                "paired_t": stats.paired_t(x, y)}

    # target: cascade +15% nDCG@10 over BM25
    t = test("cascade_mmr", "bm25")
    rel = means["cascade_mmr"]["ndcg@10"] / means["bm25"]["ndcg@10"] - 1 if means["bm25"]["ndcg@10"] else None
    res["cascade_vs_bm25"] = {**t, "relative_gain": rel, "target_plus15_met": bool(rel is not None and rel >= 0.15
                                                                                     and t["p"] < 0.05)}
    # family B: ablations vs the full model (Holm)
    fam = {name: test("ltr_full", name) for name in ("ablate_no_graph", "ablate_no_rrf", "ablate_no_cross_encoder")}
    for name, p in stats.holm({k: v["p"] for k, v in fam.items()}).items():
        fam[name]["p_holm"] = p
    res["ablations_vs_full"] = fam
    # D4 gates: each modern component vs its classical counterpart
    res["modern_vs_classical"] = {"dense_vs_bm25": test("dense", "bm25"), "rrf_vs_bm25": test("rrf", "bm25"),
                                  "ltr_full_vs_classical": test("ltr_full", "classical"),
                                  "cross_encoder_vs_bm25": test("cross_encoder", "bm25")}
    res["ranx_fisher"] = str(compare(qrels, list(runs.values()), metrics=["ndcg@10"], stat_test="fisher",
                                     max_p=0.05, random_seed=573))
    return res


def markdown(res: dict) -> str:
    lines = [f"# Ranking evaluation (labels: {res['labels']}, {res['n_profiles']} profiles)", "",
             "| System | nDCG@10 | MRR@10 | P@5 |", "|---|---|---|---|"]
    for name, m in sorted(res["means"].items(), key=lambda x: -x[1]["ndcg@10"]):
        lines.append(f"| {name} | {m['ndcg@10']:.3f} | {m['mrr@10']:.3f} | {m['precision@5']:.3f} |")
    c = res["cascade_vs_bm25"]
    lines += ["", f"Cascade vs BM25: Δ nDCG@10 = {c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}], "
              f"relative {100 * (c['relative_gain'] or 0):+.1f}%, p = {c['p']:.4f}, d_z = {c['paired_t']['cohen_dz']:.2f}; "
              f"+15% target {'met' if c['target_plus15_met'] else 'not met'}.", "",
              "| Ablation (vs full LambdaMART) | Δ nDCG@10 [95% CI] | p (Holm) |", "|---|---|---|"]
    for name, v in res["ablations_vs_full"].items():
        lines.append(f"| {name} | {v['mean_diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] | {v['p_holm']:.4f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="label")
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    pool = pd.read_csv(POOL, dtype={"label": str, "llm_label": str})
    pool["label"] = pd.to_numeric(pool[a.labels], errors="coerce")
    judged = pool.dropna(subset=["label"])
    complete = judged.groupby("profile_id")["label"].size()
    keep = complete[complete == complete.max()].index
    judged = judged[judged["profile_id"].isin(keep)]
    print(f"{len(keep)} fully judged profiles ({len(judged)} pairs) using column {a.labels!r}")
    df = build_features(judged, use_graph=not a.no_graph)
    sys = system_scores(df, df.attrs["dense"])
    res = analyze(df, sys, a.labels)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "" if a.labels == "label" else f"_{a.labels}"
    (RESULTS / f"ranking_eval{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    (RESULTS / f"ranking_eval{tag}.md").write_text(markdown(res))
    df.drop(columns=["_emb_row"]).to_parquet(Path("data/processed/ranking_features.parquet"))
    if a.labels == "label":   # the live cascade uses models trained on all human-judged profiles
        save_models(fit(df, FEATURE_SETS["ltr_full"]), fit(df, ALL_FEATURES))
    print(markdown(res))


if __name__ == "__main__":
    main()
