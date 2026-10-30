"""The live cascade used by the agents and UI (both directions).

CV -> jobs:  filters -> RRF top 200 -> LambdaMART without cross-encoder (top 20) -> cross-encoder ->
             LambdaMART with cross-encoder -> MMR diverse top k.
Before relevance judgments exist (no trained LambdaMART), the learned stages fall back to RRF, and the
final score to 0.5 * RRF + 0.5 * cross-encoder; `stages` in the output says which path ran.
Job -> candidates uses the same first stage + cross-encoder + MMR (no LambdaMART: it is trained on CV -> job pairs).
"""

import pickle
import time
from pathlib import Path

import numpy as np

from ranking.cascade import Corpus, Query, cross_scores, minmax, mmr
from ranking.ltr import ALL_FEATURES, pair_features

MODELS = Path(".cache/models")


def _load(name):
    p = MODELS / name
    return pickle.load(open(p, "rb")) if p.exists() else None


def save_models(full, no_ce):
    MODELS.mkdir(parents=True, exist_ok=True)
    pickle.dump(full, open(MODELS / "ltr_full.pkl", "wb"))
    pickle.dump(no_ce, open(MODELS / "ltr_no_ce.pkl", "wb"))


def rank(q: Query, corpus: Corpus, graph=None, k: int = 10, depth: int = 200, rerank: int = 20,
         lam: float = 0.7) -> dict:
    t0 = time.perf_counter()
    timing, stages = {}, []
    s = corpus.first_stage(q)
    mask = corpus.filter_mask(q)
    fused = np.where(mask, s.get("rrf", s["bm25"]), -np.inf)
    idx = np.argsort(-fused)[:depth]
    idx = idx[np.isfinite(fused[idx])]
    timing["first_stage_s"] = time.perf_counter() - t0
    stages.append(f"filters+RRF top {len(idx)}")
    feats = pair_features(q, corpus, s, idx, graph if corpus.kind == "jobs" else None)
    no_ce, full = (_load("ltr_no_ce.pkl"), _load("ltr_full.pkl")) if corpus.kind == "jobs" else (None, None)
    pre = no_ce.predict(feats[ALL_FEATURES]) if no_ce is not None else feats["f_rrf"].to_numpy()
    stages.append("LambdaMART (no CE)" if no_ce is not None else "RRF order (no trained LambdaMART yet)")
    top = np.argsort(-pre)[:rerank]
    t1 = time.perf_counter()
    feats = feats.iloc[top].reset_index(drop=True)
    texts = [(corpus.meta.iloc[i].get("title", "") or "") + "\n" + corpus.text[i] for i in idx[top]]
    feats["f_ce"] = minmax(cross_scores(q.text, texts))
    timing["cross_encoder_s"] = time.perf_counter() - t1
    if full is not None:
        final = full.predict(feats[ALL_FEATURES + ["f_ce"]])
        stages.append("cross-encoder -> LambdaMART (full)")
    else:
        final = 0.5 * minmax(feats["f_rrf"].to_numpy()) + 0.5 * feats["f_ce"].to_numpy()
        stages.append("cross-encoder (0.5 RRF + 0.5 CE)")
    picked = mmr(final, corpus.dense[idx[top]], k=k, lam=lam)
    stages.append(f"MMR lambda={lam} top {len(picked)}")
    timing["total_s"] = time.perf_counter() - t0
    results = []
    for rank_, p in enumerate(picked):
        i = idx[top][p]
        m = corpus.meta.iloc[i]
        row = {"rank": rank_ + 1, "doc_id": m["doc_id"], "score": float(final[p]), "ce": float(feats.at[p, "f_ce"]),
               "features": {c: (None if not np.isfinite(v) else float(v))
                            for c, v in feats.iloc[p].items() if c.startswith(("f_", "g_")) and isinstance(v, float)}}
        for c in ("title", "company", "location", "work_mode", "seniority", "category"):
            if c in m:
                row[c] = m[c]
        results.append(row)
    return {"results": results, "stages": stages, "timing": {k_: round(v, 3) for k_, v in timing.items()}}
