"""LambdaMART (LightGBM lambdarank) over lexical, dense and graph features (R4 / RQ4).

Trained on the judged (profile, job) pairs with cross-validation grouped by profile, so a profile's
scores always come from a model that never saw that profile.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold

from ranking.cascade import SENIORITY_ORDER, minmax
from ranking.graph_features import FEATURES as GRAPH_FEATURES

LEXICAL_FEATURES = ["f_bm25", "f_tfidf", "f_jaccard", "f_dense", "f_rrf", "f_family_match", "f_seniority_gap"]
ALL_FEATURES = LEXICAL_FEATURES + GRAPH_FEATURES
PARAMS = dict(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=573, verbose=-1)


def pair_features(q, corpus, s: dict, idx: np.ndarray, graph=None) -> pd.DataFrame:
    """One row per candidate index in idx. Scores are min-max normalized within the query's pool."""
    meta = corpus.meta.iloc[idx]
    rows = pd.DataFrame({"query_id": q.doc_id, "doc_id": meta["doc_id"].to_numpy()})
    for name in ("bm25", "tfidf", "jaccard", "dense", "rrf"):
        rows[f"f_{name}"] = minmax(s[name][idx]) if name in s else np.nan
    rows["f_family_match"] = (meta["family"].to_numpy() == q.family).astype(float)
    sen = meta["seniority"] if "seniority" in meta else pd.Series(["Unknown"] * len(meta))
    qs = SENIORITY_ORDER.get(q.seniority or "", np.nan)
    rows["f_seniority_gap"] = [abs(SENIORITY_ORDER.get(x, np.nan) - qs) for x in sen]
    for f in GRAPH_FEATURES:
        rows[f] = np.nan
    if graph is not None:
        pairs = [(q.doc_id, d) if corpus.kind == "jobs" else (d, q.doc_id) for d in rows["doc_id"]]
        g = pd.DataFrame([graph.features(c, j) for c, j in pairs])
        rows[GRAPH_FEATURES] = g[GRAPH_FEATURES].to_numpy()
    return rows


def cross_validated_scores(df: pd.DataFrame, features: list[str], folds: int = 5) -> tuple[np.ndarray, pd.DataFrame]:
    """df needs query_id, label (0/1/2) and features. Returns (out-of-fold scores, df sorted to match)."""
    df = df.sort_values("query_id", kind="stable").reset_index(drop=True)
    out = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=min(folds, df["query_id"].nunique())).split(df, groups=df["query_id"]):
        train = df.iloc[tr]
        model = LGBMRanker(**PARAMS)
        model.fit(train[features], train["label"].astype(int), group=train.groupby("query_id", sort=False).size())
        out[te] = model.predict(df.iloc[te][features])
    return out, df


def fit(df: pd.DataFrame, features: list[str]) -> LGBMRanker:
    df = df.sort_values("query_id", kind="stable")
    model = LGBMRanker(**PARAMS)
    model.fit(df[features], df["label"].astype(int), group=df.groupby("query_id", sort=False).size())
    return model
