"""Working set, frozen train/dev/test splits, test-set selection and leakage checks.

Splits are assigned per duplicate cluster, stratified, so near-duplicates never
straddle train and test.
"""

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH


def top_k_or_other(s: pd.Series, k: int) -> pd.Series:
    keep = s.value_counts().index[:k]
    return s.where(s.isin(keep), "Other")


def stratified_sample(df: pd.DataFrame, strata: pd.Series, n: int, seed: int, floor: int = 5) -> pd.DataFrame:
    """Proportional allocation with a small floor per stratum, trimmed back to exactly n."""
    if n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    picks = []
    for _, g in df.groupby(strata.loc[df.index]):
        take = min(len(g), max(floor, int(round(len(g) * frac))))
        picks.append(g.sample(take, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks)
    return out.sample(n, random_state=seed) if len(out) > n else out


def assign_splits(df: pd.DataFrame, strata: pd.Series, ratios: dict, seed: int) -> pd.Series:
    """Assign each duplicate cluster to train/dev/test within its stratum; rows inherit."""
    rng = np.random.default_rng(seed)
    clusters = pd.DataFrame({"cluster_id": df["cluster_id"], "stratum": strata}).drop_duplicates("cluster_id")
    split_of = {}
    for _, g in clusters.groupby("stratum"):
        ids = g["cluster_id"].to_numpy()
        rng.shuffle(ids)
        n_train = int(round(len(ids) * ratios["train"]))
        n_dev = int(round(len(ids) * ratios["dev"]))
        for cid in ids[:n_train]:
            split_of[cid] = "train"
        for cid in ids[n_train:n_train + n_dev]:
            split_of[cid] = "dev"
        for cid in ids[n_train + n_dev:]:
            split_of[cid] = "test"
    return df["cluster_id"].map(split_of)


def round_robin(df: pd.DataFrame, group: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """Pick n rows cycling through groups, so small groups are represented."""
    shuffled = df.sample(frac=1, random_state=seed)
    order = shuffled.groupby(group.loc[shuffled.index]).cumcount()
    return shuffled.assign(_r=order).sort_values(["_r"], kind="stable").head(n).drop(columns="_r")


def leakage_report(df: pd.DataFrame, sigs: list[MinHash], threshold: float = 0.8) -> tuple[dict, set]:
    """Check doc- and cluster-level split overlap, and find test docs with a near-identical train doc.

    Returns (report, doc_ids of test docs that have a train neighbor at Jaccard >= threshold);
    those docs stay in the test split but are excluded from the gold test sets.
    """
    report = {
        "docs_in_multiple_splits": int(df.groupby("doc_id")["split"].nunique().gt(1).sum()),
        "clusters_spanning_splits": int(df.groupby("cluster_id")["split"].nunique().gt(1).sum()),
    }
    split = df["split"].to_numpy()
    lsh = MinHashLSH(threshold=threshold, num_perm=len(sigs[0].hashvalues))
    for i in np.flatnonzero(split == "train"):
        lsh.insert(int(i), sigs[i])
    doc_ids = df["doc_id"].to_numpy()
    flagged = {doc_ids[i] for i in np.flatnonzero(split == "test")
               if any(sigs[i].jaccard(sigs[j]) >= threshold for j in lsh.query(sigs[i]))}
    report[f"test_docs_with_train_neighbor_jaccard>={threshold}"] = len(flagged)
    report["test_docs"] = int((split == "test").sum())
    return report, flagged
