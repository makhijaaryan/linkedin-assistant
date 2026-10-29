"""Duplicate detection: exact (title, company, text) keys plus MinHash LSH near-duplicates.

Returns a cluster id per document; documents in one cluster are duplicates and
must land in the same split. Also samples declared-duplicate pairs for hand-checking.
"""

import hashlib

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from ingest.text import normalize_key


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _shingles(text: str, k: int) -> set[bytes]:
    words = normalize_key(text).split()
    if len(words) < k:
        return {" ".join(words).encode()}
    return {" ".join(words[i:i + k]).encode() for i in range(len(words) - k + 1)}


def signatures(texts, num_perm: int, shingle_words: int, seed: int) -> list[MinHash]:
    return MinHash.bulk((_shingles(t, shingle_words) for t in texts), num_perm=num_perm, seed=seed)


def find_duplicates(df: pd.DataFrame, key_cols: list[str], sigs: list[MinHash],
                    threshold: float) -> tuple[pd.Series, pd.DataFrame]:
    """Duplicates = same normalized key columns (e.g. title + company) AND text Jaccard >= threshold.

    Requiring the key match (proposal: "title, company, MinHash of text") stops staffing-firm
    boilerplate with different job titles from collapsing into one cluster.
    Returns (cluster_id per row, DataFrame of near-duplicate pairs with estimated Jaccard).
    """
    n = len(df)
    uf = UnionFind(n)
    keys = df[key_cols].astype(str).agg(" | ".join, axis=1).map(normalize_key).to_numpy()

    # 1) exact duplicates: same key + same normalized text
    text_hash = df["text"].map(lambda t: hashlib.sha1(normalize_key(t).encode()).hexdigest()).to_numpy()
    first = {}
    for i, k in enumerate(zip(keys, text_hash)):
        uf.union(first.setdefault(k, i), i)

    # 2) near duplicates: LSH candidates with the same key and estimated Jaccard >= threshold
    lsh = MinHashLSH(threshold=threshold, num_perm=len(sigs[0].hashvalues))
    pairs = []
    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            if keys[j] == keys[i]:
                sim = m.jaccard(sigs[j])
                if sim >= threshold:
                    uf.union(i, j)
                    pairs.append((j, i, sim))
        lsh.insert(i, m)

    roots = np.array([uf.find(i) for i in range(n)])
    cluster = pd.Series(df["doc_id"].values[roots], index=df.index, name="cluster_id")
    return cluster, pd.DataFrame(pairs, columns=["i", "j", "est_jaccard"])


def handcheck_sample(df: pd.DataFrame, pairs: pd.DataFrame, k: int, seed: int,
                     title_col: str = None, company_col: str = None) -> pd.DataFrame:
    """Sample k declared-duplicate pairs, spread across similarity bands, for manual precision checking."""
    if pairs.empty:
        return pairs
    band = pd.cut(pairs["est_jaccard"], [0, 0.95, 0.99, 1.01], labels=["0.90-0.95", "0.95-0.99", "0.99-1.0"])
    pairs = pairs.assign(band=band.astype(str)).sample(frac=1, random_state=seed)
    sample = pairs.assign(_r=pairs.groupby("band").cumcount()).sort_values("_r", kind="stable").head(k)
    out = pd.DataFrame({
        "doc_a": df["doc_id"].values[sample["i"]], "doc_b": df["doc_id"].values[sample["j"]],
        "est_jaccard": sample["est_jaccard"].round(3).values, "band": sample["band"].values,
    })
    for side, col in (("a", "i"), ("b", "j")):
        if title_col:
            out[f"title_{side}"] = df[title_col].values[sample[col]]
        if company_col:
            out[f"company_{side}"] = df[company_col].values[sample[col]]
        out[f"text_{side}"] = df["text"].str[:400].values[sample[col]]
    out["is_duplicate"] = ""  # annotator fills 1 (same posting/CV) or 0 (different)
    return out
