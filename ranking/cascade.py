"""The ranking cascade (R4, R5): one class serves both directions (CV -> jobs, job -> candidates).

Classical system (M2 gate):  filters -> BM25 / TF-IDF / skill Jaccard (top 200 each) -> mean of
                             min-max scores + job-family match -> K-means round-robin for a diverse top 10.
Modern cascade:              filters -> RRF(BM25, TF-IDF, Jaccard, dense) top 200 -> LambdaMART (lexical +
                             dense + graph features) top 20 -> cross-encoder -> MMR diverse top 10.

Every scorer can rank either the whole corpus or a given candidate pool (the pooled-judgment setting, D23).
"""

import pickle
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ranking.index import DENSE_MODEL, INDEX, corpus, tokens

SENIORITY_ORDER = {"Internship": 0, "Entry level": 1, "Associate": 2, "Mid-Senior level": 3, "Director": 4,
                   "Executive": 5}
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LEXICAL = ["bm25", "tfidf", "jaccard"]


@dataclass
class Query:
    doc_id: str
    text: str
    skills: set = field(default_factory=set)       # classical (CRF) skill ids
    family: str | None = None
    seniority: str | None = None
    prefs: dict = field(default_factory=dict)      # {"states": [...], "work_modes": [...], "seniority": [...]}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def rrf(score_lists: list[np.ndarray], k: int = 60, depth: int = 1000) -> np.ndarray:
    """Reciprocal rank fusion over full-corpus score arrays (only the top `depth` of each list count)."""
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        order = np.argsort(-s)[:depth]
        fused[order] += 1.0 / (k + np.arange(1, len(order) + 1))
    return fused


@lru_cache(maxsize=1)
def bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DENSE_MODEL)


@lru_cache(maxsize=1)
def cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER, max_length=512)


class Corpus:
    """One searchable side: 'jobs' (for a CV query) or 'cvs' (for a job query)."""

    def __init__(self, kind: str):
        d = INDEX / kind
        self.kind = kind
        self.meta = pd.read_parquet(d / "meta.parquet")
        self.meta["skills"] = self.meta["skills"].map(set)
        self.text = corpus(kind).set_index("doc_id").loc[self.meta["doc_id"], "text"].to_numpy()
        self.vec, self.X = pickle.load(open(d / "tfidf.pkl", "rb"))
        self.bm25 = pickle.load(open(d / "bm25.pkl", "rb"))
        self.dense = np.load(d / "dense.npy") if (d / "dense.npy").exists() else None
        self.pos = {doc: i for i, doc in enumerate(self.meta["doc_id"])}

    # --- first-stage scorers: full-corpus arrays -------------------------------------------------------
    def score_bm25(self, q: Query) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(list(dict.fromkeys(tokens(q.text)))))

    def score_tfidf(self, q: Query) -> np.ndarray:
        return (self.vec.transform([q.text]) @ self.X.T).toarray()[0]

    def score_jaccard(self, q: Query) -> np.ndarray:
        if not q.skills:
            return np.zeros(len(self.meta))
        return np.array([len(q.skills & s) / len(q.skills | s) if s else 0.0 for s in self.meta["skills"]])

    def score_dense(self, q: Query) -> np.ndarray:
        e = bi_encoder().encode([q.text[:3000]], normalize_embeddings=True, show_progress_bar=False)[0]
        with np.errstate(all="ignore"):
            return self.dense @ e

    def first_stage(self, q: Query) -> dict[str, np.ndarray]:
        s = {"bm25": self.score_bm25(q), "tfidf": self.score_tfidf(q), "jaccard": self.score_jaccard(q)}
        if self.dense is not None:
            s["dense"] = self.score_dense(q)
            s["rrf"] = rrf([s["bm25"], s["tfidf"], s["jaccard"], s["dense"]])
        return s

    def filter_mask(self, q: Query) -> np.ndarray:
        m = np.ones(len(self.meta), bool)
        p = q.prefs or {}
        if self.kind == "jobs":
            if p.get("states"):
                m &= self.meta["state"].isin(p["states"]).to_numpy() | (self.meta["work_mode"] == "Remote").to_numpy()
            if p.get("work_modes"):
                m &= self.meta["work_mode"].isin(p["work_modes"]).to_numpy()
            if p.get("seniority"):
                m &= self.meta["seniority"].isin(p["seniority"] + ["Unknown"]).to_numpy()
        return m

    # --- systems ------------------------------------------------------------------------------------------
    def classical(self, q: Query, s: dict, idx: np.ndarray, k: int = 10, diversify: bool = True) -> list[int]:
        """Classical final ranking over candidate indices idx."""
        score = np.mean([minmax(s[n][idx]) for n in LEXICAL], axis=0)
        if q.family is not None:
            score = score + 0.1 * (self.meta["family"].to_numpy()[idx] == q.family)
        order = idx[np.argsort(-score)]
        if not diversify:
            return order[:k].tolist()
        clusters = self.meta["cluster"].to_numpy()
        picked, seen = [], set()
        for i in order:                            # best job from each K-means cluster first ...
            if clusters[i] not in seen:
                picked.append(i)
                seen.add(clusters[i])
            if len(picked) == k:
                return picked
        rest = [i for i in order if i not in set(picked)]   # ... then fill by score
        return (picked + rest)[:k]

    def candidates(self, q: Query, s: dict, depth: int = 200, by: list[str] | None = None) -> np.ndarray:
        """Union of the top `depth` of each list in `by` (after filters)."""
        mask = self.filter_mask(q)
        idx = set()
        for name in by or LEXICAL:
            sc = np.where(mask, s[name], -np.inf)
            idx.update(np.argsort(-sc)[:depth].tolist())
        return np.array(sorted(i for i in idx if mask[i]))


def mmr(rel: np.ndarray, emb: np.ndarray, k: int = 10, lam: float = 0.7) -> list[int]:
    """Maximal marginal relevance over positions 0..n-1: lam * relevance - (1 - lam) * max similarity."""
    rel = minmax(rel)
    with np.errstate(all="ignore"):
        sim = emb @ emb.T
    picked, rest = [], list(range(len(rel)))
    while rest and len(picked) < k:
        best = max(rest, key=lambda i: lam * rel[i] - (1 - lam) * (max(sim[i, j] for j in picked) if picked else 0))
        picked.append(best)
        rest.remove(best)
    return picked


def cross_scores(query_text: str, docs: list[str]) -> np.ndarray:
    pairs = [(query_text[:1500], d[:2000]) for d in docs]
    return np.asarray(cross_encoder().predict(pairs, batch_size=16, show_progress_bar=False))


def cv_query(doc_id: str, cvs: "Corpus") -> Query:
    """A ranking query for a CV already in the candidate corpus."""
    i = cvs.pos[doc_id]
    r = cvs.meta.iloc[i]
    return Query(doc_id, cvs.text[i], set(r["skills"]), r["family"])


def new_cv_query(doc_id: str, text: str, prefs: dict | None = None) -> Query:
    """A query for an uploaded CV: classical skills + family computed on the fly (all local)."""
    from extraction.crf import predict_spans
    from graph.normalize import Normalizer, clean
    names = sorted({text[s:e] for s, e, lab, _ in predict_spans(text) if lab == "Skill"})
    skills = {m.taxonomy_id or f"free:{clean(n)}" for n, m in zip(names, Normalizer("classical").normalize(names))}
    fam_vec, fam_clf = pickle.load(open(INDEX / "family_clf.pkl", "rb"))
    return Query(doc_id, text, skills, fam_clf.predict(fam_vec.transform([text]))[0], prefs=prefs or {})


def job_query(doc_id: str, jobs: "Corpus") -> Query:
    i = jobs.pos[doc_id]
    r = jobs.meta.iloc[i]
    return Query(doc_id, f"{r['title']} {jobs.text[i]}", set(r["skills"]), r["family"], r["seniority"])

