"""A searchable corpus over the cached Adzuna postings, so the live cascade (ranking/live.py), explanations,
skill gap and outreach work on live jobs exactly as on the Kaggle index.

It mirrors ranking/index.py at a small scale: CRF + classical-normalizer skills, the job-family classifier, the
Kaggle index's fitted TF-IDF vectorizer (so IDF comes from 20K postings, not ~50), BM25 over the live postings
and MiniLM embeddings. Rebuilt in memory whenever the cache gains postings (seconds for a few hundred).
"""

import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from ingest import adzuna
from ranking.cascade import Corpus, Query, bi_encoder
from ranking.index import INDEX, crf_skills, tokens

_CACHE: dict = {}


class LiveCorpus(Corpus):
    def __init__(self, df):
        self.kind = "jobs"
        meta = df.drop(columns=["text"]).reset_index(drop=True)
        text = (df["title"] + " " + df["text"]).tolist()
        fam_vec, fam_clf = pickle.load(open(INDEX / "family_clf.pkl", "rb"))
        meta["family"] = fam_clf.predict(fam_vec.transform(text)) if len(df) else []
        meta["cluster"] = np.arange(len(meta))
        meta["skills"] = [set(s) for s in crf_skills(df)] if len(df) else []
        self.meta = meta
        self.text = df["text"].to_numpy()
        self.vec, _ = pickle.load(open(INDEX / "jobs" / "tfidf.pkl", "rb"))
        self.X = self.vec.transform(text)
        self.bm25 = BM25Okapi([tokens(t) for t in text])
        self.dense = bi_encoder().encode([t[:3000] for t in text], normalize_embeddings=True,
                                         show_progress_bar=False).astype(np.float32)
        self.pos = {doc: i for i, doc in enumerate(meta["doc_id"])}

    def filter_mask(self, q: Query) -> np.ndarray:
        """The usual filters, plus prefs["doc_ids"]: rank only the postings of the current search."""
        m = super().filter_mask(q)
        ids = (q.prefs or {}).get("doc_ids")
        if ids is not None:
            m &= self.meta["doc_id"].isin(ids).to_numpy()
        return m


def corpus() -> LiveCorpus:
    """The corpus over every cached posting; rebuilt only when the set of postings changes."""
    df = adzuna.load_all()
    version = tuple(df["doc_id"])
    if _CACHE.get("version") != version:
        _CACHE.update(version=version, corpus=LiveCorpus(df))
    return _CACHE["corpus"]
