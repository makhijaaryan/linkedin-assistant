"""Precompute everything ranking needs over the corpora (R4, R5). Run once:  python -m ranking.index

Job corpus = the 20K working-set postings; candidate corpus = the 4,981 working-set resumes (job -> candidates).
Per corpus it stores, under .cache/index/<corpus>/:
  meta.parquet       doc_id, title, seniority, work_mode, state, family, cluster, crf skill_ids
  tfidf.pkl          fitted TF-IDF vectorizer + matrix (classical)
  bm25.pkl           BM25Okapi over lowercase tokens (classical)
  dense.npy          bi-encoder embeddings (modern, R5) + FAISS index built at load time
Plus a logistic-regression job-family classifier (trained on LinkedIn job-function weak labels, train split)
and K-means clusters for diversity. Skills here come from the CRF + classical normalizer, so the
classical pipeline is classical end to end.
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

INDEX = Path(".cache/index")
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN = re.compile(r"[a-z0-9+#]+")
STOP = set("a an and are as at be by for from has have in is it its of on or our that the their this to we will with "
           "you your who what all can may other such any not".split())
REMOTE = re.compile(r"\bremote\b", re.IGNORECASE)
HYBRID = re.compile(r"\bhybrid\b", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def corpus(kind: str) -> pd.DataFrame:
    if kind == "jobs":
        df = pd.read_parquet("data/processed/jobs.parquet")
        df = df[df["in_working_set"]].reset_index(drop=True)
        df["work_mode"] = np.where(df["remote_allowed"] | df["text"].str.contains(REMOTE), "Remote",
                                   np.where(df["text"].str.contains(HYBRID), "Hybrid", "On-site"))
        df["doc_type"] = "JobPosting"
        return df[["doc_id", "source", "title", "company", "location", "state", "seniority", "work_mode",
                   "industry", "text", "doc_type", "split"]]
    df = pd.read_parquet("data/processed/resumes.parquet")
    df = df[df["in_working_set"]].reset_index(drop=True)
    df["doc_type"] = "CV"
    return df[["doc_id", "source", "category", "text", "doc_type", "split"]]


def crf_skills(df: pd.DataFrame) -> list[list[str]]:
    """CRF skill mentions -> classical normalizer -> skill_ids (free:<name> when unmatched)."""
    from extraction.crf import predict_spans
    from graph.normalize import Normalizer, clean
    mentions = []
    for i, text in enumerate(df["text"]):
        mentions.append(sorted({text[s:e] for s, e, lab, _ in predict_spans(text) if lab == "Skill"}))
        if i % 2000 == 0:
            print(f"  crf skills {i}/{len(df)}")
    uniq = sorted({m for ms in mentions for m in ms})
    norm = dict(zip(uniq, Normalizer("classical").normalize(uniq)))
    return [sorted({norm[m].taxonomy_id or f"free:{clean(m)}" for m in ms}) for ms in mentions]


def train_family_classifier(jobs: pd.DataFrame):
    """TF-IDF + logistic regression: job text -> LinkedIn job function (weak labels, train split only)."""
    weak = pd.read_parquet("data/processed/weak_labels/job_weak_labels.parquet").set_index("doc_id")["skill_categories"]
    tr = jobs[(jobs["split"] == "train") & jobs["doc_id"].isin(weak.index)]
    y = tr["doc_id"].map(lambda d: weak[d][0] if weak[d] is not None and len(weak[d]) else "Other")
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(tr["title"] + " " + tr["title"] + " " + tr["text"])
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(X, y)
    return vec, clf


def build(kind: str, dense: bool = True):
    out = INDEX / kind
    out.mkdir(parents=True, exist_ok=True)
    df = corpus(kind)
    print(kind, len(df), "docs")
    text = (df["title"] + " " + df["text"]) if kind == "jobs" else df["text"]
    vec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(text)
    pickle.dump((vec, X), open(out / "tfidf.pkl", "wb"))
    pickle.dump(BM25Okapi([tokens(t) for t in text]), open(out / "bm25.pkl", "wb"))
    svd = TruncatedSVD(100, random_state=573).fit_transform(X)
    df["cluster"] = MiniBatchKMeans(50, random_state=573, n_init=3).fit_predict(svd)
    if kind == "jobs":
        fam_vec, fam_clf = train_family_classifier(df)
        pickle.dump((fam_vec, fam_clf), open(INDEX / "family_clf.pkl", "wb"))
    fam_vec, fam_clf = pickle.load(open(INDEX / "family_clf.pkl", "rb"))
    df["family"] = fam_clf.predict(fam_vec.transform(text))
    df["skills"] = crf_skills(df)
    df.drop(columns=["text"]).to_parquet(out / "meta.parquet", index=False)
    if dense:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(DENSE_MODEL)
        emb = enc.encode(text.str.slice(0, 3000).tolist(), batch_size=64, normalize_embeddings=True,
                         show_progress_bar=False)
        np.save(out / "dense.npy", emb.astype(np.float32))
    print("built", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*", default=["jobs", "cvs"])
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    for k in a.corpus:
        build(k, not a.no_dense)
