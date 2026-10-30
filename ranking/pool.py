"""Relevance pool for the 40 ranking profiles (Task C of the annotation guide, D23).

  python -m ranking.pool --make        # pool 20 jobs/profile by interleaving the first-stage systems
                                       # -> data/processed/annotation/relevance_pool.csv + graph_pool.txt
  streamlit run ui/judge.py            # judge pairs 0/1/2 (blind to system), saved into the same CSV
  python -m ranking.pool --llm-judge   # OPTIONAL second opinion in column llm_label (never overwrites `label`)

Kaggle CVs/jobs are public and may go to the judge_relevance route; resume_corpus profiles are judged
locally (D2).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ranking.cascade import Corpus, cv_query

POOL = Path("data/processed/annotation/relevance_pool.csv")
GRAPH_POOL = Path("data/processed/graph_pool.txt")
POOL_SYSTEMS = ["rrf", "bm25", "dense", "jaccard", "tfidf", "classical"]
PER_PROFILE = 20


def profile_ids() -> list[str]:
    return json.loads(Path("eval/frozen/test_sets.json").read_text())["ranking_profiles"]


def make():
    jobs, cvs = Corpus("jobs"), Corpus("cvs")
    rows = []
    for n, pid in enumerate(profile_ids()):
        q = cv_query(pid, cvs)
        s = jobs.first_stage(q)
        idx = jobs.candidates(q, s, depth=200, by=["bm25", "tfidf", "jaccard", "dense"])
        lists = {name: idx[np.argsort(-s[name][idx])].tolist() for name in POOL_SYSTEMS if name in s}
        lists["classical"] = jobs.classical(q, s, idx, k=PER_PROFILE)
        pool, depth = [], 0
        while len(pool) < PER_PROFILE and depth < 200:
            for name in POOL_SYSTEMS:
                if depth < len(lists[name]) and lists[name][depth] not in pool and len(pool) < PER_PROFILE:
                    pool.append(lists[name][depth])
            depth += 1
        rng = np.random.default_rng(573 + n)
        for i in rng.permutation(pool):            # shuffled: the judge never sees system order
            r = jobs.meta.iloc[i]
            rows.append({"profile_id": pid, "job_id": r["doc_id"], "job_title": r["title"], "label": "",
                         "llm_label": "", "notes": ""})
        print(f"  pooled {pid} ({n + 1}/40)")
    if POOL.exists():
        raise SystemExit(f"{POOL} exists; delete it first to regenerate (it may hold your labels)")
    pd.DataFrame(rows).to_csv(POOL, index=False)
    GRAPH_POOL.write_text("\n".join(sorted({r["job_id"] for r in rows})))
    print(f"wrote {len(rows)} pairs -> {POOL}; {len(set(r['job_id'] for r in rows))} jobs -> {GRAPH_POOL}")


JUDGE_PROMPT = """You judge how well a candidate fits a job, using only qualifications stated in the two texts.
2 = strong fit: meets most required qualifications. 1 = partial fit: relevant field, clear gaps.
0 = not a fit: different field or misses most required qualifications. Ignore location and salary.
CV:
\"\"\"{cv}\"\"\"
JOB:
\"\"\"{job}\"\"\"
Reply JSON {{"label": 0|1|2, "reason": "<one sentence>"}}."""


def llm_judge(limit: int = 0):
    from router.router import Router
    router = Router()
    public = set(router.cfg["public_sources"])
    pool = pd.read_csv(POOL, dtype=str).fillna("")
    jobs = pd.read_parquet("data/processed/jobs.parquet", columns=["doc_id", "title", "text", "source"]).set_index("doc_id")
    cvs = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "text", "source"]).set_index("doc_id")
    todo = pool[pool["llm_label"] == ""].index[: limit or None]
    for n, i in enumerate(todo):
        p, j = pool.at[i, "profile_id"], pool.at[i, "job_id"]
        private = cvs.at[p, "source"] not in public or jobs.at[j, "source"] not in public
        task = "judge_relevance"
        kw = {"provider": "ollama", "model": router.cfg["routes"]["critic"]["model"]} if private else {}
        msg = JUDGE_PROMPT.format(cv=cvs.at[p, "text"][:3500], job=(jobs.at[j, "title"] + "\n" + jobs.at[j, "text"])[:3500])
        try:
            out = json.loads(router.chat(task, [{"role": "user", "content": msg}], private=private, **kw).text)
            pool.at[i, "llm_label"] = str(int(out["label"]))
        except Exception as e:
            print("  judge failed", p, j, type(e).__name__)
        if n % 20 == 0:
            pool.to_csv(POOL, index=False)
            print(f"  judged {n}/{len(todo)}")
    pool.to_csv(POOL, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.make:
        make()
    if a.llm_judge:
        llm_judge(a.limit)
