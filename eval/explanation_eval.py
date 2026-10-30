"""R6 / RQ4 explanation faithfulness on 60 explanations (annotation guide, Task D).

  python -m eval.explanation_eval --make    # 60 (profile, job) pairs from the judged pool -> explanations by every
                                            # explainer -> data/processed/annotation/explanations_60.csv (one row per sentence)
  python -m eval.explanation_eval --score   # after filling `supported` (1/0) -> eval/results/explanation_eval.json

Explainers: template baseline, the local explain route, and (public Kaggle profiles only, D2) the proprietary
baseline and a hosted open model. Automatic checks (citation present, cited ids inside the bundle) are reported
next to the human judgment; the target (>= 0.95) is on the human labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.explainer import bundle, llm_explanation, template_explanation
from eval import stats
from ranking.pool import POOL
from router.router import Router

SHEET = Path("data/processed/annotation/explanations_60.csv")
RAW = Path("data/processed/explanations_60.jsonl")
SEED = 573


def pairs(n: int = 60) -> list[tuple[str, str]]:
    """Per profile: the best-judged pooled job, then a second pair for 20 random profiles."""
    pool = pd.read_csv(POOL, dtype=str).fillna("")
    pool["l"] = pd.to_numeric(pool["label"], errors="coerce").fillna(-1)
    out = []
    for pid, g in pool.groupby("profile_id"):
        out.append((pid, g.sort_values("l", ascending=False).iloc[0]["job_id"]))
    rng = np.random.default_rng(SEED)
    for pid in rng.permutation(sorted(pool["profile_id"].unique()))[: n - len(out)]:
        g = pool[pool["profile_id"] == pid].sort_values("l", ascending=False)
        out.append((pid, g.iloc[1]["job_id"]))
    return out[:n]


def make():
    router = Router()
    public = set(router.cfg["public_sources"])
    src = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "source"]).set_index("doc_id")["source"]
    prop = router.route("proprietary_baseline")
    hosted_open = ("groq", "openai/gpt-oss-120b")
    rows, raw = [], []
    for n, (cid, jid) in enumerate(pairs()):
        facts = bundle(cid, jid)
        private = src.get(cid) not in public
        exps = [template_explanation(cid, jid, facts), llm_explanation(router, cid, jid, facts, private=private)]
        if not private:
            for p, m in (prop, hosted_open):
                try:
                    exps.append(llm_explanation(router, cid, jid, facts, private=False, provider=p, model=m))
                except Exception as e:
                    print("  skipped", p, m, type(e).__name__)
        for ex in exps:
            raw.append({**ex.to_dict(), "facts": [f.__dict__ for f in facts]})
            for k, s in enumerate(ex.sentences):
                rows.append({"pair": n, "candidate_id": cid, "job_id": jid, "method": ex.method, "sentence": k,
                             "text": s["text"], "cites": " ".join(s["cites"]),
                             "facts_cited": " || ".join(f.text for f in facts if set(f.edge_ids) & set(s["cites"])),
                             "supported": ""})
        print(f"  explained pair {n + 1}/60")
    with open(RAW, "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    if SHEET.exists():
        raise SystemExit(f"{SHEET} exists; delete it first to regenerate (it may hold your labels)")
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED)   # method order hidden from the judge
    df.to_csv(SHEET, index=False)
    print(f"{len(df)} sentences -> {SHEET}")


def score() -> dict:
    raw = [json.loads(x) for x in open(RAW)]
    out = {"automatic": {}}
    for method, g in pd.DataFrame(raw).groupby("method"):
        sents = [s for ss in g["sentences"] for s in ss]
        rejected = [s for ss in g["rejected"] for s in ss]
        out["automatic"][method] = {"explanations": len(g), "sentences_shown": len(sents),
                                    "sentences_rejected_by_critic": len(rejected),
                                    "all_shown_cited": all(s["cites"] for s in sents),
                                    "empty_explanations": int(sum(len(ss) == 0 for ss in g["sentences"]))}
    df = pd.read_csv(SHEET, dtype=str).fillna("")
    df = df[df["supported"].isin(["0", "1"])]
    if len(df):
        df["ok"] = df["supported"].astype(int)
        out["faithfulness"] = {m: {**stats.wilson(int(g["ok"].sum()), len(g)),
                                   "hallucination_rate": 1 - g["ok"].mean(),
                                   "explanation_level_all_supported": float(g.groupby("pair")["ok"].min().mean())}
                               for m, g in df.groupby("method")}
        # open vs. proprietary on the same pairs: explanation-level "fully faithful", McNemar
        prop = next((m for m in out["faithfulness"] if "gemini" in m), None)
        if prop:
            full = df.groupby(["method", "pair"])["ok"].min().unstack(0)
            out["mcnemar_vs_proprietary"] = {}
            pv = {}
            for m in full.columns:
                if m == prop or m == "template":
                    continue
                both = full[[m, prop]].dropna()
                out["mcnemar_vs_proprietary"][m] = stats.mcnemar(both[m].astype(bool), both[prop].astype(bool))
                pv[m] = out["mcnemar_vs_proprietary"][m]["p"]
            for m, p in stats.holm(pv).items():
                out["mcnemar_vs_proprietary"][m]["p_holm"] = p
    out["target"] = 0.95
    Path("eval/results/explanation_eval.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.make:
        make()
    if a.score:
        print(json.dumps(score(), indent=1, default=float))
