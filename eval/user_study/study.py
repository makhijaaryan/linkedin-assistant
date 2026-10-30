"""Blind A/B user study (Phase 9): template explainer (A) vs. LLM explainer + critic (B), within-subject.

  python -m eval.user_study.study --packet    # -> eval/user_study/packet.html + responses_template.csv
  python -m eval.user_study.study --analyze   # reads eval/user_study/responses.csv -> eval/results/user_study.json

Only public Kaggle CVs/jobs appear in the packet (D2). Each case shows the two explanations in a random,
recorded order labeled "Explanation 1/2", so participants never see which method wrote which.
Analysis (STATS_PLAN): per-participant mean rating per condition; Wilcoxon signed-rank B vs A with
rank-biserial r; one-sample Wilcoxon of B against 4.0 with a bootstrap CI of the mean.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import stats

HERE = Path("eval/user_study")
RAW = Path("data/processed/explanations_60.jsonl")
N_CASES, SEED = 6, 573


def packet():
    raw = [json.loads(x) for x in open(RAW)]
    src = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "source", "text"]).set_index("doc_id")
    jobs = pd.read_parquet("data/processed/jobs.parquet", columns=["doc_id", "title", "company", "text"]).set_index("doc_id")
    by_pair = {}
    for r in raw:
        if src.at[r["candidate_id"], "source"] == "kaggle_resume_dataset":
            by_pair.setdefault((r["candidate_id"], r["job_id"]), {})[r["method"].split(":")[0]] = r
    cases = [(k, v) for k, v in by_pair.items() if "template" in v and "llm" in v
             and v["template"]["sentences"] and v["llm"]["sentences"]]
    rng = np.random.default_rng(SEED)
    cases = [cases[i] for i in rng.permutation(len(cases))[:N_CASES]]
    key, parts = [], ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px;margin:auto'>",
                      "<h1>Job-match explanations: study packet</h1><p>For each case, read the CV and the job, then "
                      "rate each explanation from 1 (not helpful or trustworthy) to 5 (very helpful and trustworthy)."
                      "</p>"]
    for n, ((cid, jid), v) in enumerate(cases):
        order = ["template", "llm"] if rng.random() < 0.5 else ["llm", "template"]
        key += [{"case": n + 1, "position": p + 1, "condition": "A_template" if m == "template" else "B_llm"}
                for p, m in enumerate(order)]
        parts.append(f"<hr><h2>Case {n + 1}: {html.escape(jobs.at[jid, 'title'])} at "
                     f"{html.escape(str(jobs.at[jid, 'company']))}</h2>"
                     f"<details><summary>CV</summary><pre style='white-space:pre-wrap'>{html.escape(src.at[cid, 'text'][:4000])}"
                     f"</pre></details><details><summary>Job</summary><pre style='white-space:pre-wrap'>"
                     f"{html.escape(jobs.at[jid, 'text'][:4000])}</pre></details>")
        for p, m in enumerate(order):
            items = "".join(f"<li>{html.escape(s['text'])}</li>" for s in v[m]["sentences"])
            parts.append(f"<h3>Explanation {p + 1}</h3><ul>{items}</ul><p>Rating (1-5): ____</p>")
    (HERE / "packet.html").write_text("".join(parts) + "</body></html>")
    pd.DataFrame(key).to_csv(HERE / "packet_key.csv", index=False)
    pd.DataFrame(columns=["participant", "case", "position", "rating"]).to_csv(HERE / "responses_template.csv",
                                                                               index=False)
    print(f"{len(cases)} cases -> {HERE / 'packet.html'} (answer key: packet_key.csv; keep it from participants)")


def analyze() -> dict:
    resp = pd.read_csv(HERE / "responses.csv").merge(pd.read_csv(HERE / "packet_key.csv"), on=["case", "position"])
    per = resp.pivot_table(index="participant", columns="condition", values="rating", aggfunc="mean").dropna()
    out = {"participants": int(len(per)), "mean": per.mean().to_dict(),
           "B_vs_A": stats.wilcoxon(per["B_llm"], per["A_template"]),
           "B_vs_4": stats.one_sample_wilcoxon(per["B_llm"], 4.0), "target_mean": 4.0}
    out["target_met"] = bool(out["B_vs_4"]["ci_low"] >= 4.0)
    Path("eval/results/user_study.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.packet:
        packet()
    if a.analyze:
        print(json.dumps(analyze(), indent=1, default=float))
