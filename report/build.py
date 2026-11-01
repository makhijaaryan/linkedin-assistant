"""Fill report/REPORT.md's <!-- BLOCK --> markers from eval/results and the annotation files -> report/REPORT_FINAL.md.

  python -m eval.report && python report/build.py

Every number in the final report comes from a result file; a missing result is printed as "pending", never estimated.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R, ANN = ROOT / "eval/results", ROOT / "data/processed/annotation"


def load(*names):
    for n in names:
        p = R / n
        if p.exists():
            return json.loads(p.read_text())
    return None


def pending(what: str) -> str:
    return f"*Pending: {what}.*"


def data_block() -> str:
    p = load("phase1_report.json")
    if not p:
        return pending("phase 1 report")
    l, d, w, lk = p["load"], p["dedupe"], p["working_set"], p["leakage"]
    return (f"Kaggle LinkedIn Job Postings 2023–24 ({l['jobs_with_text']:,} postings with text), the Kaggle Resume "
            f"Dataset ({l['resumes_kaggle']:,} CVs, 24 categories) and the MIT-licensed `resume_corpus` "
            f"({l['resumes_corpus']:,} CVs) are cleaned, contact-masked ({l['jobs_contacts_masked']:,} postings and "
            f"{l['resumes_contacts_masked']:,} CVs had contacts masked) and deduplicated with MinHash LSH on title + "
            f"company + text Jaccard ≥ 0.9: {d['job_duplicates_removed']:,} duplicate postings and "
            f"{d['resume_duplicates_removed']:,} duplicate CVs removed. Splits are by duplicate cluster "
            f"(train/dev/test 80/10/10), with {lk['jobs']['clusters_spanning_splits'] + lk['resumes']['clusters_spanning_splits']} "
            f"clusters crossing splits; test documents with a near neighbour in train (Jaccard ≥ 0.8) are excluded from "
            f"the gold sets. The working set is {w['jobs']:,} postings and {w['resumes']:,} CVs; skills are normalized to "
            f"O\\*NET 31.0 (Lightcast access required a sales contact, D14).")


def summary_block() -> str:
    s = R / "SUMMARY.md"
    if not s.exists():
        return pending("`python -m eval.report`")
    txt = s.read_text().split("\n## ")[0]
    return "\n".join(l for l in txt.splitlines() if l.startswith("|"))


def extraction_block() -> str:
    md = R / "extraction_benchmark.md"
    return md.read_text().split("\n", 2)[2] if md.exists() else pending("extraction benchmark")


def cost_block() -> str:
    ext = load("extraction_benchmark.json")
    if not ext:
        return ""
    rows = []
    for name, r in ext["results"].items():
        e = r.get("efficiency", {})
        rows.append({"extractor": name, "micro-F1": round(100 * r["micro_f1_mean_sd"][0], 1),
                     "p50 s/doc": round((e.get("p50_s") or {}).get("estimate", np.nan), 1),
                     "p95 s/doc": round((e.get("p95_s") or {}).get("estimate", np.nan), 1),
                     "$ / 1K docs": e.get("usd_per_1k_docs")})
    df = pd.DataFrame(rows).sort_values("micro-F1", ascending=False)
    return ("**Quality vs. cost and latency per document** (list or gateway-indicative prices; local models cost 0, "
            "latency is end-to-end per document including all chunks):\n\n" + df.to_markdown(index=False))


def ranking_block() -> str:
    md = R / "ranking_eval.md"
    md = md if md.exists() else R / "ranking_eval_llm_label.md"
    return md.read_text().split("\n", 1)[1] if md.exists() else pending("ranking evaluation")


def spo_block() -> str:
    s = load("spo_ablation.json")
    if not s:
        return pending("SPO extraction and `eval.spo_eval`")
    out = (f"On {s['docs']} documents, the free-form SPO graph has a duplicate-node rate of {s['spo']['rate']:.3f} vs. "
           f"{s['ontology']['rate']:.3f} for the ontology graph (two-proportion z-test p = {s['dup_rate_test']['p']:.4g}).")
    if "ndcg@10" in s:
        t = s["ndcg_test_ontology_vs_spo"]
        out += (f" With each graph's features in LambdaMART, nDCG@10 is {s['ndcg@10']['ontology_graph']:.3f} (ontology) vs. "
                f"{s['ndcg@10']['spo_graph']:.3f} (SPO); Δ = {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}], "
                f"Wilcoxon p = {t['p']:.4g}.")
    return out


def graph_block() -> str:
    g = load("graph_eval.json", "graph_eval_llm.json")
    if not g:
        return pending("mention and edge labels")
    f = lambda d: f"{d['p_hat']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
    out = []
    if "normalization_modern" in g:
        out.append(f"Top-1 skill normalization on {g['normalization_modern'].get('n', '')} mentions: modern "
                   f"{f(g['normalization_modern'])}, classical {f(g['normalization_classical'])} (McNemar p = "
                   f"{g['modern_vs_classical_mcnemar']['p']:.4g}).")
    if "edge_precision" in g:
        out.append(f"Edge precision on sampled extracted edges: {f(g['edge_precision'])}.")
    return " ".join(out) + (" Labels: Claude-judged (D30)." if g.get("labels") == "llm" else "")


def qlora_block() -> str:
    ext = load("extraction_benchmark.json")
    student = next((k for k in (ext or {}).get("results", {}) if k.startswith("mlx__")), None)
    if not student:
        return pending("teacher labels → MLX LoRA training → the student's benchmark run (D34)")
    r = ext["results"]
    line = lambda k: f"{k}: {100 * r[k]['micro_f1_mean_sd'][0]:.1f}" if k in r else f"{k}: n/a"
    return "Micro-F1 — " + "; ".join(line(k) for k in (student, "asu__gpt-oss-120b", "ollama__qwen2.5-7b", "crf")) + "."


def explanations_block() -> str:
    e = load("explanation_eval.json")
    if not e or "faithfulness" not in e:
        auto = (e or {}).get("automatic")
        return pending("human faithfulness labels on the 60 explanations") + (
            "\n\n" + pd.DataFrame(auto).T.to_markdown() if auto else "")
    df = pd.DataFrame({m: {"faithfulness": v["p_hat"], "95% CI": f"[{v['ci_low']:.3f}, {v['ci_high']:.3f}]",
                           "hallucination rate": v["hallucination_rate"],
                           "fully faithful explanations": v["explanation_level_all_supported"]}
                       for m, v in e["faithfulness"].items()}).T
    note = ("Sentences shown after the critic, judged against the graph facts they cite"
            + (" (Claude-judged, D30/D39)." if e.get("labels") == "claude" else "."))
    mc = e.get("mcnemar_vs_proprietary") or {}
    if mc:
        note += " Fully faithful explanations vs. the proprietary explainer (McNemar, Holm): " + "; ".join(
            f"{m} p_holm = {v['p_holm']:.3f}" for m, v in mc.items()) + "."
    return note + "\n\n" + df.to_markdown(floatfmt=".3f")


def kappa(a, b):
    a, b = pd.Series(a).astype(str), pd.Series(b).astype(str)
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in set(a) | set(b))
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement_block() -> str:
    rows = []
    for name, f, h, c in (("Relevance (0/1/2)", "relevance_pool.csv", "label", "llm_label"),
                          ("Dedupe pairs", "dedupe_handcheck_jobs.csv", "is_duplicate", "llm_is_duplicate"),
                          ("Skill normalization", "skill_mentions_500.csv", "gold_taxonomy_id", "llm_gold_taxonomy_id"),
                          ("Edges", "edge_sample_100.csv", "is_correct", "llm_is_correct")):
        p = ANN / f
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str).fillna("")
        if h not in d or c not in d:
            continue
        both = d[(d[h] != "") & (d[c] != "")]
        if len(both):
            k = kappa(both[h], both[c]) if both[h].nunique() < 10 else None
            rows.append({"set": name, "items checked": len(both), "agreement": f"{(both[h] == both[c]).mean():.0%}",
                         "Cohen's κ": "—" if k is None else f"{k:.2f}"})
    chk = ANN / "gold_spotcheck.json"
    if chk.exists():
        c = json.loads(chk.read_text())
        kept = sum(len(v["kept"]) for v in c.values())
        tot = sum(len(v["kept"]) + len(v["removed"]) for v in c.values())
        rows.append({"set": "Gold extraction (entities kept)", "items checked": f"{len(c)} docs",
                     "agreement": f"{kept / tot:.0%}" if tot else "—", "Cohen's κ": "—"})
    if not rows:
        return ("**No independent human check.** All test labels in this report (relevance, gold extraction, dedupe, "
                "normalization, edges and explanation faithfulness) were produced by Claude (D30); no human "
                "spot-check of them was done.")
    return "**Human spot-check agreement with Claude's labels:**\n\n" + pd.DataFrame(rows).to_markdown(index=False)


def conclusion_block() -> str:
    p = ROOT / "report/conclusion.md"
    return p.read_text().strip() if p.exists() else pending("conclusion, written once all results are in")


BLOCKS = {"DATA": data_block, "SUMMARY": summary_block, "EXTRACTION": extraction_block, "COST": cost_block,
          "RANKING": ranking_block, "SPO": spo_block, "GRAPH": graph_block, "QLORA": qlora_block,
          "EXPLANATIONS": explanations_block, "AGREEMENT": agreement_block, "CONCLUSION": conclusion_block}

if __name__ == "__main__":
    txt = (ROOT / "report/REPORT.md").read_text()
    for name, fn in BLOCKS.items():
        txt = txt.replace(f"<!-- {name} -->", fn())
    txt = txt.replace("Final report draft. `python report/build.py` fills every `<!-- … -->`\nblock from `eval/results/`; "
                      "numbers are never typed in by hand.", "Generated from `eval/results/` by `report/build.py`.")
    (ROOT / "report/REPORT_FINAL.md").write_text(txt)
    print("wrote report/REPORT_FINAL.md")
