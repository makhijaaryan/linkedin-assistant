"""Phase 8: collect every result file into eval/results/SUMMARY.md (Table II + the RQ1 matrix).

  python -m eval.report

A target whose inputs don't exist yet is listed as PENDING with what it still needs, never estimated.
"""

import json
from pathlib import Path

import pandas as pd

R = Path("eval/results")


CLAUDE = " (Claude-judged, D30)"


def load(name):
    """A result file; label-dependent results fall back to the Claude-labeled version, marked with `_claude`."""
    stem = name.removesuffix(".json")
    for cand, claude in ((name, False), (f"{stem}_llm.json", True), (f"{stem}_llm_label.json", True)):
        p = R / cand
        if p.exists():
            out = json.loads(p.read_text())
            out["_claude"] = claude
            return out
    return None


def fmt_ci(d, scale=1.0, digits=3):
    return f"{scale * d['p_hat']:.{digits}f} [{scale * d['ci_low']:.{digits}f}, {scale * d['ci_high']:.{digits}f}]"


def cost_matrix() -> pd.DataFrame | None:
    log = Path("router/logs/calls.jsonl")
    if not log.exists():
        return None
    df = pd.DataFrame([json.loads(x) for x in open(log) if x.strip()])
    df = df[~df["cached"] & (df["task"] != "test")]
    if df.empty:
        return None
    return df.groupby(["task", "provider", "model"]).agg(
        calls=("latency_s", "size"), p50_s=("latency_s", "median"), p95_s=("latency_s", lambda x: x.quantile(0.95)),
        tokens=("completion_tokens", "sum")).reset_index()


def main():
    p1, ext, graph, rank = load("phase1_report.json"), load("extraction_benchmark.json"), load("graph_eval.json"), \
        load("ranking_eval.json")
    spo, expl, us = load("spo_ablation.json"), load("explanation_eval.json"), load("user_study.json")
    rows = []
    add = lambda thread, metric, target, value: rows.append(f"| {thread} | {metric} | {target} | {value} |")
    hc = Path("data/processed/annotation/dedupe_handcheck_jobs.csv")
    h = pd.read_csv(hc, dtype=str).fillna("") if hc.exists() else pd.DataFrame()
    col = next((c for c in ("is_duplicate", "llm_is_duplicate") if c in h and (h[c] != "").sum() >= 0.9 * len(h)), None)
    if col:
        from eval import stats
        h = h[h[col] != ""]
        add("R1", "Dedupe precision (200 job pairs)" + (CLAUDE if col.startswith("llm") else ""), "≥ 0.95",
            fmt_ci(stats.wilson(int(h[col].astype(int).sum()), len(h))))
    else:
        add("R1", "Dedupe precision", "≥ 0.95", "PENDING: fill is_duplicate in dedupe_handcheck_jobs.csv")
    if p1:
        leak = p1["leakage"]["jobs"]["clusters_spanning_splits"] + p1["leakage"]["resumes"]["clusters_spanning_splits"]
        add("R1", "Split leakage (clusters across splits)", "0", str(leak))
    if ext and not ext["gold"].startswith("dry-run"):
        best = max((k for k in ext["results"] if k != "crf"), key=lambda k: ext["results"][k]["micro_f1_mean_sd"][0],
                   default=None)
        if best:
            add("R2 / RQ1", f"Micro-F1, best open ({best})", "≥ 0.80",
                f"{ext['results'][best]['micro_f1_mean_sd'][0]:.3f}")
            vc = ext["comparisons"].get(best, {}).get("vs_crf")
            if vc:
                add("R2", "ΔF1 vs CRF (points)", "≥ +10 (CI low)",
                    f"{vc['mean_diff']:+.1f} [{vc['ci_low']:+.1f}, {vc['ci_high']:+.1f}]")
    else:
        add("R2 / RQ1", "Per-type F1 on 150 docs", "≥ 0.80; +10 over CRF",
            "PENDING: gold labels (Label Studio export -> data/processed/annotation/gold_extraction.json)")
    gtag = CLAUDE if graph and graph.get("_claude") else ""
    if graph and "normalization_modern" in graph:
        add("R3", "Top-1 normalization (modern)" + gtag, "≥ 0.90", fmt_ci(graph["normalization_modern"]))
        add("R3", "Top-1 normalization (classical)" + gtag, "≥ 0.90", fmt_ci(graph["normalization_classical"]))
    else:
        add("R3", "Top-1 normalization (500)", "≥ 0.90", "PENDING: fill gold_taxonomy_id in skill_mentions_500.csv")
    if graph and "edge_precision" in graph:
        add("R3", "Edge precision (100)" + gtag, "≥ 0.90", fmt_ci(graph["edge_precision"]))
    else:
        add("R3", "Edge precision (100)", "≥ 0.90", "PENDING: fill is_correct in edge_sample_100.csv")
    if spo:
        add("R3 / RQ3", "Duplicate-node rate SPO vs ontology", "report",
            f"{spo['spo']['rate']:.3f} vs {spo['ontology']['rate']:.3f} (z-test p = {spo['dup_rate_test']['p']:.4f})")
    else:
        add("R3 / RQ3", "Ontology vs SPO", "report", "PENDING: run extraction.spo, then eval.spo_eval")
    if rank:
        c = rank["cascade_vs_bm25"]
        add("R4 / RQ4", f"nDCG@10 cascade vs BM25 ({rank['n_profiles']} profiles)" + (CLAUDE if rank.get("_claude") else ""),
            "+15%",
            f"{rank['means']['cascade_mmr']['ndcg@10']:.3f} vs {rank['means']['bm25']['ndcg@10']:.3f} "
            f"({100 * (c['relative_gain'] or 0):+.1f}%, p = {c['p']:.4f})")
        for k, v in rank["ablations_vs_full"].items():
            add("R4 ablation", k, "report", f"Δ {v['mean_diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}], "
                                           f"p_holm = {v['p_holm']:.4f}")
    else:
        add("R4 / RQ4", "nDCG@10, MRR, P@5 on 40 profiles", "+15% nDCG over BM25",
            "PENDING: 800 relevance labels (streamlit run ui/judge.py)")
    student = next((k for k in (ext or {}).get("results", {}) if k.startswith("mlx__")), None)
    teacher = "asu__gpt-oss-120b"
    if student and teacher in ext["results"]:
        s_f1, t_f1 = (100 * ext["results"][k]["micro_f1_mean_sd"][0] for k in (student, teacher))
        add("R5 / RQ2", f"LoRA student F1 vs teacher ({student})", "within 3 pts",
            f"{s_f1:.1f} vs {t_f1:.1f} ({s_f1 - t_f1:+.1f})")
    else:
        add("R5 / RQ2", "LoRA F1 vs teacher", "within 3 pts", "PENDING: teacher labels, then scripts/qlora_mlx.sh "
                                                             "(D34) and the student's benchmark run")
    if expl and "faithfulness" in expl:
        for m, v in expl["faithfulness"].items():
            add("R6 / RQ4", f"Faithfulness ({m})", "≥ 0.95", fmt_ci(v))
    else:
        add("R6 / RQ4", "Explanation faithfulness (60)", "≥ 0.95", "PENDING: fill supported in explanations_60.csv")
    if us:
        add("R6", "User study mean (LLM explainer)", "≥ 4.0", f"{us['mean'].get('B_llm', float('nan')):.2f} "
                                                            f"(n = {us['participants']})")
    else:
        add("R6", "Blind A/B Likert (15-20 users)", "mean ≥ 4.0", "PENDING: run the study (eval/user_study/PROTOCOL.md)")
    lines = ["# Results summary (Table II)", "", "Generated by `python -m eval.report`. PENDING rows list what is "
             "still needed; nothing here is estimated.", "", "| Thread | Metric | Target | Result |", "|---|---|---|---|",
             *rows, ""]
    if ext:
        lines += ["## RQ1 extraction matrix", "", (R / "extraction_benchmark.md").read_text() if
                  (R / "extraction_benchmark.md").exists() else "", ""]
    cm = cost_matrix()
    if cm is not None:
        lines += ["## Router cost / latency (live calls, all tasks)", "", cm.to_markdown(index=False, floatfmt=".2f"), ""]
    (R / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
