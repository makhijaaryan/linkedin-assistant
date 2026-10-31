"""Results: the proposal's targets (Table II) with their status, and one tab per research question, read straight from
eval/results/*.json. Nothing is estimated: a result that doesn't exist yet says what it still needs."""

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import services as svc
from ui.components import badge_html

st.title("Results")
pr = svc.progress()
c = st.columns(5)
c[0].metric("Overall", f"{pr['overall']:.0%}")
for col, (k, label) in zip(c[1:], (("runs", "Model runs"), ("labels", "Labels"), ("frontend", "Frontend"),
                                    ("train_eval", "Training & eval"))):
    col.metric(label, f"{pr[k][0]:.0%}", help=pr[k][1])

summary = svc.ROOT / "eval/results/SUMMARY.md"
with st.expander("Targets (Table II, from eval/results/SUMMARY.md)", expanded=True):
    if summary.exists():
        txt = summary.read_text().split("## Router cost")[0]
        st.markdown(txt.replace("PENDING", "⏳ PENDING"))
        st.caption("Regenerate with `python -m eval.report`.")
    else:
        st.info("No summary yet: run `python -m eval.report`.")


def pending(msg: str) -> None:
    st.markdown(badge_html("pending", "neutral") + f" {msg}", unsafe_allow_html=True)


tabs = st.tabs(["RQ1 · Extraction & cost", "RQ4 · Ranking", "R3 · Graph", "RQ3 · Ontology vs SPO",
                "R6 · Explanations", "RQ2 · QLoRA"])
with tabs[0]:
    ext = svc.result("extraction_benchmark.json")
    if not ext:
        pending("needs the gold extraction labels and the benchmark runs (Models & router shows run progress).")
    else:
        if str(ext.get("gold", "")).startswith("dry-run"):
            st.warning("Dry run: the 'gold' is another model's output. Pipeline check only, not a result.")
        rows = []
        for name, r in ext["results"].items():
            eff = r.get("efficiency", {})
            rows.append({"extractor": name, "micro-F1": 100 * r["micro_f1_mean_sd"][0],
                         "sd": 100 * r["micro_f1_mean_sd"][1], "docs": r["n_docs"], "runs": r["runs"],
                         "p50 s/doc": (eff.get("p50_s") or {}).get("estimate"),
                         "$/1K docs": eff.get("usd_per_1k_docs"),
                         **{lab: 100 * v["f1"][0] for lab, v in r["per_type_mean_sd"].items()}})
        df = pd.DataFrame(rows).sort_values("micro-F1", ascending=False)
        st.dataframe(df.round(2), hide_index=True, width="stretch")
        plot = df.dropna(subset=["$/1K docs"])
        if len(plot):
            st.plotly_chart(px.scatter(plot, x="$/1K docs", y="micro-F1", text="extractor", size="docs",
                                       error_y="sd", title="Quality vs cost (RQ1)")
                            .update_traces(textposition="top center").update_layout(height=420), width="stretch")
        md = svc.ROOT / "eval/results/extraction_benchmark.md"
        if md.exists():
            with st.expander("Full table and pre-registered tests (vs proprietary, vs CRF)"):
                st.markdown(md.read_text())
with tabs[1]:
    rk = svc.result("ranking_eval.json")
    if not rk:
        pending("needs the relevance labels for the 40 profiles (`python -m eval.ranking_eval`).")
    else:
        m = pd.DataFrame(rk["means"]).T.reset_index(names="system").sort_values("ndcg@10", ascending=False)
        st.plotly_chart(px.bar(m, x="system", y="ndcg@10", title=f"nDCG@10 ({rk['n_profiles']} profiles)")
                        .update_layout(height=360), width="stretch")
        st.dataframe(m.round(3), hide_index=True, width="stretch")
        cvb = rk["cascade_vs_bm25"]
        st.markdown(f"**Cascade vs BM25:** Δ nDCG@10 = {cvb['mean_diff']:+.3f} [{cvb['ci_low']:+.3f}, "
                    f"{cvb['ci_high']:+.3f}], relative {100 * (cvb['relative_gain'] or 0):+.1f}%, p = {cvb['p']:.4f} · "
                    + badge_html("+15% met" if cvb["target_plus15_met"] else "+15% not met",
                                 "good" if cvb["target_plus15_met"] else "warn"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rk["ablations_vs_full"]).T[["mean_diff", "ci_low", "ci_high", "p_holm"]].round(4),
                     width="stretch")
        st.caption(f"Labels used: {rk['labels']}.")
with tabs[2]:
    g = svc.result("graph_eval.json")
    if not g:
        pending("needs the 500-mention and 100-edge sheets (`python -m graph.annotate --score`).")
    else:
        for k in ("normalization_classical", "normalization_modern", "edge_precision"):
            if k in g:
                v = g[k]
                st.markdown(f"**{k.replace('_', ' ')}:** {v['p_hat']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] "
                            f"(n = {v.get('n', '?')}) · target 0.90")
        if "modern_vs_classical_mcnemar" in g:
            st.caption(f"Modern vs classical normalization, McNemar p = {g['modern_vs_classical_mcnemar']['p']:.4f}")
    gc = svc.graph_counts()
    if gc["ok"]:
        st.bar_chart(pd.Series(gc["nodes"], name="nodes").sort_values(ascending=False))
with tabs[3]:
    spo = svc.result("spo_ablation.json")
    if not spo:
        pending("needs the SPO extraction run and `python -m eval.spo_eval`.")
    else:
        st.json(spo, expanded=False)
        if "ndcg@10" in spo:
            st.bar_chart(pd.Series(spo["ndcg@10"], name="nDCG@10"))
with tabs[4]:
    ex = svc.result("explanation_eval.json")
    if not ex:
        pending("needs the 60-explanation faithfulness sheet (Annotation page).")
    else:
        if "faithfulness" in ex:
            f = pd.DataFrame({m: {"faithfulness": v["p_hat"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                                  "all-supported explanations": v["explanation_level_all_supported"]}
                              for m, v in ex["faithfulness"].items()}).T
            st.dataframe(f.round(3), width="stretch")
        st.json(ex.get("automatic", {}), expanded=False)
with tabs[5]:
    q = svc.result("qlora_eval.json")
    if not q:
        pending("needs the teacher labels (running) and the QLoRA training run.")
    else:
        st.json(q, expanded=False)
