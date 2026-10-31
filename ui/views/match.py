"""Job matches: filters as removable chips, any ranking method, job cards with skill overlap, and a cited,
critic-verified "Why this job?" per card. A second tab compares ranking methods side by side."""

import html

import pandas as pd
import streamlit as st

from ui import services as svc
from ui.components import (badge_html, chips_html, empty_state, privacy_note, require_profile, score_tone)

st.title("Job matches")
p = require_profile()
if not p or not p.get("parsed"):
    st.stop()
privacy_note(p)

SENIORITY = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
with st.expander("Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    f1.text_input("States (comma-separated, e.g. AZ, CA; remote jobs always kept)", key="f_states")
    f2.multiselect("Work mode", ["On-site", "Remote", "Hybrid"], key="f_modes")
    f3.multiselect("Seniority", SENIORITY, key="f_seniority")
states = [s.strip().upper() for s in st.session_state.get("f_states", "").split(",") if s.strip()]
prefs = {"states": states, "work_modes": st.session_state.get("f_modes", []),
         "seniority": st.session_state.get("f_seniority", [])}


def drop_filter(key: str, value=None) -> None:
    cur = st.session_state.get(key)
    if key == "f_states":
        st.session_state[key] = ", ".join(s for s in states if s != value)
    elif isinstance(cur, list):
        st.session_state[key] = [v for v in cur if v != value]
    st.session_state["rerun_match"] = True


active = [("f_states", s, f"State: {s}") for s in states] + \
    [("f_modes", v, v) for v in prefs["work_modes"]] + [("f_seniority", v, v) for v in prefs["seniority"]]
if active:
    with st.container(horizontal=True, key="filter-chips"):
        for i, (key, value, label) in enumerate(active):
            st.button(f"{label}  ✕", key=f"chip-{i}-{key}", on_click=drop_filter, args=(key, value),
                      help="Remove this filter")

tab_rec, tab_cmp = st.tabs(["Recommendations", "Compare ranking methods"])
with tab_rec:
    c1, c2, c3 = st.columns([3, 1, 1])
    method = c1.selectbox("Ranking method", list(svc.METHODS), format_func=svc.METHODS.get)
    k = c2.number_input("Results", 3, 20, 10)
    use_llm = c3.toggle("LLM explanations", value=False,
                        help="Off: template sentences built from graph facts (instant). On: local LLM writes the "
                             "explanation and the critic rejects any sentence a cited edge doesn't support.")
    if st.button("Find matches", type="primary") or st.session_state.pop("rerun_match", False):
        with st.spinner("Filtering, retrieving, fusing and re-ranking…"):
            st.session_state["matches"] = svc.rank(p, prefs, method, int(k))
        st.session_state.pop("explain", None)
    res = st.session_state.get("matches")
    if not res:
        empty_state("Press Find matches to run the ranking cascade.")
        st.stop()

    steps = "".join(f"<span class='om-step'>{html.escape(s)}</span><span class='om-arrow'>→</span>"
                    for s in res["stages"])[:-len("<span class='om-arrow'>→</span>")]
    t = res.get("timing", {})
    st.markdown(f"<div class='om-funnel'>{steps}</div>", unsafe_allow_html=True)
    st.caption(f"{res['filtered_pool']:,} jobs passed the filters" +
               (f" · ranked in {t['total_s']:.2f}s (cross-encoder {t.get('cross_encoder_s', 0):.2f}s)"
                if t.get("total_s") else ""))
    if not res["results"]:
        st.warning("No jobs matched these filters.")
    explained = st.session_state.setdefault("explain", {})
    for r in res["results"]:
        with st.container(border=True):
            meta = " · ".join(html.escape(str(x)) for x in [r.get("company"), r.get("location"), r.get("work_mode"),
                                                             r.get("seniority"), r.get("industry")]
                              if x and x != "Unknown")
            cov = r.get("coverage")
            score_badge = badge_html(f"score {r['score']:.3f}", "info", "0.85rem")
            st.markdown(
                f"<div class='om-job-head'><span class='om-job-title'>{r['rank']}. {html.escape(r.get('title') or '')}"
                f"</span><span>{score_badge}"
                + (badge_html(f'{cov:.0%} of listed skills', score_tone(cov), '0.85rem') if cov is not None else "")
                + f"</span></div><div class='om-job-meta'>{meta}</div>"
                f"<div class='om-job-skills'><span class='om-job-label'>Matched</span><span>"
                + (chips_html(r["matched"], "good", limit=8) or "<span class='om-job-none'>no normalized skill overlap</span>")
                + "</span></div><div class='om-job-skills'><span class='om-job-label'>Missing</span><span>"
                + (chips_html(r["missing"], "neutral", limit=8) or "<span class='om-job-none'>none listed</span>")
                + "</span></div>", unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns([2, 1, 1, 1])
            if b2.button("Why this job?", key=f"why-{r['doc_id']}"):
                with st.spinner("Parsing the posting into the graph if needed, then explaining…"):
                    try:
                        explained[r["doc_id"]] = svc.explain(p, r["doc_id"], use_llm)
                    except Exception as e:
                        st.error(f"Explanation failed: {type(e).__name__}: {str(e)[:200]}")
            if b3.button("Skill gap", key=f"gap-{r['doc_id']}"):
                st.session_state["selected_job"] = r["doc_id"]
                st.switch_page("views/skill_gap.py")
            if b4.button("Outreach", key=f"out-{r['doc_id']}"):
                st.session_state["selected_job"] = r["doc_id"]
                st.switch_page("views/assistant.py")
            with b1.popover("Job description"):
                st.write(svc.job_text(r["doc_id"])[:4000])
            e = explained.get(r["doc_id"])
            if e:
                llm = e.get("llm")
                sents = (llm or {}).get("sentences") or e["template"]["sentences"]
                src = "LLM (local) · critic-verified" if llm and llm.get("sentences") else "template from graph facts"
                st.markdown(f"**Why this job?** {badge_html(src, 'good' if llm and llm.get('sentences') else 'info')}",
                            unsafe_allow_html=True)
                if not sents:
                    st.caption("The graph has no shared facts for this pair yet (no matched requirement, gap or "
                               "related skill), so nothing can be claimed.")
                for s in sents:
                    cites = "".join(f"<span class='om-cite'>{html.escape(c[:6])}</span>" for c in s["cites"][:4])
                    st.markdown(f"<div class='om-sent'>{html.escape(s['text'])}{cites}</div>", unsafe_allow_html=True)
                for s in (llm or {}).get("rejected", []):
                    st.markdown(f"<div class='om-sent rej'>{html.escape(s['text'])}</div>", unsafe_allow_html=True)
                    st.caption(f"Rejected by the critic: {s.get('reason', '')}")
                with st.expander("Evidence behind each citation"):
                    ids = sorted({c for s in sents for c in s["cites"]})
                    if ids:
                        st.dataframe(svc.evidence_for(ids), hide_index=True, width="stretch")
                    st.dataframe(pd.DataFrame(e["facts"]), hide_index=True, width="stretch")
            if r.get("features"):
                with st.expander("How the score was computed"):
                    f = pd.Series({k_: v for k_, v in r["features"].items() if v is not None}, name="value")
                    st.dataframe(f.to_frame(), width="stretch")

with tab_cmp:
    st.caption("Runs every ranking method with the same CV and filters. Rankings here are not judged; nDCG/MRR "
               "against the relevance labels are on the Results page.")
    kc = st.slider("Top-k per method", 5, 15, 10)
    if st.button("Compare all methods"):
        out, prog = {}, st.progress(0.0)
        for i, m in enumerate(svc.METHODS):
            out[m] = svc.rank(p, prefs, m, kc)
            prog.progress((i + 1) / len(svc.METHODS), text=f"{m} done")
        st.session_state["cmp"] = out
    cmp = st.session_state.get("cmp")
    if cmp:
        table = {m: [f"{r.get('title', '')[:34]} ({(r.get('company') or '?')[:18]})" for r in v["results"]]
                 for m, v in cmp.items()}
        n = max(len(v) for v in table.values())
        st.dataframe(pd.DataFrame({m: v + [""] * (n - len(v)) for m, v in table.items()}, index=range(1, n + 1)),
                     width="stretch")
        overlap = {m: len({r["doc_id"] for r in v["results"]} & {r["doc_id"] for r in cmp["cascade"]["results"]})
                   for m, v in cmp.items()}
        cov = {m: pd.Series([r["coverage"] for r in v["results"] if r["coverage"] is not None]).mean()
               for m, v in cmp.items()}
        st.dataframe(pd.DataFrame({"jobs shared with the full cascade": overlap,
                                   "mean listed-skill coverage (descriptive)": cov}).round(2), width="stretch")
