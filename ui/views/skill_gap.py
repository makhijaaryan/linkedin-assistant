"""Skill gap for one job: requirement coverage, missing required skills with the related skills the CV can build
from (RELATED_TO bridges), an upskilling plan, and critic-checked CV rewrites. Every row cites graph edges."""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import services as svc
from ui.components import page_header, badge_html, chips, empty_state, privacy_note, require_profile

page_header("Skill gap", "What a job asks for, what your CV shows, and how to close the gap.")
p = require_profile()
if not p or not p.get("parsed"):
    st.stop()
privacy_note(p)

res = st.session_state.get("matches")
options = {r["doc_id"]: f"{r['rank']}. {r.get('title')} — {r.get('company')}" for r in (res or {}).get("results", [])}
sel = st.session_state.get("selected_job")
if sel and sel not in options:
    m = svc.job_meta(sel)
    options = {sel: f"{m['title']} — {m['company']}", **options}
if not options:
    empty_state("Run Job matches first, then pick a job here.")
    st.stop()
job_id = st.selectbox("Job", list(options), format_func=options.get,
                      index=list(options).index(sel) if sel in options else 0)
st.session_state["selected_job"] = job_id

cache = st.session_state.setdefault("explain", {})
if job_id not in cache:
    with st.spinner("Parsing the posting into the graph if needed and collecting facts…"):
        try:
            cache[job_id] = svc.explain(p, job_id, use_llm=False)
        except Exception as e:
            st.error(f"Could not build the facts: {type(e).__name__}: {str(e)[:200]}")
            st.stop()
facts = cache[job_id]["facts"]
met = [f for f in facts if f["kind"] == "met"]
gaps = [f for f in facts if f["kind"] == "gap"]
bridges = [f for f in facts if f["kind"] == "bridge"]
exp = [f for f in facts if f["kind"] in ("experience", "occupation")]

total = len(met) + len(gaps)
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    cov = len(met) / total if total else 0
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * cov, number={"suffix": "%"},
                                 title={"text": "Requirement skills you show"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0a66c2"}}))
    st.plotly_chart(fig.update_layout(height=220, margin=dict(t=40, b=0, l=30, r=30)), width="stretch")
c2.metric("Matched", len(met))
c3.metric("Missing (required)", len(gaps))
c4.metric("Bridgeable", len({b["text"].split(" (required")[0] for b in bridges}))
if not facts:
    st.info("The graph has no requirement facts for this job yet; its posting may list no normalizable skills.")

t1, t2, t3, t4 = st.tabs(["Matched", "Missing & how to close it", "Experience", "CV rewrites"])
with t1:
    for f in met:
        st.markdown(f"<div class='om-sent'>{html.escape(f['text'])}</div>", unsafe_allow_html=True)
    if not met:
        st.caption("No requirement skill is matched by the CV.")
with t2:
    plan = svc.gap_plan(facts)
    if not plan:
        st.caption("Nothing required is missing.")
    for g in plan:
        with st.container(border=True):
            st.markdown(f"**{g['skill']}** " + (badge_html("bridge from your skills", "warn") if g["build_from"]
                                                 else badge_html("new skill", "neutral")), unsafe_allow_html=True)
            st.write(g["pointer"])
            if g["build_from"]:
                chips(g["build_from"], "good")
            st.caption("Cites " + ", ".join(c[:6] for c in g["requirement_cites"] + g["bridge_cites"]))
with t3:
    for f in exp:
        st.markdown(f"<div class='om-sent'>{html.escape(f['text'])}</div>", unsafe_allow_html=True)
    if not exp:
        st.caption("The posting states no years requirement, or no role maps to the same O*NET occupation.")
with t4:
    st.caption("A local LLM suggests rewrites that surface skills the job asks for; the critic rejects any rewrite "
               "that claims something the CV evidence doesn't support.")
    if st.button("Suggest rewrites (local model)"):
        with st.spinner("Writing and checking rewrites on local Ollama…"):
            st.session_state.setdefault("rewrites", {})[job_id] = svc.rewrites(facts, options[job_id].split(". ", 1)[-1])
    for w in st.session_state.get("rewrites", {}).get(job_id, []):
        ok = w["verdict"] == "approved"
        st.markdown(f"{':material/check_circle:' if ok else ':material/block:'} **Original:** {w['original']}  \n**Rewrite:** {w['rewrite']}  \n*{w['reason']}*")

with st.expander("All facts (with edge citations)"):
    st.dataframe(pd.DataFrame(facts), hide_index=True, width="stretch")
