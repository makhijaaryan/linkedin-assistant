"""Dashboard (bento layout after Krish Naik's design): your profile and best matches, the job market in the index,
the knowledge graph, and how far the research pipeline has got."""

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import services as svc
from ui.components import (badge_html, chips, empty_state, hero, nav_link, page_header, profile, row_html, score_tone,
                           stat, tile)

page_header("Dashboard", "Your profile, your best matches and the job market at a glance.")
p = profile()
res = st.session_state.get("matches")
BLUES = ["#0a66c2", "#3b8be0", "#70b5f9", "#a8d4ff", "#d0e8ff", "#e8f3ff"]

a, b, s = st.columns([2, 1, 1])
with a, tile("profile", 330):
    st.subheader("Your profile")
    if not p or not p.get("parsed"):
        empty_state("No CV loaded. Upload one or pick a public sample to get started.")
        nav_link("views/profile.py", "Load a CV")
    else:
        parsed = p["parsed"]
        st.markdown(f"**{p['label']}** " + badge_html(p.get("parser", ""), "info")
                    + badge_html("private" if p.get("private", True) else "public sample",
                                 "bad" if p.get("private", True) else "info"), unsafe_allow_html=True)
        st.caption(f"{len(parsed.get('experiences', []))} roles · {len(parsed.get('credentials', []))} credentials · "
                   f"occupation family: {(p.get('query') or {}).get('family') or '—'}")
        st.markdown("**Skills**")
        chips([s_["name"] for s_ in parsed.get("skills", [])], "info", limit=16, empty="No skills found.")
with b, tile("hero", 330):
    covs = [r["coverage"] for r in (res or {}).get("results", []) if r.get("coverage") is not None]
    hero("Listed-skill coverage", f"{sum(covs) / len(covs):.0%}" if covs else "—",
         "Share of each recommended job's normalized skills that your CV shows, averaged over your current matches.")
with s:
    g = svc.graph_counts()
    with tile("stat-jobs", 98):
        stat("Indexed jobs", f"{len(svc.jobs().meta):,}", "LinkedIn postings (Kaggle)")
    with tile("stat-cvs", 98):
        stat("Indexed CVs", f"{len(svc.cvs().meta):,}", "for the recruiter view")
    with tile("stat-graph", 98):
        stat("Graph", f"{sum(g['nodes'].values()):,} nodes" if g["ok"] else "offline",
             f"{g['edges']:,} edges" if g["ok"] else "start Neo4j (docker compose)")

a, b = st.columns([2, 1])
with a, tile("recs", 400):
    st.subheader("Top matches")
    if not res or not res.get("results"):
        empty_state("No matches yet.")
        if p:
            nav_link("views/match.py", "Find matches")
    else:
        st.markdown("".join(row_html(r["rank"], r.get("title") or "", f"{r.get('company') or '—'} · {r.get('location') or ''}",
                                     badge_html(f"{r['coverage']:.0%} skills", score_tone(r["coverage"]))
                                     if r.get("coverage") is not None else "") for r in res["results"][:6]),
                    unsafe_allow_html=True)
        nav_link("views/match.py", "See all matches")
with b, tile("modes", 400):
    st.subheader("Work mode in the index")
    wm = svc.jobs().meta["work_mode"].fillna("Unknown").value_counts().rename_axis("work mode").reset_index(name="jobs")
    st.plotly_chart(px.pie(wm, names="work mode", values="jobs", hole=0.6, color_discrete_sequence=BLUES)
                    .update_layout(height=300, margin=dict(t=6, b=6, l=6, r=6), legend=dict(orientation="h", y=-0.1)),
                    width="stretch")

a, b = st.columns([1, 1])
with a, tile("industries", 360):
    st.subheader("Top industries")
    ind = svc.jobs().meta["industry"].fillna("Unknown").value_counts().head(8).rename_axis("industry").reset_index(name="jobs")
    st.plotly_chart(px.bar(ind, x="jobs", y="industry", orientation="h", color_discrete_sequence=BLUES)
                    .update_layout(height=280, margin=dict(t=6, b=6, l=6, r=6), yaxis_title=None,
                                   yaxis=dict(autorange="reversed")), width="stretch")
with b, tile("research", 360):
    st.subheader("Research pipeline")
    pr = svc.progress()
    st.markdown(f"**Overall {pr['overall']:.0%}**")
    for k, label in (("runs", "Model runs"), ("labels", "Labels"), ("train_eval", "Training & evaluation")):
        pct, detail = pr[k]
        st.progress(min(pct, 1.0), text=f"{label}: {pct:.0%}")
    nav_link("views/research.py", "Open results")
