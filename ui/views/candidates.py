"""Find candidates (recruiter view): job -> best CVs with the same first stage + cross-encoder + MMR."""

import pandas as pd
import streamlit as st

from ui import services as svc
from ui.components import page_header, SOURCE_LABEL, badge_html

page_header("Find candidates", "Pick a job posting and rank the indexed CVs for it.")
st.caption("Ranks the 4,981 indexed CVs for one posting. CV text is only shown for public Kaggle resumes.")
jobs = svc.jobs()
c1, c2 = st.columns([3, 1])
title = c1.text_input("Search job titles", "Data Analyst")
k = c2.number_input("Candidates", 5, 30, 10)
hits = jobs.meta[jobs.meta["title"].str.contains(title, case=False, na=False, regex=False)].head(80)
if not len(hits):
    st.info("No job title matches that search.")
    st.stop()
labels = dict(zip(hits["doc_id"], hits["title"] + " — " + hits["company"].fillna("") + " · " + hits["location"].fillna("")))
job_id = st.selectbox("Job", list(labels), format_func=labels.get)
with st.expander("Job description"):
    st.write(svc.job_text(job_id)[:3000])
if st.button("Rank candidates", type="primary"):
    with st.spinner("Ranking candidates…"):
        st.session_state["cands"] = (job_id, svc.rank_candidates(job_id, int(k)))
got = st.session_state.get("cands")
if got and got[0] == job_id:
    out = got[1]
    st.caption(" → ".join(out["stages"]) + f" · {out['timing']['total_s']:.2f}s")
    texts = svc.resumes_meta().set_index("doc_id")
    for r in out["results"]:
        public = r.get("source") == "kaggle_resume_dataset"
        with st.container(border=True):
            st.markdown(f"**{r['rank']}. {r['doc_id']}** · {r.get('category', '')} "
                        + badge_html(f"score {r['score']:.3f}", "info")
                        + badge_html(SOURCE_LABEL.get(r.get("source"), r.get("source")), "info" if public else "bad"),
                        unsafe_allow_html=True)
            if public:
                with st.expander("CV text"):
                    st.write(texts.at[r["doc_id"], "text"][:3000])
    st.dataframe(pd.DataFrame(out["results"])[["rank", "doc_id", "category", "score", "ce"]], hide_index=True,
                 width="stretch")
