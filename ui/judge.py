"""Relevance judging tool (annotation guide, Task C):  streamlit run ui/judge.py

Shows one (CV, job) pair at a time, blind to which system retrieved it, and saves 0/1/2 into the `label`
column of data/processed/annotation/relevance_pool.csv after every click.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranking.pool import POOL  # noqa: E402

st.set_page_config(page_title="Relevance judging", layout="wide")


@st.cache_data
def texts():
    jobs = pd.read_parquet("data/processed/jobs.parquet", columns=["doc_id", "title", "company", "text"]).set_index("doc_id")
    cvs = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "text"]).set_index("doc_id")
    return jobs, cvs


pool = pd.read_csv(POOL, dtype=str).fillna("")
jobs, cvs = texts()
todo = pool.index[pool["label"] == ""].tolist()
st.progress(1 - len(todo) / len(pool), text=f"{len(pool) - len(todo)} / {len(pool)} pairs judged")
if not todo:
    st.success("All pairs judged.")
    st.stop()
i = todo[0]
row = pool.loc[i]
st.caption("2 = strong fit (meets most required qualifications) · 1 = partial fit (clear gaps) · "
           "0 = not a fit. Judge stated qualifications only, not location or salary.")
left, right = st.columns(2)
with left:
    st.subheader(f"CV {row['profile_id']}")
    st.text_area("cv", cvs.at[row["profile_id"], "text"], height=600, label_visibility="collapsed")
with right:
    j = jobs.loc[row["job_id"]]
    st.subheader(f"{j['title']} — {j['company']}")
    st.text_area("job", j["text"], height=600, label_visibility="collapsed")
notes = st.text_input("Notes (optional)")
cols = st.columns(3)
for label, col in zip(("0", "1", "2"), cols):
    if col.button(f"{label}", width="stretch", type="primary" if label == "2" else "secondary"):
        pool.at[i, "label"] = label
        pool.at[i, "notes"] = notes
        pool.to_csv(POOL, index=False)
        st.rerun()
