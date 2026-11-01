"""Landing page: what the assistant does, how it works, and two ways to start (your CV, or a public sample)."""

import streamlit as st

from ui import services as svc
from ui.components import profile

DEMO_CV = "cvk_28897981"       # public Kaggle sample (Information Technology) used by "Try a sample CV"


def try_sample() -> None:
    samples = svc.sample_profiles().set_index("doc_id")
    doc_id = DEMO_CV if DEMO_CV in samples.index else samples.index[0]
    row = samples.loc[doc_id]
    p = svc.new_profile(row["text"], source="kaggle_resume_dataset", doc_id=doc_id,
                        label=f"Sample CV ({row['category'].title().replace('-', ' ')})")
    svc.parse_profile(p, fast=True)
    st.session_state["profile"] = p
    for k in ("matches", "explain", "outreach", "chat"):
        st.session_state.pop(k, None)


st.markdown("""
<div class="om-landing-hero">
  <div class="om-eyebrow">Explainable job matching</div>
  <h1>Find the jobs that fit your CV, and see exactly why.</h1>
  <p>Upload your CV and get ranked matches from 20,000 LinkedIn postings or live openings. Every recommendation
  comes with reasons drawn straight from your CV and the job post, plus the skills you would need to close the gap.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.2, 1.2, 3])
if c1.button("Upload your CV", type="primary", icon=":material/upload_file:", width="stretch"):
    st.switch_page("views/profile.py")
if c2.button("Try a sample CV", icon=":material/bolt:", width="stretch",
             help="Loads an anonymized public resume and parses it in a few seconds."):
    with st.spinner("Loading and parsing a sample CV…"):
        try_sample()
    st.switch_page("views/match.py")
if profile():
    c3.page_link("views/match.py", label="Continue to your job matches", icon=":material/arrow_forward:")

st.markdown("""
<div class="om-stats">
  <div class="om-stat-card"><div class="v">20,000</div><div class="l">LinkedIn job postings indexed</div></div>
  <div class="om-stat-card"><div class="v">Live</div><div class="l">openings searched through Adzuna</div></div>
  <div class="om-stat-card"><div class="v">100%</div><div class="l">of CV processing stays on your computer</div></div>
  <div class="om-stat-card"><div class="v">Cited</div><div class="l">every explanation points to a fact</div></div>
</div>

<div class="om-section-title">How it works</div>
<div class="om-section-sub">Three steps from CV to a short list you can act on.</div>
<div class="om-steps">
  <div class="om-step-card"><div class="om-step-num">1</div><h4>Add your CV</h4>
    <p>Upload a PDF or Word file. The assistant reads your skills, roles, credentials and projects, and lets you
    correct anything it got wrong.</p></div>
  <div class="om-step-card"><div class="om-step-num">2</div><h4>Get matched</h4>
    <p>Jobs are ranked by how well they fit your experience, with filters for location, work mode and seniority,
    or a live search with keywords suggested from your CV.</p></div>
  <div class="om-step-card"><div class="om-step-num">3</div><h4>Understand and act</h4>
    <p>See why each job fits, which skills are missing and how to close the gap, then draft a message to the
    recruiter.</p></div>
</div>

<div class="om-section-title">What you can do</div>
<div class="om-section-sub">Everything in one place, for job seekers and recruiters.</div>
""", unsafe_allow_html=True)

FEATURES = [
    ("lock", "Private by design", "Your CV is read and explained by a model running on this computer. It is never "
     "sent to an online AI service.", "views/profile.py"),
    ("work", "Ranked job matches", "Matches from 20,000 postings or live openings, ranked on skills, experience "
     "and role fit, with the posting date and full description.", "views/match.py"),
    ("fact_check", "Reasons you can check", "Each \"Why this job?\" sentence cites the fact it comes from, and "
     "anything that cannot be backed up is removed.", "views/match.py"),
    ("trending_up", "Skill gap plan", "See which requirements you meet, which you miss, related skills you already "
     "have, and suggested CV rewrites.", "views/skill_gap.py"),
    ("forum", "Assistant and outreach", "Ask questions about a match and get a draft message to the recruiter with "
     "a follow-up reminder.", "views/assistant.py"),
    ("group", "Recruiter view", "Pick a job and rank thousands of candidate CVs for it, with the same matching "
     "and explanations.", "views/candidates.py"),
]
for row in (FEATURES[:3], FEATURES[3:]):
    cols = st.columns(3)
    for col, (icon, title, text, page) in zip(cols, row):
        with col, st.container(border=True, height=215, key=f"feature-{icon}"):
            st.markdown(f"#### :material/{icon}: {title}")
            st.markdown(text)
            st.page_link(page, label="Open", icon=":material/arrow_forward:")

st.markdown("<div class='om-footer'>LinkedIn Assistant · CSE 573, Group 3, Fall 2026 · Matching runs on open models; "
            "job postings come from the public LinkedIn dataset on Kaggle and from Adzuna.</div>",
            unsafe_allow_html=True)
