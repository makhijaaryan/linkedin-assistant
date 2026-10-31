"""LinkedIn Assistant — Streamlit frontend (R6):  streamlit run ui/app.py

Multi-page app. Page layout and visual design are ported from Krish Naik's OpenMatch frontend (team repo, branch
Krish) and wired straight to this project's pipeline (router, extraction, Neo4j graph, ranking cascade, agents,
evaluation results). Uploaded CVs are private: parsed, explained and checked on local Ollama only (D2).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.components import badge_html, load_theme, profile  # noqa: E402

st.set_page_config(page_title="LinkedIn Assistant", page_icon="🧭", layout="wide")
load_theme()

HERE = Path(__file__).parent
pages = {
    "Career": [
        st.Page(HERE / "views/dashboard.py", title="Dashboard", icon="🏠", default=True),
        st.Page(HERE / "views/profile.py", title="My profile", icon="📄"),
        st.Page(HERE / "views/match.py", title="Job matches", icon="🎯"),
        st.Page(HERE / "views/skill_gap.py", title="Skill gap", icon="🧩"),
        st.Page(HERE / "views/graph.py", title="Knowledge graph", icon="🕸️"),
        st.Page(HERE / "views/assistant.py", title="Assistant & outreach", icon="💬"),
    ],
    "Recruiter": [
        st.Page(HERE / "views/candidates.py", title="Find candidates", icon="🔎"),
    ],
    "Research": [
        st.Page(HERE / "views/models.py", title="Models & router", icon="⚙️"),
        st.Page(HERE / "views/research.py", title="Results", icon="📊"),
        st.Page(HERE / "views/annotate.py", title="Annotation", icon="✍️"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🧭 LinkedIn Assistant")
    st.caption("Explainable, open-weight job matching")
    p = profile()
    if p:
        st.markdown(f"**Active CV:** {p['label']}  \n"
                    + badge_html("private · local only" if p.get("private", True) else "public sample",
                                 "bad" if p.get("private", True) else "info")
                    + (badge_html("parsed", "good") if p.get("parsed") else badge_html("not parsed", "neutral")),
                    unsafe_allow_html=True)
    else:
        st.caption("No CV loaded yet.")
    st.divider()
    st.caption("CV text never leaves this laptop: parsing, explanations and the critic run on local Ollama. Only "
               "public job postings may be parsed on ASU Voyager.")

nav.run()
