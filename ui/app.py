"""LinkedIn Assistant — Streamlit frontend (R6):  streamlit run ui/app.py

Multi-page app with top navigation, a landing page and a light/dark toggle (light by default). Card and badge
design adapted from Krish Naik's OpenMatch frontend (team repo, branch Krish), wired straight to this project's
pipeline (router, extraction, Neo4j graph, ranking cascade, agents, evaluation results). Uploaded CVs are private:
parsed, explained and checked on local Ollama only (D2).
"""

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ranking.live  # noqa: E402,F401  loads the LambdaMART models before torch is imported (macOS OpenMP clash)
import streamlit as st  # noqa: E402

from ui.components import badge_html, load_theme, profile, sync_theme, theme_toggle  # noqa: E402

HERE = Path(__file__).parent
st.set_page_config(page_title="LinkedIn Assistant", page_icon=str(HERE / "assets/logo_icon.svg"), layout="wide")
sync_theme()
load_theme()
st.logo(str(HERE / "assets/logo.svg"), icon_image=str(HERE / "assets/logo_icon.svg"), size="large")

pages = {
    "": [st.Page(HERE / "views/home.py", title="Home", icon=":material/home:", default=True)],
    "Job seeker": [
        st.Page(HERE / "views/profile.py", title="My profile", icon=":material/person:"),
        st.Page(HERE / "views/dashboard.py", title="Dashboard", icon=":material/space_dashboard:"),
        st.Page(HERE / "views/match.py", title="Job matches", icon=":material/work:"),
        st.Page(HERE / "views/skill_gap.py", title="Skill gap", icon=":material/trending_up:"),
        st.Page(HERE / "views/graph.py", title="Knowledge graph", icon=":material/hub:"),
        st.Page(HERE / "views/assistant.py", title="Assistant & outreach", icon=":material/forum:"),
    ],
    "Recruiter": [
        st.Page(HERE / "views/candidates.py", title="Find candidates", icon=":material/group:"),
    ],
    "Research": [
        st.Page(HERE / "views/models.py", title="Models & router", icon=":material/tune:"),
        st.Page(HERE / "views/research.py", title="Results", icon=":material/insights:"),
    ],
}
nav = st.navigation(pages, position="top")

# Status strip on every page: the active CV (with its privacy status) and the theme toggle.
with st.container(key="topbar"):
    left, right = st.columns([5, 1], vertical_alignment="center")
    p = profile()
    with left:
        if p:
            st.markdown(f"<span class='om-cv'>Active CV: <b>{html.escape(p['label'])}</b></span> "
                        + badge_html("private, stays on this computer" if p.get("private", True) else "public sample",
                                     "bad" if p.get("private", True) else "info", "0.76rem")
                        + (badge_html("parsed", "good", "0.76rem") if p.get("parsed")
                           else badge_html("not parsed", "neutral", "0.76rem")), unsafe_allow_html=True)
        else:
            st.markdown("<span class='om-cv'>No CV loaded yet. Upload one on <b>My profile</b> or try a sample on "
                        "<b>Home</b>.</span>", unsafe_allow_html=True)
    with right:
        theme_toggle()

nav.run()
