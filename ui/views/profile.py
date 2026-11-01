"""My profile: load a CV (upload, paste, or a public sample), parse it locally, review and correct the skills."""

import re
import pandas as pd
import streamlit as st

from ui import services as svc
from ui.components import page_header, SOURCE_LABEL, badge_html, chips, chips_html, empty_state, privacy_note, profile

page_header("My profile", "Add your CV, check what was extracted, and correct your skills.")

with st.expander("Load a CV" if not profile() else "Load a different CV", expanded=not profile()):
    t_up, t_paste, t_sample = st.tabs(["Upload", "Paste text", "Public sample (demo)"])
    raw, source, doc_id, label = None, None, None, ""
    with t_up:
        up = st.file_uploader("PDF, DOCX or TXT (max 5 MB)", type=["pdf", "docx", "txt"])
        if up is not None:
            if up.size > 5 * 1024 * 1024:
                st.error("That file is over 5 MB.")
            else:
                raw, source, label = svc.read_upload(up.name, up.getvalue()), "upload", up.name
    with t_paste:
        pasted = st.text_area("CV text", height=180, key="paste_cv")
        if pasted.strip() and raw is None:
            raw, source, label = pasted, "upload", "Pasted CV"
    with t_sample:
        samples = svc.sample_profiles()
        pick = st.selectbox("Kaggle resume (anonymized, public)", [""] + samples["doc_id"].tolist(),
                            format_func=lambda d: "— choose —" if not d else
                            f"{d} · {samples.set_index('doc_id').at[d, 'category']}")
        if pick and raw is None:
            row = samples.set_index("doc_id").loc[pick]
            raw, source, doc_id, label = row["text"], "kaggle_resume_dataset", pick, f"{pick} ({row['category']})"
    c1, c2 = st.columns([3, 1])
    fast = c2.toggle("Fast parse (CRF)", value=True,
                     help="Off = LLM parser on local Ollama: better extraction, 1–3 minutes on a laptop.")
    if c1.button("Parse CV", type="primary", disabled=raw is None):
        p = svc.new_profile(raw, source=source, doc_id=doc_id, label=label)
        with st.spinner("Parsing locally and adding the CV to the knowledge graph…"):
            try:
                svc.parse_profile(p, fast)
            except Exception as e:
                st.error(f"Parsing failed: {type(e).__name__}: {e}")
                st.stop()
        st.session_state["profile"] = p
        for k in ("matches", "explain", "outreach", "chat"):
            st.session_state.pop(k, None)
        st.rerun()

p = profile()
if not p or not p.get("parsed"):
    empty_state("Load a CV to see what the parser found.")
    st.stop()

privacy_note(p)
parsed = p["parsed"]
skills, exps = parsed.get("skills", []), parsed.get("experiences", [])
creds, projs = parsed.get("credentials", []), parsed.get("projects", [])
cand = parsed.get("candidate") or {}
m = st.columns(5)
for col, (label, n) in zip(m, [("Skills", len(skills)), ("Roles", len(exps)), ("Credentials", len(creds)),
                               ("Projects", len(projs)), ("Graph edges added", p.get("graph_stats", {}).get("edges",
                                                                                                         "—"))]):
    col.metric(label, n)
st.markdown(f"**Source:** {SOURCE_LABEL.get(p['source'], p['source'])} · **Parser:** {p.get('parser')} · "
            f"{p.get('masked', 0)} contact details masked"
            + (f" · **Headline:** {cand['headline']}" if cand.get("headline") else ""), unsafe_allow_html=True)


def quote(item) -> str:
    return ((item.get("evidence") or {}).get("quote") or "")[:160]


tabs = st.tabs(["Skills", "Experience", "Credentials", "Projects", "Matching skills", "Text"])
with tabs[0]:
    if not skills:
        st.caption("No skills found.")
    else:
        st.dataframe(pd.DataFrame([{"skill": s["name"], "type": s.get("skill_type"), "evidence": quote(s)}
                                   for s in skills]), hide_index=True, width="stretch")
with tabs[1]:
    if not exps:
        st.caption("No roles found.")
    for x in exps:
        with st.container(border=True):
            dates = " – ".join(str(v) for v in (x.get("start_date"), x.get("end_date")) if v)
            st.markdown(f"**{x.get('title') or 'Untitled role'}**" + (f" · {x['company']}" if x.get("company") else "")
                        + (f"  \n<small class='om-muted'>{dates}</small>" if dates else ""), unsafe_allow_html=True)
            if x.get("skills_used"):
                st.markdown(chips_html(x["skills_used"], "info"), unsafe_allow_html=True)
            if quote(x):
                st.caption(f"“{quote(x)}”")
with tabs[2]:
    if not creds:
        st.caption("No credentials found.")
    for c in creds:
        st.markdown(f"- **{c['name']}**" + (f" — {c.get('credential_type')}" if c.get("credential_type") else ""))
with tabs[3]:
    if not projs:
        st.caption("No projects found.")
        if "CRF" in str(p.get("parser")) and re.search(r"\bprojects?\b", p["text"], re.I):
            st.info("Your CV mentions projects, but the fast CRF parser rarely finds them. Turn off **Fast parse "
                    "(CRF)** under *Load a different CV* and parse again: the local LLM parser extracts projects.")
    for x in projs:
        st.markdown(f"**{x['name']}**")
        if x.get("skills_used"):
            st.markdown(chips_html(x["skills_used"], "info"), unsafe_allow_html=True)
with tabs[4]:
    st.caption("The ranking uses these normalized skills (CRF + O*NET). Remove wrong ones or add skills you have; "
               "added skills are marked as your claim, not CV evidence.")
    q = svc.query_for(p)
    base = sorted(p["query"]["skills"])
    keep = st.multiselect("Skills used for matching", base, default=[s for s in base if s not in p["excluded_skills"]],
                          format_func=svc.skill_label)
    added = st.text_input("Add skills (comma-separated)", ", ".join(p.get("added_skills", [])))
    if st.button("Save skill corrections"):
        p["excluded_skills"] = [s for s in base if s not in keep]
        p["added_skills"] = [s.strip() for s in added.split(",") if s.strip()]
        st.success("Saved. Matches will use the corrected skills.")
        st.session_state.pop("matches", None)
    st.markdown(f"Occupation family: {badge_html(p['query']['family'] or 'unknown', 'info')}", unsafe_allow_html=True)
    chips([svc.skill_label(s) for s in sorted(q.skills)], "good", empty="No normalized skills.")
with tabs[5]:
    st.text_area("Masked CV text", p["text"], height=420, disabled=True)
