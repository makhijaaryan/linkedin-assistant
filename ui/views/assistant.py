"""Assistant & outreach: grounded Q&A about a match (answers may only use cited graph facts, local model), a company
brief + recruiter message draft (nothing is sent), and a follow-up reminder (.ics) after explicit approval."""

import html

import streamlit as st

from ui import services as svc
from ui.components import badge_html, empty_state, privacy_note, require_profile

st.title("Assistant & outreach")
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


def facts() -> list[dict]:
    if job_id not in cache:
        with st.spinner("Collecting graph facts for this job…"):
            cache[job_id] = svc.explain(p, job_id, use_llm=False)
    return cache[job_id]["facts"]


chat_tab, out_tab = st.tabs(["Ask about this match", "Outreach draft & follow-up"])
with chat_tab:
    st.caption("Answers come from a local model and may only use the numbered graph facts for this job; each claim "
               "cites its fact.")
    hist = st.session_state.setdefault("chat", {}).setdefault(job_id, [])
    for turn in hist:
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("cited"):
                with st.expander("Cited facts"):
                    for i in turn["cited"]:
                        st.markdown(f"**[{i}]** {html.escape(turn['facts'][i - 1]['text'])}")
    examples = ["Why am I a good fit?", "What am I missing for this role?", "Do I have enough experience?"]
    cols = st.columns(len(examples))
    pick = next((ex for c, ex in zip(cols, examples) if c.button(ex, width="stretch")), None)
    q = st.chat_input("Ask about this match…") or pick
    if q:
        fs = facts()
        hist.append({"role": "user", "text": q})
        with st.spinner("Thinking locally…"):
            try:
                a = svc.ask(p, job_id, fs, q)
                hist.append({"role": "assistant", "text": a["answer"] + f"\n\n<small>{a['model']} · "
                             f"{a['latency_s']:.1f}s</small>", "cited": a["cited"], "facts": fs})
            except Exception as e:
                hist.append({"role": "assistant", "text": f"The local model is unavailable ({type(e).__name__})."})
        st.rerun()

with out_tab:
    st.caption("Drafts only: nothing is ever sent. You approve before a reminder is created.")
    if st.button("Draft company brief and message", type="primary"):
        facts()
        with st.spinner("Researching the company and drafting (local model)…"):
            st.session_state["outreach"] = svc.outreach_pack(p, job_id)
    o = st.session_state.get("outreach")
    if o and o["job_id"] == job_id:
        b = o["brief"]
        c1, c2 = st.columns([1, 2])
        with c1, st.container(border=True):
            st.markdown(f"**{b.get('company') or 'Company'}**")
            st.metric("Postings in our corpus", b.get("postings_in_corpus", 0))
            if b.get("titles"):
                st.caption("Common titles: " + ", ".join(list(b["titles"])[:4]))
        with c2, st.container(border=True):
            st.markdown("**About the role** " + badge_html("from the posting text", "info"), unsafe_allow_html=True)
            for x in b.get("summary") or ["No summary (the local model is busy or unavailable)."]:
                st.markdown(f"- {x}")
        d = o["draft"]
        if not d.get("body"):
            st.warning("No draft: the graph has no matched facts to mention, and the assistant never invents any.")
        else:
            st.text_input("Subject", d.get("subject", ""), key="subj")
            st.text_area("Message", d.get("body", ""), height=200, key="body")
            st.caption(f"Grounded in {len(d.get('cites', []))} graph edges · {d.get('status', '')}")
            days = st.slider("Remind me to follow up in (days)", 3, 21, 7)
            if st.checkbox("I approve this draft (I'll send it myself)"):
                st.download_button("Download follow-up reminder (.ics)", svc.follow_up(job_id, days),
                                   file_name="follow_up.ics", mime="text/calendar")
