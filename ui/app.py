"""Streamlit UI (R6):  streamlit run ui/app.py

Tabs: Match (upload a CV -> ranked jobs with cited explanations, skill-gap plan, rewrites),
Outreach (company brief + draft message; nothing is sent; approve -> download a follow-up .ics),
Find candidates (job -> best CVs), Router (routes, live model swap), Dashboard (cost / latency from router logs).
Uploaded CVs are private: they are masked, parsed and explained on local Ollama only (D2).
"""

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import harness                                   # noqa: E402
from graph.build import driver                               # noqa: E402
from ingest.text import clean_text, mask_contacts            # noqa: E402
from ranking import live                                     # noqa: E402
from ranking.cascade import Corpus, job_query                # noqa: E402
from router.router import PrivacyError, get_router           # noqa: E402

st.set_page_config(page_title="LinkedIn Assistant", layout="wide")


def read_upload(f) -> str:
    data = f.read()
    if f.name.lower().endswith(".pdf"):
        import pymupdf
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)
    if f.name.lower().endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def evidence_for(edge_ids: list[str]) -> list[dict]:
    with driver().session() as s:
        return s.run("""MATCH ()-[r]->() WHERE r.edge_id IN $ids
                        OPTIONAL MATCH (e:Evidence) WHERE e.evidence_id IN r.evidence_ids
                        RETURN r.edge_id AS edge_id, type(r) AS rel, r.source AS source,
                               collect(DISTINCT e.text)[0..2] AS evidence""", ids=edge_ids).data()


def show_explanation(e: dict):
    llm, tmpl = e["llm"], e["template"]
    for s in llm["sentences"] or tmpl["sentences"]:
        st.markdown(f"- {s['text']} " + " ".join(f"`{c[:6]}`" for c in s["cites"][:4]))
    if not llm["sentences"]:
        st.caption("No LLM sentence passed the critic; showing the template explanation.")
    with st.expander(f"Citations and critic log ({len(llm['rejected'])} sentences rejected)"):
        ids = sorted({c for s in llm["sentences"] + tmpl["sentences"] for c in s["cites"]})
        if ids:
            st.dataframe(pd.DataFrame(evidence_for(ids)), width="stretch", hide_index=True)
        for r in llm["rejected"]:
            st.markdown(f"~~{r['text']}~~ — *{r.get('reason', '')}*")


tab_match, tab_out, tab_cand, tab_router, tab_dash = st.tabs(
    ["Match", "Outreach", "Find candidates", "Router", "Dashboard"])

with tab_match:
    st.header("Match my CV to jobs")
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        pasted = st.text_area("…or paste CV text", height=150)
    with c2:
        fast = st.toggle("Fast parse (CRF, seconds)", value=True,
                         help="Off = LLM parser on local Ollama: better extraction, a few minutes on a laptop.")
        states = st.text_input("States (comma-separated, e.g. AZ, CA; remote jobs always kept)")
        modes = st.multiselect("Work mode", ["On-site", "Remote", "Hybrid"])
        seniority = st.multiselect("Seniority", ["Internship", "Entry level", "Associate", "Mid-Senior level",
                                                 "Director", "Executive"])
        n_explain = st.slider("Jobs to explain", 1, 10, 3)
    if st.button("Find matches", type="primary"):
        raw = read_upload(up) if up else pasted
        text, n_masked = mask_contacts(clean_text(raw))
        if not text.strip():
            st.error("Add a CV first.")
        else:
            cv_id = "upload_" + hashlib.sha1(text.encode()).hexdigest()[:12]
            prefs = {"states": [s.strip().upper() for s in states.split(",") if s.strip()], "work_modes": modes,
                     "seniority": seniority}
            with st.spinner("Parsing, building the graph, ranking and explaining (local models)…"):
                st.session_state["match"] = harness.match_cv(cv_id, text, fast_parse=fast, prefs=prefs,
                                                             n_explain=n_explain)
            st.session_state["cv_id"] = cv_id
            st.caption(f"{n_masked} contact details masked before processing.")
    res = st.session_state.get("match")
    if res:
        r = res["ranking"]
        st.caption(" → ".join(r["stages"]) + f"  ·  ranking {r['timing']['total_s']:.2f}s  ·  agents: " +
                   ", ".join(f"{t['agent']} {t['seconds']}s" for t in res["trace"]))
        expl = {e["job_id"]: e for e in res.get("explanations", [])}
        for row in r["results"]:
            with st.container(border=True):
                st.subheader(f"{row['rank']}. {row.get('title', '')} — {row.get('company', '')}")
                st.caption(f"{row.get('location', '')} · {row.get('work_mode', '')} · {row.get('seniority', '')} · "
                           f"score {row['score']:.3f}")
                if row["doc_id"] in expl:
                    show_explanation(expl[row["doc_id"]])
        for o in res.get("optimizer", []):
            st.header("Skill-gap plan and CV rewrites (top job)")
            if o["upskilling"]:
                st.table(pd.DataFrame(o["upskilling"])[["skill", "pointer"]])
            for w in o["rewrites"]:
                icon = "✅" if w["verdict"] == "approved" else "⛔"
                st.markdown(f"{icon} **Original:** {w['original']}  \n**Rewrite:** {w['rewrite']}  \n*{w['reason']}*")

with tab_out:
    st.header("Research & outreach (drafts only — nothing is sent)")
    res = st.session_state.get("match")
    if not res:
        st.info("Run a match first.")
    else:
        options = {f"{x['rank']}. {x.get('title')} — {x.get('company')}": x["doc_id"] for x in res["ranking"]["results"]}
        job_id = options[st.selectbox("Job", list(options))]
        if st.button("Draft brief and message"):
            with st.spinner("Researching (local model)…"):
                st.session_state["outreach"] = harness.build_outreach_graph().invoke(
                    {"cv_id": st.session_state["cv_id"], "job_id": job_id, "private": True, "approved": False,
                     "trace": []})
        o = st.session_state.get("outreach")
        if o and o["job_id"] == job_id:
            st.json(o["brief"])
            st.text_input("Subject", o["draft"].get("subject", ""), key="subj")
            st.text_area("Message", o["draft"].get("body", ""), height=200, key="body")
            if st.button("Approve and create a follow-up reminder"):
                state = harness.scheduler_agent({**o, "approved": True})
                st.download_button("Download follow-up (.ics)", state["ics"], file_name="follow_up.ics",
                                   mime="text/calendar")

with tab_cand:
    st.header("Best candidates for a job")
    jobs = harness.jobs_corpus()
    title = st.text_input("Search job titles", "Data Analyst")
    hits = jobs.meta[jobs.meta["title"].str.contains(title, case=False, na=False)].head(50)
    if len(hits):
        pick = st.selectbox("Job", hits["doc_id"] + " — " + hits["title"] + " — " + hits["company"].fillna(""))
        if st.button("Rank candidates"):
            cvs = Corpus("cvs")
            out = live.rank(job_query(pick.split(" — ")[0], jobs), cvs, None, k=10)
            st.caption(" → ".join(out["stages"]))
            st.dataframe(pd.DataFrame(out["results"])[["rank", "doc_id", "category", "score", "ce"]], hide_index=True)

with tab_router:
    st.header("Model router")
    router = get_router()
    routes = pd.DataFrame([{"task": t, **{k: v for k, v in r.items() if k in ("provider", "model")}}
                           for t, r in router.cfg["routes"].items()])
    st.dataframe(routes, hide_index=True, width="stretch")
    st.markdown("**Live swap** (this session only; the permanent change is one line in `config.yaml`).")
    task = st.selectbox("Task", list(router.cfg["routes"]))
    choices = [f"{m['provider']}/{m['model']}" for m in router.cfg["benchmark_models"]]
    new = st.selectbox("Model", choices)
    if st.button("Swap"):
        p, m = new.split("/", 1)
        router.cfg["routes"][task] = {"provider": p, "model": m}
        st.success(f"{task} -> {new}")
    st.markdown("**Privacy check**")
    if st.button("Try sending a private CV to a hosted model"):
        try:
            router.chat("test", [{"role": "user", "content": "hi"}], private=True, provider="asu",
                        model="gpt-oss-120b")
        except PrivacyError as e:
            st.warning(f"Blocked by the router: {e}")

with tab_dash:
    st.header("Cost and latency (from router/logs/calls.jsonl)")
    log = Path(get_router().cfg["paths"]["router_log"])
    if log.exists():
        df = pd.DataFrame([json.loads(x) for x in open(log) if x.strip()])
        live_calls = df[~df["cached"]]
        agg = live_calls.groupby(["task", "provider", "model"]).agg(
            calls=("latency_s", "size"), p50_s=("latency_s", "median"),
            p95_s=("latency_s", lambda x: x.quantile(0.95)), prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"), cost_usd=("cost_usd", "sum")).reset_index()
        st.dataframe(agg, hide_index=True, width="stretch")
        st.caption(f"{len(df)} calls logged, {int(df['cached'].sum())} served from cache. Cost uses list prices "
                   "in config.yaml (all runs here are free tier); local models cost 0.")
        st.bar_chart(agg, x="model", y="p50_s")
    else:
        st.info("No router calls logged yet.")
