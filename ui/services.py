"""The UI's single entry point into the pipeline (same process: no API layer).

Everything a page needs (parse a CV, rank jobs with any method, explain a match, query the graph, read results)
goes through here, so the pages stay thin and the privacy rule (D2) is enforced in one place: uploaded CVs are
always `private=True`, which makes the router refuse every hosted provider.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agents import explainer, harness, optimizer, outreach
from graph.build import driver
from graph.normalize import clean, taxonomy
from ingest import adzuna
from ingest.text import clean_text, mask_contacts
from ranking import live
from ranking.cascade import Corpus, Query, job_query, minmax, new_cv_query
from ranking.graph_features import GraphView
from router.router import get_router

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval/results"
METHODS = {"cascade": "Full cascade (RRF → LambdaMART → cross-encoder → MMR)", "rrf": "RRF (lexical + dense fusion)",
           "classical": "Classical (BM25 + TF-IDF + Jaccard, cluster-diverse)", "bm25": "BM25", "tfidf": "TF-IDF cosine",
           "jaccard": "Jaccard skill overlap", "dense": "Dense (MiniLM bi-encoder)"}


# ---------------------------------------------------------------------------------------------- corpora & lookups
@st.cache_resource(show_spinner="Loading the job index…")
def jobs() -> Corpus:
    return harness.jobs_corpus()


@st.cache_resource(show_spinner="Loading the candidate index…")
def cvs() -> Corpus:
    return Corpus("cvs")


@st.cache_resource
def skill_names() -> dict:
    t = taxonomy()
    return dict(zip(t["taxonomy_id"], t["canonical_name"]))


def corpus_for(job_id: str) -> Corpus:
    """Live Adzuna postings (adz_*) or the Kaggle index."""
    return harness.corpus_for(job_id) if job_id.startswith("adz_") else jobs()


def skill_label(sid: str) -> str:
    return sid.split(":", 1)[1] if sid.startswith("free:") else skill_names().get(sid, sid)


@st.cache_data
def resumes_meta() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/resumes.parquet", columns=["doc_id", "source", "category", "text"])


def sample_profiles() -> pd.DataFrame:
    """Public Kaggle ranking profiles (anonymized) for demos; resume_corpus profiles are never shown (D24)."""
    ids = json.loads((ROOT / "eval/frozen/test_sets.json").read_text())["ranking_profiles"]
    m = resumes_meta().set_index("doc_id").loc[ids].reset_index()
    return m[m["source"] == "kaggle_resume_dataset"]


# ---------------------------------------------------------------------------------------------- profile (CV)
def read_upload(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        import pymupdf
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)
    if name.lower().endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def new_profile(raw: str, *, source: str, doc_id: str | None = None, label: str = "") -> dict:
    text, n_masked = mask_contacts(clean_text(raw))
    h = hashlib.sha1(text.encode()).hexdigest()[:12]
    cv_id = f"demo_{doc_id}" if doc_id else f"upload_{h}"
    return {"cv_id": cv_id, "text": text, "source": source, "label": label or cv_id, "masked": n_masked,
            "private": source != "kaggle_resume_dataset", "parsed": None, "excluded_skills": [], "added_skills": []}


def parse_profile(p: dict, fast: bool) -> dict:
    """Parser + KG-Build agents. Uploaded CVs are parsed locally (CRF, or the LLM on Ollama)."""
    st_parser = harness.parser_agent({"cv_id": p["cv_id"], "cv_text": p["text"], "fast_parse": fast,
                                      "private": True})
    graph = harness.kg_build_agent({"cv_id": p["cv_id"], "cv_text": p["text"], "parsed": st_parser["parsed"]})
    p.update(parsed=st_parser["parsed"], parse_stats=st_parser["parse_stats"], graph_stats=graph["graph_stats"],
             parser="CRF (classical)" if fast else "LLM on local Ollama")
    p.pop("query", None)
    return p


def query_for(p: dict, prefs: dict | None = None) -> Query:
    if "query" not in p:
        q = new_cv_query(p["cv_id"], p["text"])
        p["query"] = {"skills": sorted(q.skills), "family": q.family}
    skills = (set(p["query"]["skills"]) - set(p.get("excluded_skills", []))) | {f"free:{clean(s)}" for s in
                                                                               p.get("added_skills", [])}
    return Query(p["cv_id"], p["text"], skills, p["query"]["family"], prefs=prefs or {})


# ---------------------------------------------------------------------------------------------- ranking
@st.cache_data(max_entries=5000, show_spinner=False)
def job_skills(job_id: str) -> frozenset:
    """Normalized skills of a posting: the index's weak labels, or (56% of postings have none) CRF + O*NET now."""
    c = corpus_for(job_id)
    have = c.meta.iloc[c.pos[job_id]]["skills"]
    if have:
        return frozenset(have)
    from extraction.crf import predict_spans
    text = c.text[c.pos[job_id]]
    names = sorted({text[s:e] for s, e, lab, _ in predict_spans(text) if lab == "Skill"})
    return frozenset(m.taxonomy_id or f"free:{clean(n)}" for n, m in zip(names, _classical().normalize(names)))


@st.cache_resource
def _classical():
    from graph.normalize import Normalizer
    return Normalizer("classical")


def overlap(q: Query, job_skills: set) -> dict:
    matched = sorted(q.skills & job_skills)
    missing = sorted(job_skills - q.skills)
    return {"matched": [skill_label(s) for s in matched], "missing": [skill_label(s) for s in missing],
            "coverage": (len(matched) / len(job_skills)) if job_skills else None}


def adzuna_search(what: str, where: str, pages: int = 1, max_days: int | None = None) -> list[str]:
    """Live postings for a keyword search (cached; one Adzuna call per uncached page). Returns their doc_ids."""
    return adzuna.search(what, where, pages, max_days)["doc_id"].tolist()


def posted_label(created: str) -> str:
    """Adzuna's ISO `created` timestamp -> 'Posted Oct 28, 2026 (3 days ago)'."""
    t = pd.to_datetime(created, utc=True, errors="coerce")
    if pd.isna(t):
        return ""
    days = (pd.Timestamp.now(tz="UTC") - t).days
    ago = "today" if days <= 0 else "1 day ago" if days == 1 else f"{days} days ago"
    return f"Posted {t:%b %-d, %Y} ({ago})"


_TITLE_NOISE = re.compile(r"\b(intern(ship)?|trainee|co-?op|sr|senior|jr|junior|lead|principal|staff|graduate|"
                          r"student|i{1,3}|iv|[0-9]+)\b\.?", re.I)
_DATE_ORDER = lambda e: (bool(e.get("is_current")) or str(e.get("end_date") or "").lower() == "present",
                         str(e.get("end_date") or ""), str(e.get("start_date") or ""))


def adzuna_keywords(p: dict) -> dict:
    """Search terms suggested from the parsed CV: recent role titles without seniority words, the headline and the
    predicted job family (the first two-word option is the default), plus the location's US state. The user picks
    or edits them before any search."""
    parsed = p.get("parsed") or {}
    exps = sorted([e for e in parsed.get("experiences", []) if e.get("title")], key=_DATE_ORDER, reverse=True)
    cand = parsed.get("candidate") or {}
    options = {}
    for text in [e["title"] for e in exps[:4]] + [cand.get("headline") or ""]:
        words = " ".join(_TITLE_NOISE.sub(" ", re.split(r"[|,/(@–-]| at ", text)[0]).split()[:4]).lower()
        if words and words not in options and re.fullmatch(r"[a-z&+# ]+", words) and len(text.split()) <= 8:
            options[words] = None
    family = query_for(p).family if p.get("parsed") else None
    if family and family.lower() not in options:
        options[family.lower()] = None
    options = list(options)
    what = next((o for o in options if len(o.split()) >= 2), options[0] if options else "")
    loc = cand.get("location") or next((e["location"] for e in exps if e.get("location")), None) or {}
    state = (loc.get("state") or "").strip()
    where = {v: k for k, v in adzuna.US_STATES.items()}.get(state.upper(), state if state in adzuna.US_STATES else "")
    return {"what": what, "where": where, "options": options}


def adzuna_calls_used() -> int:
    return adzuna.calls_used()


def rank(p: dict, prefs: dict, method: str = "cascade", k: int = 10, live_ids: list[str] | None = None) -> dict:
    """Rank the Kaggle index, or (live_ids given) only those live Adzuna postings."""
    if live_ids is not None:
        from ranking import live_jobs
        corpus = live_jobs.corpus()
        prefs = {**prefs, "doc_ids": live_ids}
    else:
        corpus = jobs()
    q = query_for(p, prefs)
    if method == "cascade":
        out = live.rank(q, corpus, GraphView(), k=k)
    else:
        s = corpus.first_stage(q)
        mask = corpus.filter_mask(q)
        if method == "classical":
            idx = corpus.candidates(q, s, depth=200, by=["bm25", "tfidf", "jaccard", "dense"])
            picked = corpus.classical(q, s, idx, k=k)
            score = np.mean([minmax(s[n][idx]) for n in ("bm25", "tfidf", "jaccard")], axis=0)
            sc = dict(zip(idx.tolist(), score))
            order = [(i, float(sc.get(i, 0))) for i in picked]
        else:
            v = np.where(mask, s[method], -np.inf)
            top = np.argsort(-v)[:k]
            order = [(int(i), float(v[i])) for i in top if np.isfinite(v[i])]
        rows = []
        for r, (i, score) in enumerate(order):
            m = corpus.meta.iloc[i]
            rows.append({"rank": r + 1, "doc_id": m["doc_id"], "score": score,
                         **{c: m[c] for c in ("title", "company", "location", "work_mode", "seniority") if c in m}})
        out = {"results": rows, "stages": [f"filters → {METHODS[method]} top {len(rows)}"], "timing": {}}
    for r in out["results"]:
        i = corpus.pos[r["doc_id"]]
        r.update(overlap(q, set(job_skills(r["doc_id"]))))
        r["industry"] = corpus.meta.iloc[i].get("industry")
        if live_ids is not None:
            r["posted"] = posted_label(corpus.meta.iloc[i].get("created", ""))
    out["method"] = method
    out["source"] = "adzuna" if live_ids is not None else "kaggle"
    out["filtered_pool"] = int(corpus.filter_mask(q).sum())
    return out


def job_text(job_id: str) -> str:
    c = corpus_for(job_id)
    return c.text[c.pos[job_id]]


def job_meta(job_id: str) -> dict:
    c = corpus_for(job_id)
    return c.meta.iloc[c.pos[job_id]].to_dict()


# ---------------------------------------------------------------------------------------------- explanations
def explain(p: dict, job_id: str, use_llm: bool) -> dict:
    """Critic-verified explanation for one (CV, job) pair. The job is parsed into the graph on demand."""
    harness.ensure_jobs_in_graph([job_id])
    facts = explainer.bundle(p["cv_id"], job_id)
    tmpl = explainer.template_explanation(p["cv_id"], job_id, facts)
    out = {"job_id": job_id, "facts": [f.__dict__ for f in facts], "template": tmpl.to_dict(), "llm": None}
    if use_llm and facts:
        ex = explainer.llm_explanation(get_router(), p["cv_id"], job_id, facts, private=True)
        out["llm"] = ex.to_dict()
    return out


def evidence_for(edge_ids: list[str]) -> pd.DataFrame:
    with driver().session() as s:
        rows = s.run("""MATCH ()-[r]->() WHERE r.edge_id IN $ids
                        OPTIONAL MATCH (e:Evidence) WHERE e.evidence_id IN r.evidence_ids
                        RETURN r.edge_id AS edge_id, type(r) AS relation, r.source AS source,
                               collect(DISTINCT e.text)[0..2] AS evidence""", ids=edge_ids).data()
    return pd.DataFrame(rows)


def gap_plan(facts: list[dict]) -> list[dict]:
    return optimizer.upskilling([explainer.Fact(**f) for f in facts])


def rewrites(facts: list[dict], title: str) -> list[dict]:
    return optimizer.rewrites(get_router(), [explainer.Fact(**f) for f in facts], title, private=True)


def outreach_pack(p: dict, job_id: str) -> dict:
    c = corpus_for(job_id)
    m = c.meta.iloc[c.pos[job_id]]
    brief = outreach.company_brief(get_router(), job_id, c.meta, c.text[c.pos[job_id]], public=False)
    facts = explainer.bundle(p["cv_id"], job_id)
    draft = outreach.outreach_draft(get_router(), m["title"], m["company"], facts, private=True)
    return {"job_id": job_id, "brief": brief, "draft": draft}


def follow_up(job_id: str, days: int) -> str:
    m = job_meta(job_id)
    return outreach.follow_up_ics(m["title"], m["company"], days=days)


ASK_PROMPT = """You are a careful career assistant. Answer the user's question about their job match using ONLY the
numbered facts below (they come from a knowledge graph built from the CV and the job posting). Cite every claim with
the fact number in square brackets, like [2]. If the facts don't answer the question, say so plainly.
Return JSON: {{"answer": "...", "cited": [numbers]}}

Job: {title} at {company}
Facts:
{facts}

Question: {question}"""


def ask(p: dict, job_id: str, facts: list[dict], question: str) -> dict:
    """Grounded Q&A on local Ollama (route `chat`): answers may only use the cited graph facts."""
    m = job_meta(job_id)
    numbered = "\n".join(f"[{i + 1}] {f['text']}" for i, f in enumerate(facts)) or "(no facts)"
    prompt = ASK_PROMPT.format(title=m["title"], company=m["company"], facts=numbered, question=question)
    r = get_router().chat("chat", [{"role": "user", "content": prompt}], private=True)
    try:
        out = json.loads(r.text)
    except json.JSONDecodeError:
        out = {"answer": r.text, "cited": []}
    cited = [int(c) for c in out.get("cited", []) if str(c).isdigit() and 0 < int(c) <= len(facts)]
    return {"answer": out.get("answer", ""), "cited": cited, "latency_s": r.latency_s, "model": r.model}


# ---------------------------------------------------------------------------------------------- candidates
def rank_candidates(job_id: str, k: int = 10) -> dict:
    out = live.rank(job_query(job_id, jobs()), cvs(), None, k=k)
    src = resumes_meta().set_index("doc_id")["source"]
    for r in out["results"]:
        r["source"] = src.get(r["doc_id"])
    return out


# ---------------------------------------------------------------------------------------------- graph
def subgraph(cv_id: str, job_id: str | None, labels: list[str], limit: int = 150) -> dict:
    """Candidate (and optionally one job) with its 1–2 hop neighbourhood, Evidence nodes excluded."""
    rows = []
    with driver().session() as s:
        for rid in [cv_id] + ([job_id] if job_id else []):
            # hop 1: everything on the root; hop 2: only the ontology links that explain a match (the O*NET
            # TYPICAL_FOR taxonomy would otherwise flood the picture)
            rows += s.run("""
                MATCH (root) WHERE (root:Candidate AND root.candidate_id = $id) OR (root:JobPosting AND root.job_id = $id)
                MATCH (root)-[r1]-(a) WHERE NOT a:Evidence
                OPTIONAL MATCH (a)-[r2:TARGETS|AS_ROLE|RELATED_TO|AT|FOR_ROLE]-(b) WHERE NOT b:Evidence AND b <> root
                WITH collect(DISTINCT r1)[0..$lim] + collect(DISTINCT r2)[0..$lim] AS rels
                UNWIND rels AS r
                WITH DISTINCT r
                RETURN elementId(startNode(r)) AS src, labels(startNode(r))[0] AS src_label,
                       properties(startNode(r)) AS sp, elementId(endNode(r)) AS dst, labels(endNode(r))[0] AS dst_label,
                       properties(endNode(r)) AS dp, type(r) AS type, properties(r) AS props""",
                          id=rid, lim=limit).data()
    nodes, edges = {}, []
    for r in rows:
        if r["src_label"] not in labels or r["dst_label"] not in labels:
            continue
        nodes[r["src"]] = {"id": r["src"], "label": r["src_label"], **r["sp"]}
        nodes[r["dst"]] = {"id": r["dst"], "label": r["dst_label"], **r["dp"]}
        edges.append({"src": r["src"], "dst": r["dst"], "type": r["type"], "props": r["props"]})
    keep = list(nodes)[:limit]
    edges = [e for e in edges if e["src"] in keep and e["dst"] in keep]
    return {"nodes": [nodes[k] for k in keep], "edges": edges}


def node_evidence(element_id: str) -> pd.DataFrame:
    with driver().session() as s:
        rows = s.run("""MATCH (n)-[r]-(m) WHERE elementId(n) = $id AND r.evidence_ids IS NOT NULL
                        UNWIND r.evidence_ids AS eid
                        MATCH (e:Evidence {evidence_id: eid})
                        RETURN type(r) AS relation, coalesce(m.name, m.canonical_name, m.title, m.text) AS other,
                               r.source AS extractor, e.text AS evidence LIMIT 25""", id=element_id).data()
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def graph_counts() -> dict:
    try:
        with driver().session() as s:
            n = {r["l"]: r["c"] for r in s.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c")}
            e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": n, "edges": e, "ok": True}
    except Exception as ex:
        return {"nodes": {}, "edges": 0, "ok": False, "error": str(ex)[:120]}


# ---------------------------------------------------------------------------------------------- results & logs
def result(name: str) -> dict | None:
    """A result file; for label-dependent results, the human-labeled version wins, else Claude-labeled (D30)."""
    stem = name.removesuffix(".json")
    for cand in (name, f"{stem}_llm.json", f"{stem}_llm_label.json"):
        p = RESULTS / cand
        if p.exists():
            out = json.loads(p.read_text())
            if cand != name and isinstance(out, dict):
                out["_labels_note"] = "Claude-judged labels (D30)."
            return out
    return None


@st.cache_data(ttl=30)
def router_log() -> pd.DataFrame:
    log = ROOT / get_router().cfg["paths"]["router_log"]
    if not log.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(x) for x in open(log) if x.strip()])


def _progress_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("progress", ROOT / "scripts/progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(ttl=60)
def benchmark_progress() -> pd.DataFrame:
    parts = _progress_module().run_parts()
    return pd.DataFrame([{"stream": k, "docs done": a, "planned": b, "progress": a / b if b else 0}
                         for k, (a, b) in parts.items()])


@st.cache_data(ttl=60)
def progress() -> dict:
    mod = _progress_module()
    out = {}
    for k, fn in (("runs", mod.runs), ("labels", mod.labels), ("frontend", mod.frontend), ("train_eval", mod.train_eval)):
        try:
            out[k] = fn()
        except Exception as e:
            out[k] = (0.0, f"unavailable ({type(e).__name__})")
    out["overall"] = sum(mod.WEIGHTS[k] * out[k][0] for k in mod.WEIGHTS)
    return out

