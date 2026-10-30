"""LangGraph harness (R6): the 7 agents as one fixed state graph; every LLM call goes through the router,
so swapping a model is a one-line change in config.yaml (or a UI override of the in-memory routes).

  match graph:     Parser -> KG-Build -> Matching -> Critic-verified explanations -> Profile Optimizer
  outreach graph:  Research & Outreach -> (user approval in UI) -> Scheduler

Uploaded CVs are private: every call is made with private=True, so the router refuses hosted providers (D2).
"""

import operator
import time
from functools import lru_cache
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents import explainer, optimizer, outreach
from extraction import crf, llm
from graph.build import Builder, driver
from graph.normalize import Normalizer, OccupationNormalizer
from ranking import live
from ranking.cascade import Corpus, new_cv_query
from ranking.graph_features import GraphView
from router.router import get_router


class MatchState(TypedDict, total=False):
    cv_id: str
    cv_text: str
    private: bool
    fast_parse: bool            # True: CRF parser (seconds); False: LLM parser (minutes on a laptop)
    prefs: dict
    k: int
    n_explain: int
    parsed: dict
    parse_stats: dict
    graph_stats: dict
    ranking: dict
    explanations: list
    optimizer: list
    trace: Annotated[list, operator.add]


class OutreachState(TypedDict, total=False):
    cv_id: str
    job_id: str
    private: bool
    approved: bool
    brief: dict
    draft: dict
    ics: str
    trace: Annotated[list, operator.add]


@lru_cache(maxsize=1)
def jobs_corpus() -> Corpus:
    return Corpus("jobs")


@lru_cache(maxsize=1)
def normalizers():
    return Normalizer("modern", get_router()), OccupationNormalizer()


def _timed(name, fn, state):
    t = time.perf_counter()
    out = fn(state)
    out["trace"] = [{"agent": name, "seconds": round(time.perf_counter() - t, 2)}]
    return out


def parser_agent(state: MatchState) -> dict:
    if state.get("fast_parse"):
        parsed, stats = crf.extract_document(state["cv_id"], "CV", state["cv_text"])
    else:
        parsed, stats = llm.extract_document(get_router(), state["cv_id"], "CV", state["cv_text"],
                                             private=state.get("private", True))
    return {"parsed": parsed.model_dump(mode="json"), "parse_stats": stats}


def kg_build_agent(state: MatchState) -> dict:
    docs = pd.DataFrame([{"doc_id": state["cv_id"], "source": "upload", "text": state["cv_text"], "doc_type": "CV"}])
    norm, occ = normalizers()
    b = Builder(docs, norm, occ)
    b.add_doc({"parsed": state["parsed"]})
    b.finalize()
    b.g.write(driver())
    return {"graph_stats": dict(b.stats)}


def matching_agent(state: MatchState) -> dict:
    q = new_cv_query(state["cv_id"], state["cv_text"], state.get("prefs"))
    return {"ranking": live.rank(q, jobs_corpus(), GraphView(), k=state.get("k", 10))}


def ensure_jobs_in_graph(job_ids: list[str]) -> list[str]:
    """Top jobs not yet in the graph are extracted on demand (public postings: extract_job route) and loaded."""
    with driver().session() as s:
        have = {r["j"] for r in s.run("MATCH (j:JobPosting) WHERE j.job_id IN $ids RETURN j.job_id AS j", ids=job_ids)}
    missing = [j for j in job_ids if j not in have]
    if not missing:
        return []
    c = jobs_corpus()
    rows = []
    for j in missing:
        m = c.meta.iloc[c.pos[j]]
        parsed, stats = llm.extract_document(get_router(), j, "JobPosting", c.text[c.pos[j]], private=False,
                                             title=m["title"], company=m["company"])
        rows.append({"parsed": parsed.model_dump(mode="json"), "stats": stats})
    docs = pd.DataFrame([{"doc_id": j, "source": c.meta.iloc[c.pos[j]]["source"], "text": c.text[c.pos[j]],
                          "doc_type": "JobPosting"} for j in missing])
    norm, occ = normalizers()
    b = Builder(docs, norm, occ)
    for r in rows:
        b.add_doc(r)
    b.finalize()
    b.g.write(driver())
    return missing


def critic_explain_agent(state: MatchState) -> dict:
    out = []
    ensure_jobs_in_graph([r["doc_id"] for r in state["ranking"]["results"][: state.get("n_explain", 3)]])
    for r in state["ranking"]["results"][: state.get("n_explain", 3)]:
        ex, facts = explainer.explain(state["cv_id"], r["doc_id"], get_router(), private=state.get("private", True))
        tmpl = explainer.template_explanation(state["cv_id"], r["doc_id"], facts)
        out.append({"job_id": r["doc_id"], "title": r.get("title"), "llm": ex.to_dict(), "template": tmpl.to_dict(),
                    "facts": [f.__dict__ for f in facts]})
    return {"explanations": out}


def optimizer_agent(state: MatchState) -> dict:
    out = []
    for e in state["explanations"][:1]:          # the top job; more on demand in the UI
        facts = [explainer.Fact(**f) for f in e["facts"]]
        out.append({"job_id": e["job_id"], "upskilling": optimizer.upskilling(facts),
                    "rewrites": optimizer.rewrites(get_router(), facts, e.get("title") or "",
                                                   private=state.get("private", True))})
    return {"optimizer": out}


def research_agent(state: OutreachState) -> dict:
    c = jobs_corpus()
    i = c.pos[state["job_id"]]
    return {"brief": outreach.company_brief(get_router(), state["job_id"], c.meta, c.text[i], public=False)}


def outreach_agent(state: OutreachState) -> dict:
    c = jobs_corpus()
    m = c.meta.iloc[c.pos[state["job_id"]]]
    facts = explainer.bundle(state["cv_id"], state["job_id"])
    return {"draft": outreach.outreach_draft(get_router(), m["title"], m["company"], facts,
                                             private=state.get("private", True))}


def scheduler_agent(state: OutreachState) -> dict:
    if not state.get("approved"):
        return {"ics": ""}
    c = jobs_corpus()
    m = c.meta.iloc[c.pos[state["job_id"]]]
    return {"ics": outreach.follow_up_ics(m["title"], m["company"])}


def build_match_graph():
    g = StateGraph(MatchState)
    for name, fn in (("parser", parser_agent), ("kg_build", kg_build_agent), ("matching", matching_agent),
                     ("critic_explain", critic_explain_agent), ("profile_optimizer", optimizer_agent)):
        g.add_node(name, lambda s, n=name, f=fn: _timed(n, f, s))
    g.add_edge(START, "parser")
    g.add_edge("parser", "kg_build")
    g.add_edge("kg_build", "matching")
    g.add_edge("matching", "critic_explain")
    g.add_edge("critic_explain", "profile_optimizer")
    g.add_edge("profile_optimizer", END)
    return g.compile()


def build_outreach_graph():
    g = StateGraph(OutreachState)
    for name, fn in (("research", research_agent), ("outreach", outreach_agent), ("scheduler", scheduler_agent)):
        g.add_node(name, lambda s, n=name, f=fn: _timed(n, f, s))
    g.add_edge(START, "research")
    g.add_edge("research", "outreach")
    g.add_conditional_edges("outreach", lambda s: "scheduler" if s.get("approved") else END)
    g.add_edge("scheduler", END)
    return g.compile()


def match_cv(cv_id: str, text: str, *, fast_parse: bool = True, prefs: dict | None = None, k: int = 10,
             n_explain: int = 3) -> MatchState:
    """Run the full match graph for one CV (always private: uploaded CVs stay on local models)."""
    return build_match_graph().invoke({"cv_id": cv_id, "cv_text": text, "private": True, "fast_parse": fast_parse,
                                       "prefs": prefs or {}, "k": k, "n_explain": n_explain, "trace": []})
