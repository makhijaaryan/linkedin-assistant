"""Grounded match explanations (R6 / RQ4): evidence bundle -> explainer -> critic.

1. `bundle()` collects, with Cypher, only facts the graph can cite: requirements the candidate meets
   (with the CV evidence quote), unmet required skills, RELATED_TO bridges to unmet skills,
   experience vs. required years, and occupation matches. Every fact carries the edge_ids behind it.
2. `template_explanation()` is the classical baseline: fixed sentences straight from the bundle.
3. `llm_explanation()` asks the explain route to write sentences ONLY from the bundle, each citing edge ids.
4. `critic()` rejects any sentence with no citation, an edge id outside the bundle, or (LLM check,
   route critic) a claim its cited facts don't support. Rejected sentences trigger one regeneration;
   whatever still fails is dropped, so the user only ever sees cited, verified sentences.
"""

import json
import re
from dataclasses import asdict, dataclass, field

from graph.build import driver



@dataclass
class Fact:
    kind: str                      # met | gap | bridge | experience | occupation
    text: str
    edge_ids: list[str]


@dataclass
class Explanation:
    candidate_id: str
    job_id: str
    method: str
    sentences: list[dict] = field(default_factory=list)   # {"text", "cites": [...], "verdict": "supported"|...}
    rejected: list[dict] = field(default_factory=list)
    rounds: int = 0

    def to_dict(self):
        return asdict(self)


def bundle(candidate_id: str, job_id: str, drv=None, max_facts: int = 25) -> list[Fact]:
    drv = drv or driver()
    facts: list[Fact] = []
    with drv.session() as s:
        met = s.run("""
            MATCH (j:JobPosting {job_id: $j})-[hr:HAS_REQUIREMENT]->(q:Requirement)-[t:TARGETS]->(k:Skill)
                  <-[hs:HAS_SKILL]-(c:Candidate {candidate_id: $c})
            MATCH (ev:Evidence) WHERE ev.evidence_id IN hs.evidence_ids
            RETURN q.text AS req, q.importance AS imp, k.canonical_name AS skill, hr.edge_id AS e1, t.edge_id AS e2,
                   hs.edge_id AS e3, collect(ev.text)[0] AS quote ORDER BY imp DESC LIMIT $n""",
                    j=job_id, c=candidate_id, n=max_facts).data()
        by_skill = {}
        for r in met:                              # one fact per skill: its most specific requirement
            cur = by_skill.get(r["skill"])
            if cur is None or (r["imp"], len(r["req"])) > (cur["imp"], len(cur["req"])):
                by_skill[r["skill"]] = r
        for r in by_skill.values():
            facts.append(Fact("met", f"The posting's {r['imp'].lower()} requirement \"{r['req']}\" asks for "
                                     f"{r['skill']}; the CV states \"{r['quote']}\".", [r["e1"], r["e2"], r["e3"]]))
        gaps = s.run("""
            MATCH (j:JobPosting {job_id: $j})-[hr:HAS_REQUIREMENT]->(q:Requirement {importance: 'Required'})
                  -[t:TARGETS]->(k:Skill)
            WHERE NOT EXISTS { MATCH (:Candidate {candidate_id: $c})-[:HAS_SKILL]->(k) }
            RETURN q.text AS req, k.canonical_name AS skill, k.skill_id AS sid, hr.edge_id AS e1, t.edge_id AS e2
            LIMIT $n""", j=job_id, c=candidate_id, n=max_facts).data()
        gaps = list({r["sid"]: r for r in gaps}.values())
        for r in gaps:
            facts.append(Fact("gap", f"The required \"{r['req']}\" asks for {r['skill']}, which the CV does not mention.",
                              [r["e1"], r["e2"]]))
        bridges = s.run("""
            MATCH (j:JobPosting {job_id: $j})-[:HAS_REQUIREMENT]->(:Requirement {importance: 'Required'})
                  -[t:TARGETS]->(k:Skill)-[rel:RELATED_TO]-(k2:Skill)<-[hs:HAS_SKILL]-(c:Candidate {candidate_id: $c})
            WHERE NOT EXISTS { MATCH (c)-[:HAS_SKILL]->(k) }
            RETURN k.canonical_name AS missing, k2.canonical_name AS have, t.edge_id AS e1, rel.edge_id AS e2,
                   hs.edge_id AS e3, rel.source AS src LIMIT $n""", j=job_id, c=candidate_id, n=max_facts).data()
        for r in bridges:
            facts.append(Fact("bridge", f"{r['missing']} (required, missing) is related to {r['have']}, which the "
                                        f"candidate has ({r['src']} link).", [r["e1"], r["e2"], r["e3"]]))
        exp = s.run("""
            MATCH (j:JobPosting {job_id: $j})-[hr:HAS_REQUIREMENT]->(q:Requirement) WHERE q.min_years IS NOT NULL
            WITH q, hr ORDER BY q.min_years DESC LIMIT 1
            MATCH (c:Candidate {candidate_id: $c})
            OPTIONAL MATCH (c)-[h:HELD]->(x:Experience)
            RETURN q.text AS req, q.min_years AS need, hr.edge_id AS e1, c.total_years_experience AS total,
                   sum(x.duration_years) AS summed, collect(h.edge_id) AS held""", j=job_id, c=candidate_id).single()
        if exp:
            have = exp["total"] if exp["total"] is not None else exp["summed"]
            if have:
                verdict = "meets" if have >= exp["need"] else "falls short of"
                facts.append(Fact("experience", f"The posting requires {exp['need']:g}+ years (\"{exp['req']}\"); the CV's "
                                                f"listed roles add up to about {have:.1f} years, which {verdict} it.",
                                  [exp["e1"], *exp["held"][:5]]))
        occ = s.run("""
            MATCH (j:JobPosting {job_id: $j})-[f:FOR_ROLE]->(o:Occupation)<-[a:AS_ROLE]-(x:Experience)
                  <-[h:HELD]-(:Candidate {candidate_id: $c})
            RETURN o.title AS occ, x.title AS role, f.edge_id AS e1, a.edge_id AS e2, h.edge_id AS e3 LIMIT 3""",
                    j=job_id, c=candidate_id).data()
        for r in occ:
            facts.append(Fact("occupation", f"The job and the candidate's past role \"{r['role']}\" both map to the "
                                            f"O*NET occupation {r['occ']}.", [r["e1"], r["e2"], r["e3"]]))
    return facts


def template_explanation(candidate_id: str, job_id: str, facts: list[Fact]) -> Explanation:
    """Classical baseline: one sentence per fact type, matched and missing skills listed with citations."""
    ex = Explanation(candidate_id, job_id, "template")
    met = [f for f in facts if f.kind == "met"]
    gaps = [f for f in facts if f.kind == "gap"]
    skill = lambda f: re.search(r"asks for (.+?)(?:;|, which)", f.text).group(1)
    if met:
        ex.sentences.append({"text": "Matched skills: " + ", ".join(sorted({skill(f) for f in met})) + ".",
                             "cites": sorted({e for f in met for e in f.edge_ids})})
    if gaps:
        ex.sentences.append({"text": "Missing required skills: " + ", ".join(sorted({skill(f) for f in gaps})) + ".",
                             "cites": sorted({e for f in gaps for e in f.edge_ids})})
    for f in facts:
        if f.kind in ("experience", "occupation", "bridge"):
            ex.sentences.append({"text": f.text, "cites": f.edge_ids})
    for s in ex.sentences:
        s["verdict"] = "supported"
    return ex


EXPLAIN_PROMPT = """Explain to a job seeker how well their CV matches this job. Use ONLY the facts below; each
fact has an id in brackets. Write 3-6 short sentences: strengths first, then gaps, then one concrete next step
that follows from a gap or bridge. Every sentence must cite the ids of the facts it uses, e.g. [3] or [1][4].
Do not mention anything that is not in the facts. If no fact is a gap, do not claim any gap.
Facts:
{facts}
Reply JSON {{"sentences": [{{"text": "...", "cites": [fact ids]}}]}}."""

CRITIC_PROMPT = """You check an explanation sentence against the facts it cites. Answer whether every claim in
the sentence is stated or directly implied by the cited facts (a suggested next step is fine if it follows
from a cited gap).
Sentence: "{sentence}"
Cited facts:
{facts}
Reply JSON {{"supported": true|false, "reason": "<short>"}}."""


def _numbered(facts):
    return "\n".join(f"[{i}] {f.text}" for i, f in enumerate(facts))


def llm_explanation(router, candidate_id: str, job_id: str, facts: list[Fact], *, private: bool,
                    provider=None, model=None, max_rounds: int = 2) -> Explanation:
    ex = Explanation(candidate_id, job_id, f"llm:{provider or router.route('explain')[0]}/{model or router.route('explain')[1]}")
    if not facts:
        return ex
    kw = dict(private=private, provider=provider, model=model)
    feedback, keep = "", []
    for rnd in range(max_rounds):
        ex.rounds = rnd + 1
        prompt = EXPLAIN_PROMPT.format(facts=_numbered(facts)) + feedback
        try:
            out = json.loads(router.chat("explain", [{"role": "user", "content": prompt}], run=rnd, **kw).text)
            draft = out.get("sentences", []) if isinstance(out, dict) else []
        except (json.JSONDecodeError, AttributeError):
            draft = []
        sentences = []
        for s in draft:
            if not isinstance(s, dict) or not s.get("text"):
                continue
            idx = [int(c) for c in s.get("cites", []) if str(c).isdigit()]
            sentences.append({"text": s["text"], "fact_idx": idx,
                              "cites": sorted({e for i in idx if 0 <= i < len(facts) for e in facts[i].edge_ids}),
                              "bad_ids": [i for i in idx if not 0 <= i < len(facts)]})
        verdicts = critic(router, sentences, facts, private=private)
        keep = [s for s, v in zip(sentences, verdicts) if v["verdict"] == "supported"]
        bad = [dict(s, **v) for s, v in zip(sentences, verdicts) if v["verdict"] != "supported"]
        ex.rejected += bad
        if not bad:
            break
        feedback = "\nYour previous draft had unsupported sentences; rewrite without them:\n" + \
                   "\n".join(f"- \"{b['text']}\" ({b['reason']})" for b in bad)
    ex.sentences = [{"text": s["text"], "cites": s["cites"], "verdict": "supported"} for s in keep]
    return ex


def critic(router, sentences: list[dict], facts: list[Fact], *, private: bool) -> list[dict]:
    """Rule check (citations exist and are in the bundle), then an LLM entailment check per sentence."""
    out = []
    for s in sentences:
        if not s.get("fact_idx"):
            out.append({"verdict": "rejected", "reason": "no citation"})
            continue
        if s.get("bad_ids"):
            out.append({"verdict": "rejected", "reason": f"cites ids not in the bundle: {s['bad_ids']}"})
            continue
        if router is None:
            out.append({"verdict": "supported", "reason": "rule check only"})
            continue
        cited = "\n".join(f"[{i}] {facts[i].text}" for i in s["fact_idx"])
        try:
            v = json.loads(router.chat("critic", [{"role": "user", "content": CRITIC_PROMPT.format(
                sentence=s["text"], facts=cited)}], private=private).text)
            ok = bool(v.get("supported"))
            out.append({"verdict": "supported" if ok else "rejected", "reason": v.get("reason", "")})
        except Exception as e:
            out.append({"verdict": "rejected", "reason": f"critic failed: {type(e).__name__}"})
    return out


def explain(candidate_id: str, job_id: str, router=None, *, private: bool = True, method: str = "llm",
            **kw) -> tuple[Explanation, list[Fact]]:
    facts = bundle(candidate_id, job_id)
    if method == "template" or router is None:
        return template_explanation(candidate_id, job_id, facts), facts
    return llm_explanation(router, candidate_id, job_id, facts, private=private, **kw), facts
