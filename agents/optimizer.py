"""Profile Optimizer agent: CV rewrites that only rephrase existing claims, plus upskilling pointers.

Rewrites: for skills the job asks for and the CV already has, the LLM rephrases the CV's own evidence
quote in the posting's vocabulary. The Critic then checks each rewrite against the original quote and
rejects anything that adds a claim (new skill, number, title, employer). Upskilling pointers come
straight from the graph (gaps, and RELATED_TO bridges from skills the candidate already has).
"""

import json

from agents.explainer import Fact

REWRITE_PROMPT = """Rewrite each CV line so it reads well for this job, using the job's wording where it fits.
You may only rephrase: do not add skills, tools, numbers, titles, employers or results that the original
line does not state.
Job title: {title}
Lines:
{lines}
Reply JSON {{"rewrites": [{{"id": <line id>, "text": "<rewritten line>"}}]}}."""

CHECK_PROMPT = """Original CV line: "{orig}"
Rewritten line: "{new}"
Does the rewrite claim anything (skill, tool, number, title, employer, result) that the original does not
state? Reply JSON {{"adds_claims": true|false, "reason": "<short>"}}."""


def rewrites(router, facts: list[Fact], job_title: str, *, private: bool, limit: int = 5) -> list[dict]:
    met = [f for f in facts if f.kind == "met"][:limit]
    quotes = [f.text.split('the CV states "', 1)[-1].rstrip('".') for f in met]
    if not quotes:
        return []
    lines = "\n".join(f"{i}. {q}" for i, q in enumerate(quotes))
    try:
        out = json.loads(router.chat("optimize_profile", [{"role": "user", "content": REWRITE_PROMPT.format(
            title=job_title, lines=lines)}], private=private).text).get("rewrites", [])
    except Exception:
        return []
    results = []
    for r in out:
        i = r.get("id")
        if not isinstance(i, int) or not 0 <= i < len(quotes) or not r.get("text"):
            continue
        try:
            v = json.loads(router.chat("critic", [{"role": "user", "content": CHECK_PROMPT.format(
                orig=quotes[i], new=r["text"])}], private=private).text)
            ok = not v.get("adds_claims", True)
            reason = v.get("reason", "")
        except Exception as e:
            ok, reason = False, f"critic failed: {type(e).__name__}"
        results.append({"original": quotes[i], "rewrite": r["text"], "cites": met[i].edge_ids,
                        "verdict": "approved" if ok else "rejected", "reason": reason})
    return results


def upskilling(facts: list[Fact]) -> list[dict]:
    """Deterministic: each unmet required skill, with the related skill the candidate can build from."""
    bridges = {}
    for f in facts:
        if f.kind == "bridge":
            missing, have = f.text.split(" (required, missing) is related to ")
            bridges.setdefault(missing, []).append((have.split(", which")[0], f.edge_ids))
    out = []
    for f in facts:
        if f.kind != "gap":
            continue
        skill = f.text.split("asks for ", 1)[1].split(", which")[0]
        b = bridges.get(skill, [])
        out.append({"skill": skill, "requirement_cites": f.edge_ids,
                    "build_from": [x for x, _ in b], "bridge_cites": [e for _, ids in b for e in ids],
                    "pointer": (f"Learn {skill}; you can build on " + ", ".join(x for x, _ in b) + ".") if b
                    else f"Learn {skill}; no related skill on your CV yet, so start with the basics."})
    return out
