"""Research & Outreach and Scheduler agents. Both only draft: nothing leaves the machine.

- company_brief(): facts about the company from the graph and the job corpus (postings, roles, locations),
  plus a short LLM summary of the posting (public text only when the route is hosted).
- outreach_draft(): a short message to the recruiter built from cited match facts (CV facts -> local model).
- follow_up_ics(): an .ics calendar file for a follow-up reminder, created only after the user approves
  in the UI (no Calendar API, so no account access is needed).
"""

import json
import uuid
from datetime import datetime, timedelta

import pandas as pd

from agents.explainer import Fact
from graph.build import driver

BRIEF_PROMPT = """Summarize this job posting for a candidate in 3 bullet points: what the company does,
what the role focuses on, and what they emphasize in candidates. Use only the text.
Posting:
\"\"\"{text}\"\"\"
Reply JSON {{"bullets": ["...", "...", "..."]}}."""

OUTREACH_PROMPT = """Write a short, polite message (under 120 words) from a candidate to the recruiter for
"{title}" at {company}. Mention only these facts about the candidate's fit:
{facts}
Do not invent experience. End by asking about next steps. Reply JSON {{"subject": "...", "body": "..."}}."""


def company_brief(router, job_id: str, jobs_meta: pd.DataFrame, job_text: str, *, public: bool) -> dict:
    with driver().session() as s:
        r = s.run("MATCH (j:JobPosting {job_id: $j})-[:POSTED_BY]->(c:Company) "
                  "OPTIONAL MATCH (c)<-[:POSTED_BY]-(o:JobPosting) RETURN c.name AS name, count(o) AS n_graph",
                  j=job_id).single()
    row = jobs_meta.set_index("doc_id").loc[job_id]
    company = (r and r["name"]) or row.get("company")
    others = jobs_meta[jobs_meta["company"] == row.get("company")]
    brief = {"company": company, "postings_in_corpus": int(len(others)),
             "titles": others["title"].value_counts().head(5).to_dict(),
             "locations": others["location"].value_counts().head(5).to_dict() if "location" in others else {},
             "summary": []}
    try:
        out = json.loads(router.chat("research", [{"role": "user", "content": BRIEF_PROMPT.format(
            text=job_text[:4000])}], private=not public).text)
        brief["summary"] = out.get("bullets", [])
    except Exception:
        pass
    return brief


def outreach_draft(router, title: str, company: str, facts: list[Fact], *, private: bool) -> dict:
    use = [f for f in facts if f.kind in ("met", "experience", "occupation")][:5]
    if not use:
        return {"subject": "", "body": "", "cites": []}
    try:
        out = json.loads(router.chat("research", [{"role": "user", "content": OUTREACH_PROMPT.format(
            title=title, company=company or "your company", facts="\n".join(f"- {f.text}" for f in use))}],
            private=private).text)
    except Exception:
        out = {}
    return {"subject": out.get("subject", ""), "body": out.get("body", ""),
            "cites": sorted({e for f in use for e in f.edge_ids}), "status": "draft (needs your approval)"}


def follow_up_ics(title: str, company: str, days: int = 7, when: datetime | None = None) -> str:
    start = (when or datetime.now()).replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=days)
    stamp = lambda d: d.strftime("%Y%m%dT%H%M%S")
    return "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//LinkedIn Assistant//CSE573//EN", "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}", f"DTSTAMP:{stamp(datetime.now())}", f"DTSTART:{stamp(start)}",
        f"DTEND:{stamp(start + timedelta(minutes=15))}", f"SUMMARY:Follow up: {title} at {company}",
        "DESCRIPTION:Follow up on your application (created by LinkedIn Assistant).", "END:VEVENT", "END:VCALENDAR", ""])
