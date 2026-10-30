"""Modern extraction path (R2): chunk -> ontology-constrained JSON prompt -> validate (one repair retry)
-> merge chunks -> locate evidence offsets in the full document.

Items that fail validation after the repair are dropped one by one (not the whole chunk) and counted,
so a single bad item doesn't cost a whole chunk. Items whose quote can't be found in the text keep
char_start=None and count as ungrounded; the graph loader skips them (every edge needs evidence).
"""

import hashlib
import json
import re
from typing import Optional

from pydantic import ValidationError

from extraction.chunk import chunk_text
from extraction.prompts import CV_SYSTEM, JOB_SYSTEM, REPAIR, messages
from ontology import ontology as O
from ontology.schema import (CandidateItem, CredentialItem, ExperienceItem, JobItem, ParsedCV, ParsedJob,
                             ProjectItem, RequirementItem, SkillItem, Strict, locate_evidence)
from router.router import Router

PROMPT_VERSION = hashlib.sha256((CV_SYSTEM + JOB_SYSTEM).encode()).hexdigest()[:8]

LIST_FIELDS = {
    "CV": {"skills": SkillItem, "experiences": ExperienceItem, "credentials": CredentialItem, "projects": ProjectItem},
    "JobPosting": {"requirements": RequirementItem},
}
HEAD_FIELD = {"CV": ("candidate", CandidateItem), "JobPosting": ("job", JobItem)}
ENUMS = {"seniority": O.SENIORITY, "employment_type": O.EMPLOYMENT_TYPE, "work_mode": O.WORK_MODE,
         "skill_type": O.SKILL_TYPE, "requirement_type": O.REQUIREMENT_TYPE, "importance": O.IMPORTANCE,
         "level": O.EDUCATION_LEVEL, "education_level": O.EDUCATION_LEVEL, "credential_type": O.CREDENTIAL_TYPE,
         "proficiency": O.PROFICIENCY}
ENUM_LOOKUP = {k: {v.lower().replace("-", " "): v for v in vals} for k, vals in ENUMS.items()}
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class CVChunk(Strict):
    candidate: Optional[CandidateItem] = None
    skills: list[SkillItem] = []
    experiences: list[ExperienceItem] = []
    credentials: list[CredentialItem] = []
    projects: list[ProjectItem] = []


class JobChunk(Strict):
    job: Optional[JobItem] = None
    requirements: list[RequirementItem] = []


CHUNK_MODEL = {"CV": CVChunk, "JobPosting": JobChunk}


def parse_json(text: str) -> dict | None:
    """The JSON object in a reply (tolerates code fences / chatter); None if there is no valid object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    i, j = text.find("{"), text.rfind("}")
    for candidate in (text, text[i:j + 1] if 0 <= i < j else ""):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def coerce(obj):
    """Light, model-agnostic cleanup before validation: enum casing, "null" strings, "5+" years."""
    if isinstance(obj, list):
        return [coerce(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, str) and v.strip().lower() in ("null", "none", "n/a", ""):
            v = None
        if k in ENUM_LOOKUP and isinstance(v, str):
            v = ENUM_LOOKUP[k].get(v.lower().replace("-", " "), v)
        if k in ("min_years", "total_years_experience", "year") and isinstance(v, str):
            m = NUMBER.search(v)
            v = (int(float(m.group())) if k == "year" else float(m.group())) if m else None
        out[k] = coerce(v)
    return out


def salvage(data: dict, doc_type: str) -> tuple[Strict, int]:
    """Keep every item that validates on its own; return (chunk model, number of dropped items)."""
    kept, dropped = {}, 0
    head, head_model = HEAD_FIELD[doc_type]
    if isinstance(data.get(head), dict):
        try:
            kept[head] = head_model.model_validate(data[head])
        except ValidationError:
            try:
                kept[head] = head_model.model_validate({**data[head], "location": None})
            except ValidationError:
                dropped += 1
    for field, model in LIST_FIELDS[doc_type].items():
        items = data.get(field) if isinstance(data.get(field), list) else []
        good = []
        for item in items:
            try:
                good.append(model.model_validate(item))
            except ValidationError:
                dropped += 1
        kept[field] = good
    return CHUNK_MODEL[doc_type](**kept), dropped


def extract_chunk(router: Router, doc_type: str, text: str, part: str, *, task: str, private: bool, doc_id: str,
                  run: int, provider: str | None, model: str | None, stats: dict) -> Strict:
    msgs = messages(doc_type, text, part)
    # 2500 keeps prompt + output inside Ollama's default 4096-token context and stops runaway lists
    kw = dict(private=private, doc_id=doc_id, run=run, provider=provider, model=model, max_tokens=2500)
    reply = router.chat(task, msgs, **kw)
    stats["served_by"].add(f"{reply.provider}/{reply.model}")
    stats["calls"] += 1
    stats["latency_s"] += reply.latency_s
    stats["prompt_tokens"] += reply.prompt_tokens
    stats["completion_tokens"] += reply.completion_tokens
    stats["cost_usd"] = None if reply.cost_usd is None or stats["cost_usd"] is None else stats["cost_usd"] + reply.cost_usd
    data = coerce(parse_json(reply.text))
    try:
        if data is None:
            raise ValueError("the output was not a complete JSON object (it may have been cut off: be more concise)")
        return CHUNK_MODEL[doc_type].model_validate(data)
    except (ValidationError, ValueError) as e:
        stats["repairs"] += 1
        errors = ("\n".join(f"- {'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:15])
                  if isinstance(e, ValidationError) else f"- {e}")
        fix = router.chat(task, msgs + [{"role": "assistant", "content": reply.text},
                                        {"role": "user", "content": REPAIR.format(errors=errors)}], **kw)
        stats["served_by"].add(f"{fix.provider}/{fix.model}")
        stats["calls"] += 1
        stats["latency_s"] += fix.latency_s
        stats["prompt_tokens"] += fix.prompt_tokens
        stats["completion_tokens"] += fix.completion_tokens
        stats["cost_usd"] = None if fix.cost_usd is None or stats["cost_usd"] is None else stats["cost_usd"] + fix.cost_usd
        fixed = coerce(parse_json(fix.text))
        try:
            return CHUNK_MODEL[doc_type].model_validate(fixed)
        except ValidationError:
            out, dropped = salvage(fixed or data or {}, doc_type)
            stats["dropped_items"] += dropped
            return out


def _ground(item, doc: str, chunk_start: int, chunk_len: int):
    """Locate the item's quote inside its chunk first (precise), then anywhere in the document."""
    ev = locate_evidence(doc[chunk_start:chunk_start + chunk_len], item.evidence)
    if ev.char_start is not None:
        ev = ev.model_copy(update={"char_start": ev.char_start + chunk_start, "char_end": ev.char_end + chunk_start})
    else:
        ev = locate_evidence(doc, item.evidence)
    item.evidence = ev
    if getattr(item, "location", None) is not None:
        _ground(item.location, doc, chunk_start, chunk_len)
    return item


def _key(field: str, item) -> tuple:
    norm = lambda s: re.sub(r"[^a-z0-9+#]+", " ", (s or "").lower()).strip()
    if field == "skills":
        return (norm(item.name),)
    if field == "experiences":
        return norm(item.title), norm(item.company), item.start_date
    if field == "requirements":
        return (norm(item.text),)
    return (norm(item.name),)


def extract_document(router: Router, doc_id: str, doc_type: str, text: str, *, private: bool,
                     provider: str | None = None, model: str | None = None, run: int = 0,
                     title: str | None = None, company: str | None = None) -> tuple[ParsedCV | ParsedJob, dict]:
    task = "extract_cv" if doc_type == "CV" else "extract_job"
    label = (provider, model) if provider is not None else router.route(task)   # pinned models never fall back
    cfg = router.cfg["llm"]
    chunks = chunk_text(text, cfg["chunk_words"][0], cfg["chunk_words"][1], cfg["chunk_overlap"])
    stats = {"calls": 0, "repairs": 0, "dropped_items": 0, "latency_s": 0.0, "prompt_tokens": 0,
             "completion_tokens": 0, "cost_usd": 0.0, "chunks": len(chunks), "served_by": set()}
    head, merged, seen = None, {f: [] for f in LIST_FIELDS[doc_type]}, set()
    for c in chunks:
        part = f" (part {c.index + 1} of {len(chunks)})" if len(chunks) > 1 else ""
        out = extract_chunk(router, doc_type, c.text, part, task=task, private=private, doc_id=doc_id, run=run,
                            provider=provider, model=model, stats=stats)
        h = getattr(out, HEAD_FIELD[doc_type][0])
        if head is None and h is not None:
            head = h
            if getattr(head, "location", None) is not None:
                _ground(head.location, text, c.start, len(c.text))
        for field in merged:
            for item in getattr(out, field):
                k = (field, *_key(field, item))
                if k[1:] and k[1] and k not in seen:
                    seen.add(k)
                    merged[field].append(_ground(item, text, c.start, len(c.text)))
    served = sorted(stats.pop("served_by"))
    stats["served_by"] = served
    extractor = served[0] if len(served) == 1 else "/".join(label) if not served else "+".join(served)
    common = {"doc_id": doc_id, "extractor": extractor, "extractor_version": PROMPT_VERSION}
    if doc_type == "CV":
        parsed = ParsedCV(**common, candidate=head or CandidateItem(), **merged)
    else:
        if head is None:
            head = JobItem(title=title or "Unknown", company=company or None)
        parsed = ParsedJob(**common, job=head, **merged)
    items = [i for f in merged for i in merged[f]]
    stats["items"] = len(items)
    stats["ungrounded_items"] = sum(i.evidence.char_start is None for i in items)
    stats["latency_s"] = round(stats["latency_s"], 3)
    return parsed, stats
