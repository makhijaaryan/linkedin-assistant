"""Ontology-constrained extraction prompts (one per document type) and the free-form SPO prompt (RQ3).

The JSON shape mirrors ontology/schema.py minus the fields code fills in (doc_id,
extractor, char offsets). Enumerations come from ontology.py, so prompt and schema agree.
"""

import json

from ontology import ontology as O

RULES = """Rules:
- Extract only what the text states. Do not infer or add skills that are not written.
- Every item needs "evidence": {"quote": <text copied exactly from the document>, "confidence": 0-1}.
  The quote must be a verbatim, short span (one line or clause) containing the item.
- Skill names are short and canonical as written ("Python", "Tableau", "stakeholder management"),
  never whole sentences. List each distinct skill once.
- Use only the allowed values for enumerated fields. Use null when a field is not stated.
- Output a single JSON object and nothing else."""

CV_SHAPE = {
    "candidate": {"headline": "str|null", "total_years_experience": "number|null",
                  "seniority": "|".join(O.SENIORITY),
                  "location": {"city": "str|null", "state": "str|null", "country": "str|null", "is_remote": "bool",
                               "evidence": "EVIDENCE"}},
    "skills": [{"name": "str", "skill_type": "|".join(O.SKILL_TYPE), "proficiency": "|".join(O.PROFICIENCY),
                "evidence": "EVIDENCE"}],
    "experiences": [{"title": "str", "company": "str|null", "start_date": "YYYY or YYYY-MM|null",
                     "end_date": "YYYY, YYYY-MM or Present|null", "is_current": "bool",
                     "employment_type": "|".join(O.EMPLOYMENT_TYPE) + "|null", "skills_used": ["str"],
                     "evidence": "EVIDENCE (the role header line)"}],
    "credentials": [{"name": "str", "credential_type": "|".join(O.CREDENTIAL_TYPE),
                     "level": "|".join(O.EDUCATION_LEVEL) + "|null", "field": "str|null", "issuer": "str|null",
                     "year": "int|null", "evidence": "EVIDENCE"}],
    "projects": [{"name": "str", "description": "str|null", "skills_used": ["str"], "evidence": "EVIDENCE"}],
}

JOB_SHAPE = {
    "job": {"title": "str", "company": "str|null", "seniority": "|".join(O.SENIORITY),
            "employment_type": "|".join(O.EMPLOYMENT_TYPE) + "|null", "work_mode": "|".join(O.WORK_MODE),
            "location": {"city": "str|null", "state": "str|null", "country": "str|null", "is_remote": "bool",
                         "evidence": "EVIDENCE"}},
    "requirements": [{"text": "str (the condition, close to the posting's wording)",
                      "requirement_type": "|".join(O.REQUIREMENT_TYPE), "importance": "|".join(O.IMPORTANCE),
                      "min_years": "number|null", "education_level": "|".join(O.EDUCATION_LEVEL) + "|null",
                      "skills": ["skill names this requirement asks for"], "evidence": "EVIDENCE"}],
}

CV_SYSTEM = f"""You extract structured facts from a CV (resume) into a fixed job-market ontology.
Return JSON with exactly this shape ("a|b" means choose one value; EVIDENCE is {{"quote": str, "confidence": number}}):
{json.dumps(CV_SHAPE, indent=1)}
"candidate" may be null if the text has no headline/location. Lists may be empty.
{RULES}"""

JOB_SYSTEM = f"""You extract structured facts from a job posting into a fixed job-market ontology.
Return JSON with exactly this shape ("a|b" means choose one value; EVIDENCE is {{"quote": str, "confidence": number}}):
{json.dumps(JOB_SHAPE, indent=1)}
A requirement is a condition on applicants (skills, years of experience, degrees, certifications), not a duty
or benefit. importance is "Preferred" for preferred / nice to have / a plus / bonus, otherwise "Required".
"job" may be null if this part of the posting has no title/company/location information.
{RULES}"""

SPO_SYSTEM = """Extract all facts from the text as (subject, predicate, object) triples. Use your own wording
for predicates and entity names; there is no fixed schema.
Return JSON: {"triples": [{"subject": str, "predicate": str, "object": str, "quote": <verbatim supporting text>}]}
Output a single JSON object and nothing else."""

REPAIR = """Your previous output was not valid for the required shape. Errors:
{errors}
Return the corrected JSON object only."""


def messages(doc_type: str, chunk: str, part: str = "") -> list[dict]:
    system = CV_SYSTEM if doc_type == "CV" else JOB_SYSTEM
    label = "CV" if doc_type == "CV" else "Job posting"
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{label}{part}:\n\"\"\"\n{chunk}\n\"\"\""}]
