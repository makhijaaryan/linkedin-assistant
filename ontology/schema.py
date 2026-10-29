"""Extraction schema: what an extractor (LLM or classical) must output for one document.

Every extracted item carries an `Evidence` with a verbatim quote from the source.
Extractors give the quote; `locate_evidence` fills char offsets in code, because
LLMs are unreliable at counting characters.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ontology import ontology as O

Seniority = Literal[tuple(O.SENIORITY)]
EmploymentType = Literal[tuple(O.EMPLOYMENT_TYPE)]
WorkMode = Literal[tuple(O.WORK_MODE)]
SkillType = Literal[tuple(O.SKILL_TYPE)]
RequirementType = Literal[tuple(O.REQUIREMENT_TYPE)]
Importance = Literal[tuple(O.IMPORTANCE)]
EducationLevel = Literal[tuple(O.EDUCATION_LEVEL)]
CredentialType = Literal[tuple(O.CREDENTIAL_TYPE)]
Proficiency = Literal[tuple(O.PROFICIENCY)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(Strict):
    quote: str = Field(description="Verbatim text copied from the document that supports this item.")
    confidence: float = Field(ge=0, le=1, description="Extractor's confidence that the item is correct.")
    char_start: Optional[int] = Field(None, description="Filled in by code, not by the extractor.")
    char_end: Optional[int] = Field(None, description="Filled in by code, not by the extractor.")


class LocationItem(Strict):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    is_remote: bool = False
    evidence: Evidence


class SkillItem(Strict):
    name: str = Field(description="The skill as written, e.g. 'PyTorch'. Normalized to O*NET later.")
    skill_type: SkillType
    proficiency: Proficiency = "Unstated"
    evidence: Evidence


class ExperienceItem(Strict):
    title: str
    company: Optional[str] = None
    location: Optional[LocationItem] = None
    start_date: Optional[str] = Field(None, description="YYYY or YYYY-MM")
    end_date: Optional[str] = Field(None, description="YYYY, YYYY-MM or 'Present'")
    is_current: bool = False
    employment_type: Optional[EmploymentType] = None
    skills_used: list[str] = Field(default_factory=list, description="Skill names applied in this role.")
    evidence: Evidence


class CredentialItem(Strict):
    name: str
    credential_type: CredentialType
    level: Optional[EducationLevel] = None
    field: Optional[str] = None
    issuer: Optional[str] = None
    year: Optional[int] = None
    evidence: Evidence


class ProjectItem(Strict):
    name: str
    description: Optional[str] = None
    skills_used: list[str] = Field(default_factory=list)
    evidence: Evidence


class CandidateItem(Strict):
    headline: Optional[str] = None
    total_years_experience: Optional[float] = None
    seniority: Seniority = "Unknown"
    location: Optional[LocationItem] = None


class ParsedCV(Strict):
    """One CV, extracted. Ontology: Candidate + HAS_SKILL/HELD/EARNED/BUILT."""
    doc_id: str
    extractor: str = Field(description="e.g. 'ollama/llama3.1:8b' or 'crf-baseline'")
    extractor_version: str
    candidate: CandidateItem
    skills: list[SkillItem] = Field(default_factory=list)
    experiences: list[ExperienceItem] = Field(default_factory=list)
    credentials: list[CredentialItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)


class RequirementItem(Strict):
    text: str = Field(description="The requirement, close to how the posting words it.")
    requirement_type: RequirementType
    importance: Importance
    min_years: Optional[float] = None
    education_level: Optional[EducationLevel] = None
    skills: list[str] = Field(default_factory=list, description="Skill names this requirement targets.")
    evidence: Evidence


class JobItem(Strict):
    title: str
    company: Optional[str] = None
    seniority: Seniority = "Unknown"
    employment_type: Optional[EmploymentType] = None
    work_mode: WorkMode = "Unknown"
    location: Optional[LocationItem] = None


class ParsedJob(Strict):
    """One job posting, extracted. Ontology: JobPosting + POSTED_BY/HAS_REQUIREMENT/TARGETS."""
    doc_id: str
    extractor: str
    extractor_version: str
    job: JobItem
    requirements: list[RequirementItem] = Field(default_factory=list)


def locate_evidence(text: str, ev: Evidence) -> Evidence:
    """Fill char offsets by finding the quote in the document (exact, then case/space-insensitive)."""
    i = text.find(ev.quote)
    if i < 0:
        import re
        pattern = r"\s+".join(map(re.escape, ev.quote.split()))
        m = re.search(pattern, text, flags=re.IGNORECASE) if pattern else None
        if not m:
            return ev  # unlocated: the validator treats this as an ungrounded item
        i, j = m.start(), m.end()
    else:
        j = i + len(ev.quote)
    return ev.model_copy(update={"char_start": i, "char_end": j})
