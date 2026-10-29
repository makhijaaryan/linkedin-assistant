"""The job-market ontology (proposal Fig. 2): the single source of truth.

Follows the course's starter-ontology method: the smallest set of entity types,
relationships and type descriptors that still supports extraction and reasoning,
with typed attributes in the style of the Land/Property Dispute reference.
`ontology/build.py` turns this into the JSON Schema, ONTOLOGY.md and ONTOLOGY.docx.
"""

VERSION = "1.0.0"

# Shared enumerations (also used by schema.py so extraction and graph agree).
SENIORITY = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Unknown"]
EMPLOYMENT_TYPE = ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Volunteer", "Other"]
WORK_MODE = ["On-site", "Remote", "Hybrid", "Unknown"]
SKILL_TYPE = ["Tool/Technology", "Technical", "Soft", "Domain Knowledge", "Language"]
REQUIREMENT_TYPE = ["Skill", "Experience", "Education", "Credential", "Other"]
IMPORTANCE = ["Required", "Preferred"]
EDUCATION_LEVEL = ["High School", "Associate", "Bachelor", "Master", "Doctorate", "None stated"]
CREDENTIAL_TYPE = ["Degree", "Certification", "License", "Course"]
PROFICIENCY = ["Beginner", "Intermediate", "Advanced", "Expert", "Unstated"]
DOC_TYPE = ["CV", "JobPosting"]
RELATION_SOURCE = ["Extracted", "Taxonomy", "Embedding", "LLM-inferred"]


def enum(values):
    return "Enumeration: " + ", ".join(values)


# name -> (type descriptor, {attribute: type})
ENTITIES = {
    "Candidate": (
        "The person a CV describes. One per CV; the anchor for everything the candidate claims.",
        {"candidate_id": "String", "headline": "String",
         "total_years_experience": "Number", "seniority": enum(SENIORITY)},
    ),
    "JobPosting": (
        "An advertised position with the conditions an applicant must meet.",
        {"job_id": "String", "title": "String", "seniority": enum(SENIORITY),
         "employment_type": enum(EMPLOYMENT_TYPE), "work_mode": enum(WORK_MODE),
         "salary_min": "Number", "salary_max": "Number", "currency": "String"},
    ),
    "Company": (
        "An organization that posts jobs or employed a candidate.",
        {"name": "String", "industry": "String"},
    ),
    "Occupation": (
        "A standardized role from the O*NET-SOC taxonomy; lets differently worded titles meet.",
        {"onet_soc_code": "String", "title": "String"},
    ),
    "Skill": (
        "A capability a person can have and a job can ask for, normalized to an O*NET entry.",
        {"canonical_name": "String", "skill_type": enum(SKILL_TYPE), "taxonomy_id": "String",
         "category": "String", "hot_technology": "Boolean"},
    ),
    "Requirement": (
        "One condition a posting places on applicants (a claim the candidate's evidence must satisfy).",
        {"text": "String", "requirement_type": enum(REQUIREMENT_TYPE), "importance": enum(IMPORTANCE),
         "min_years": "Number", "education_level": enum(EDUCATION_LEVEL)},
    ),
    "Experience": (
        "A role the candidate held for a period of time.",
        {"title": "String", "start_date": "String (YYYY or YYYY-MM)", "end_date": "String (YYYY, YYYY-MM or Present)",
         "duration_years": "Number", "is_current": "Boolean", "employment_type": enum(EMPLOYMENT_TYPE)},
    ),
    "Credential": (
        "A degree, certification, license or course the candidate earned.",
        {"name": "String", "credential_type": enum(CREDENTIAL_TYPE), "level": enum(EDUCATION_LEVEL),
         "field": "String", "issuer": "String", "year": "Number"},
    ),
    "Project": (
        "A piece of work the candidate built, outside or within a role.",
        {"name": "String", "description": "String"},
    ),
    "Location": (
        "A place a job is based or a candidate lives; remote work is a location flag.",
        {"city": "String", "state": "String", "country": "String", "is_remote": "Boolean"},
    ),
    "Evidence": (
        "The exact source span a fact was extracted from. Every extracted fact must point to one.",
        {"evidence_id": "String", "doc_id": "String", "doc_type": enum(DOC_TYPE), "char_start": "Number",
         "char_end": "Number", "text": "String", "extractor": "String", "extractor_version": "String",
         "confidence": "Number (0-1)"},
    ),
}

# (name, source(s), target(s), descriptor). Source/target lists mean "any of".
RELATIONS = [
    ("HAS_SKILL", ["Candidate"], ["Skill"], "The candidate claims this skill anywhere in the CV (attr: proficiency)."),
    ("HELD", ["Candidate"], ["Experience"], "The candidate held this role."),
    ("EARNED", ["Candidate"], ["Credential"], "The candidate earned this degree, certification or license."),
    ("BUILT", ["Candidate"], ["Project"], "The candidate built this project."),
    ("AT", ["Experience"], ["Company"], "The role was held at this company."),
    ("AS_ROLE", ["Experience"], ["Occupation"], "The role normalizes to this O*NET occupation."),
    ("USED", ["Experience", "Project"], ["Skill"], "The skill was applied in this role or project."),
    ("POSTED_BY", ["JobPosting"], ["Company"], "The company advertising the job."),
    ("FOR_ROLE", ["JobPosting"], ["Occupation"], "The job normalizes to this O*NET occupation."),
    ("HAS_REQUIREMENT", ["JobPosting"], ["Requirement"], "The posting states this requirement."),
    ("TARGETS", ["Requirement"], ["Skill"], "The requirement asks for this skill."),
    ("LOCATED_IN", ["Candidate", "JobPosting", "Company"], ["Location"], "Where the entity is based."),
    ("RELATED_TO", ["Skill"], ["Skill"],
     "The skills are closely related (attrs: source, weight, inferred=true when not from the taxonomy)."),
    ("TYPICAL_FOR", ["Skill"], ["Occupation"], "O*NET lists the skill as used in this occupation (attr: hot_technology)."),
    ("SUPPORTED_BY", ["Candidate", "JobPosting", "Company", "Skill", "Requirement", "Experience", "Credential",
                      "Project", "Location"], ["Evidence"],
     "The fact is grounded in this source span. Extracted relationships also carry evidence_ids."),
]

# Proposal relations kept verbatim; LOCATED_IN is the one addition (Location had no relation in Fig. 2).
ADDED_RELATIONS = {"LOCATED_IN"}
