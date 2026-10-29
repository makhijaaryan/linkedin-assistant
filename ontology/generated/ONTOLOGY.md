# Job-Market Ontology v1.0.0

Generated from `ontology/ontology.py` — edit that file, then run `python -m ontology.build`.

## Entity types

| Entity Type | Type Descriptor | Attributes [Type] |
|---|---|---|
| **Candidate** | The person a CV describes. One per CV; the anchor for everything the candidate claims. | candidate_id [String]<br>headline [String]<br>total_years_experience [Number]<br>seniority [Enumeration: Internship, Entry level, Associate, Mid-Senior level, Director, Executive, Unknown] |
| **JobPosting** | An advertised position with the conditions an applicant must meet. | job_id [String]<br>title [String]<br>seniority [Enumeration: Internship, Entry level, Associate, Mid-Senior level, Director, Executive, Unknown]<br>employment_type [Enumeration: Full-time, Part-time, Contract, Temporary, Internship, Volunteer, Other]<br>work_mode [Enumeration: On-site, Remote, Hybrid, Unknown]<br>salary_min [Number]<br>salary_max [Number]<br>currency [String] |
| **Company** | An organization that posts jobs or employed a candidate. | name [String]<br>industry [String] |
| **Occupation** | A standardized role from the O*NET-SOC taxonomy; lets differently worded titles meet. | onet_soc_code [String]<br>title [String] |
| **Skill** | A capability a person can have and a job can ask for, normalized to an O*NET entry. | canonical_name [String]<br>skill_type [Enumeration: Tool/Technology, Technical, Soft, Domain Knowledge, Language]<br>taxonomy_id [String]<br>category [String]<br>hot_technology [Boolean] |
| **Requirement** | One condition a posting places on applicants (a claim the candidate's evidence must satisfy). | text [String]<br>requirement_type [Enumeration: Skill, Experience, Education, Credential, Other]<br>importance [Enumeration: Required, Preferred]<br>min_years [Number]<br>education_level [Enumeration: High School, Associate, Bachelor, Master, Doctorate, None stated] |
| **Experience** | A role the candidate held for a period of time. | title [String]<br>start_date [String (YYYY or YYYY-MM)]<br>end_date [String (YYYY, YYYY-MM or Present)]<br>duration_years [Number]<br>is_current [Boolean]<br>employment_type [Enumeration: Full-time, Part-time, Contract, Temporary, Internship, Volunteer, Other] |
| **Credential** | A degree, certification, license or course the candidate earned. | name [String]<br>credential_type [Enumeration: Degree, Certification, License, Course]<br>level [Enumeration: High School, Associate, Bachelor, Master, Doctorate, None stated]<br>field [String]<br>issuer [String]<br>year [Number] |
| **Project** | A piece of work the candidate built, outside or within a role. | name [String]<br>description [String] |
| **Location** | A place a job is based or a candidate lives; remote work is a location flag. | city [String]<br>state [String]<br>country [String]<br>is_remote [Boolean] |
| **Evidence** | The exact source span a fact was extracted from. Every extracted fact must point to one. | evidence_id [String]<br>doc_id [String]<br>doc_type [Enumeration: CV, JobPosting]<br>char_start [Number]<br>char_end [Number]<br>text [String]<br>extractor [String]<br>extractor_version [String]<br>confidence [Number (0-1)] |

## Relationships

| Relationship | Source Entity → Target Entity | Relationship Descriptor |
|---|---|---|
| **HAS_SKILL** | Candidate → Skill | The candidate claims this skill anywhere in the CV (attr: proficiency). |
| **HELD** | Candidate → Experience | The candidate held this role. |
| **EARNED** | Candidate → Credential | The candidate earned this degree, certification or license. |
| **BUILT** | Candidate → Project | The candidate built this project. |
| **AT** | Experience → Company | The role was held at this company. |
| **AS_ROLE** | Experience → Occupation | The role normalizes to this O*NET occupation. |
| **USED** | Experience | Project → Skill | The skill was applied in this role or project. |
| **POSTED_BY** | JobPosting → Company | The company advertising the job. |
| **FOR_ROLE** | JobPosting → Occupation | The job normalizes to this O*NET occupation. |
| **HAS_REQUIREMENT** | JobPosting → Requirement | The posting states this requirement. |
| **TARGETS** | Requirement → Skill | The requirement asks for this skill. |
| **LOCATED_IN** | Candidate | JobPosting | Company → Location | Where the entity is based. *(added: Location had no relation in Fig. 2)* |
| **RELATED_TO** | Skill → Skill | The skills are closely related (attrs: source, weight, inferred=true when not from the taxonomy). |
| **TYPICAL_FOR** | Skill → Occupation | O*NET lists the skill as used in this occupation (attr: hot_technology). |
| **SUPPORTED_BY** | Candidate | JobPosting | Company | Skill | Requirement | Experience | Credential | Project | Location → Evidence | The fact is grounded in this source span. Extracted relationships also carry evidence_ids. |
