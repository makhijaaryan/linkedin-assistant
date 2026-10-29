# Annotation Guide (v1.0, frozen 2026-10-28)

Four labeling tasks produce the gold test data. Gold labels are used **only for evaluation** (plus the 100 hand-corrected LoRA examples, which are kept apart from the test set). Ontology reference: `ontology/generated/ONTOLOGY.md`.

General rules:
- Label what the text **says**, not what you infer. If it's not stated, leave it out.
- Spans are **minimal and exact**: select the shortest text that fully expresses the item ("PyTorch", not "experience with PyTorch frameworks").
- When unsure, add a note instead of guessing. Record the time spent per document.
- Contact details are already masked as `[EMAIL]`, `[URL]`, `[PHONE]`; never unmask.

---

## Task A: Entity extraction (150 docs: 75 job postings + 75 CVs)

Tool: Label Studio, config `eval/annotation/label_studio_extraction.xml`, tasks `data/processed/annotation/extraction_tasks.json`.

| Label | Mark this | Don't mark |
|---|---|---|
| **Skill** | A tool, technology, technical skill, soft skill, domain area or language: "Python", "Tableau", "stakeholder management", "GAAP", "Spanish" | Generic words ("experience", "skills", "team"); whole sentences |
| **Requirement** (jobs only) | One condition on applicants, the full clause: "5+ years of experience in Python", "Bachelor's degree in CS or related field" | Responsibilities ("You will build dashboards"), benefits, company blurbs |
| **Experience** (CVs only) | The role header of one job held: title (+ company and dates if on the same line) | Bullet points under the role (their skills get **Skill** labels) |
| **Credential** | A degree, certification, license or course: "B.S. Computer Science", "AWS Certified Solutions Architect", "CPA" | School names alone |
| **Project** (CVs only) | A named project or a clearly separate project entry | Ordinary job duties |
| **Company** | An organization name: employer (CVs) or hiring company (jobs) | "the company", "our client" |
| **Occupation** | The job title as a role: "Data Analyst", "Registered Nurse" | Seniority words alone ("Senior") |
| **Location** | City, state, country, or "Remote" / "Hybrid" | Addresses (should already be masked) |

Requirement attributes (use the per-region choices):
- **importance:** *Required* (must / required / minimum) or *Preferred* (preferred / nice to have / a plus / bonus). If not stated → Required.
- **requirement_type:** Skill, Experience, Education, Credential, Other.
- **min_years:** the number, if stated.

Agreement:
- Proposal plan: two annotators per document, report Cohen's κ.
- **Solo-work fallback:** label all 150, then re-label a random 30 after at least 3 days without looking at the first pass. Report the intra-annotator κ, and state this limitation in the report.

## Task B: Duplicate hand-check (200 job pairs + 50 resume pairs)

Files: `data/processed/annotation/dedupe_handcheck_*.csv`. For each row, fill `is_duplicate`:
- **1:** the same posting or CV. Reposts, minor edits, and the same job at a different location with an identical description all count.
- **0:** genuinely different. Same company template but a different role or requirements → 0.

Dedupe precision = mean of `is_duplicate`. **Target ≥ 0.95.**

## Task C: Relevance judging (40 CV profiles × pooled top-20 jobs = 800 pairs)

Created in Phase 5, when systems exist to pool from. Judge each (CV, job) pair, blind to which system retrieved it:
- **2: Strong fit.** The candidate meets most required qualifications; you'd shortlist them.
- **1: Partial fit.** Relevant field and some requirements met, but clear gaps (seniority, key skill, credential).
- **0: Not a fit.** Different field, or misses most required qualifications.

Judge on stated qualifications only, not on location or salary.

## Task D: Explanation faithfulness (60 explanations)

Created in Phase 6. For each sentence of an explanation, mark:
- **Supported:** the cited graph edge exists and says what the sentence claims.
- **Unsupported:** there's no citation, the cited edge doesn't exist, or the sentence claims more than the edge says (hallucination).

Faithfulness = supported sentences / all sentences. **Target ≥ 0.95.**
