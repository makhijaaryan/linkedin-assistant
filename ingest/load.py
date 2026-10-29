"""Load the raw datasets into two unified tables: jobs and resumes.

Every row gets a stable `doc_id` (source prefix + original id) and cleaned, masked `text`.
"""

import glob
from pathlib import Path

import pandas as pd

from ingest.text import clean_text, mask_contacts

RAW = Path("data/raw")
SENIORITY_MAP = {"Mid-Senior level": "Mid-Senior level", "Entry level": "Entry level", "Associate": "Associate",
                 "Director": "Director", "Internship": "Internship", "Executive": "Executive"}


def _clean_and_mask(series: pd.Series) -> tuple[pd.Series, int]:
    cleaned = series.map(clean_text).map(mask_contacts)
    return cleaned.map(lambda x: x[0]), int(cleaned.map(lambda x: x[1]).sum())


def _state(location: str) -> str:
    """'Austin, TX' -> 'TX'; 'United States' -> 'US'; otherwise 'Other'."""
    parts = [p.strip() for p in str(location).split(",")]
    if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isupper():
        return parts[-1]
    return "US" if "United States" in str(location) else "Other"


def load_jobs() -> tuple[pd.DataFrame, dict]:
    src = RAW / "linkedin_job_postings"
    p = pd.read_csv(src / "postings.csv")
    ind = (pd.read_csv(src / "jobs/job_industries.csv")
           .merge(pd.read_csv(src / "mappings/industries.csv"), on="industry_id", how="left")
           .groupby("job_id")["industry_name"].first())
    text, n_masked = _clean_and_mask(p["description"])
    jobs = pd.DataFrame({
        "doc_id": "job_" + p["job_id"].astype(str),
        "source": "kaggle_linkedin_postings",
        "title": p["title"].fillna("").map(clean_text),
        "company": p["company_name"].fillna("").map(clean_text),
        "location": p["location"].fillna(""),
        "state": p["location"].map(_state),
        "seniority": p["formatted_experience_level"].map(SENIORITY_MAP).fillna("Unknown"),
        "employment_type": p["formatted_work_type"].fillna("Other"),
        "remote_allowed": p["remote_allowed"].fillna(0).astype(bool),
        "industry": p["job_id"].map(ind).fillna("Unknown"),
        "text": text,
    })
    jobs = jobs[jobs["text"].str.len() > 0].reset_index(drop=True)
    return jobs, {"jobs_loaded": len(p), "jobs_with_text": len(jobs), "jobs_contacts_masked": n_masked}


def load_resumes() -> tuple[pd.DataFrame, dict]:
    k = pd.read_csv(RAW / "resume_dataset/Resume/Resume.csv")
    k_text, k_masked = _clean_and_mask(k["Resume_str"])
    kaggle = pd.DataFrame({"doc_id": "cvk_" + k["ID"].astype(str), "source": "kaggle_resume_dataset",
                           "category": k["Category"], "labels": k["Category"].map(lambda c: [c]), "text": k_text})

    rows = []
    for txt in sorted(glob.glob(str(RAW / "resume_corpus/extracted/*.txt"))):
        stem = Path(txt).stem
        lab = Path(txt).with_suffix(".lab")
        labels = lab.read_text().split() if lab.exists() else []
        rows.append({"doc_id": f"cvc_{stem}", "source": "resume_corpus",
                     "category": labels[0] if labels else "Unknown", "labels": labels,
                     "raw": Path(txt).read_text(errors="ignore")})
    corpus = pd.DataFrame(rows)
    c_text, c_masked = _clean_and_mask(corpus.pop("raw"))
    corpus["text"] = c_text

    resumes = pd.concat([kaggle, corpus], ignore_index=True)
    resumes = resumes[resumes["text"].str.len() > 0].reset_index(drop=True)
    return resumes, {"resumes_kaggle": len(kaggle), "resumes_corpus": len(corpus),
                     "resumes_contacts_masked": k_masked + c_masked}
