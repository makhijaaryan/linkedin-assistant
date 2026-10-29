"""Free weak labels (training only, never evaluation): job skill categories and industries,
resume categories, and O*NET occupation-skill / title lists."""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/linkedin_job_postings")
ONET = Path("data/external/onet/db_31_0_csv")


def job_weak_labels() -> pd.DataFrame:
    skills = (pd.read_csv(RAW / "jobs/job_skills.csv")
              .merge(pd.read_csv(RAW / "mappings/skills.csv"), on="skill_abr", how="left")
              .groupby("job_id")["skill_name"].agg(sorted).rename("skill_categories"))
    industries = (pd.read_csv(RAW / "jobs/job_industries.csv")
                  .merge(pd.read_csv(RAW / "mappings/industries.csv"), on="industry_id", how="left")
                  .groupby("job_id")["industry_name"].agg(lambda s: sorted(s.dropna())).rename("industries"))
    out = pd.concat([skills, industries], axis=1).reset_index().rename(columns={"index": "job_id"})
    out["doc_id"] = "job_" + out["job_id"].astype(str)
    return out.drop(columns="job_id")


def onet_tables() -> dict[str, pd.DataFrame]:
    tech = pd.read_csv(ONET / "software_skills.csv")
    tech = tech.rename(columns={"O*NET-SOC Code": "onet_soc_code", "Title": "occupation",
                                "Workplace Example": "skill", "Element ID": "category_id",
                                "Element Name": "category", "Hot Technology": "hot", "In Demand": "in_demand"})
    tech["hot"] = tech["hot"].eq("Y")
    tech["in_demand"] = tech["in_demand"].eq("Y")
    titles = pd.read_csv(ONET / "job_titles.csv").rename(columns={
        "O*NET-SOC Code": "onet_soc_code", "Title": "occupation", "Job Title": "alt_title", "Short Title": "short_title"})
    occupations = pd.read_csv(ONET / "occupation_data.csv").rename(columns={
        "O*NET-SOC Code": "onet_soc_code", "Title": "occupation", "Description": "description"})
    return {"onet_occupation_skills": tech, "onet_titles": titles, "onet_occupations": occupations}
