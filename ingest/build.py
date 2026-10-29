"""Phase 1 (R1) pipeline: load -> clean/mask -> dedupe -> working set -> splits -> test sets -> weak labels.

Run from the repo root:  python -m ingest.build
Private outputs (contain text) go to data/processed/ (git-ignored).
Frozen, text-free manifests go to eval/frozen/ and the report to eval/results/ (committed).
"""

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from ingest.dedupe import find_duplicates, handcheck_sample, signatures
from ingest.load import load_jobs, load_resumes
from ingest.split import assign_splits, leakage_report, round_robin, stratified_sample, top_k_or_other
from ingest.weak_labels import job_weak_labels, onet_tables

PROCESSED = Path("data/processed")
FROZEN = Path("eval/frozen")
RESULTS = Path("eval/results")
EXTRACTION_LABELS = ["Skill", "Requirement", "Experience", "Credential", "Project", "Company", "Occupation", "Location"]


def label_studio_tasks(docs: pd.DataFrame, doc_type: str) -> list[dict]:
    return [{"data": {"doc_id": r.doc_id, "doc_type": doc_type, "text": r.text}} for r in docs.itertuples()]


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    seed, dcfg = cfg["seed"], cfg["data"]
    dd, ts = dcfg["dedupe"], dcfg["test_sets"]
    for d in (PROCESSED / "annotation", PROCESSED / "test_sets", PROCESSED / "weak_labels", FROZEN, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    report, t0 = {"seed": seed}, time.time()

    # 1) load + clean + mask
    jobs, s1 = load_jobs()
    resumes, s2 = load_resumes()
    report["load"] = {**s1, **s2}
    print("loaded", report["load"])

    # 2) dedupe
    job_sigs = signatures(jobs["text"], dd["num_perm"], dd["shingle_words"], seed)
    cv_sigs = signatures(resumes["text"], dd["num_perm"], dd["shingle_words"], seed)
    jobs["cluster_id"], job_pairs = find_duplicates(jobs, ["title", "company"], job_sigs, dd["lsh_threshold"])
    resumes["cluster_id"], cv_pairs = find_duplicates(resumes, ["source"], cv_sigs, dd["lsh_threshold"])
    for df in (jobs, resumes):
        df["is_canonical"] = df["doc_id"] == df["cluster_id"]
    report["dedupe"] = {
        "job_clusters": int(jobs["cluster_id"].nunique()), "job_duplicates_removed": int((~jobs["is_canonical"]).sum()),
        "job_near_dup_pairs": len(job_pairs),
        "resume_clusters": int(resumes["cluster_id"].nunique()),
        "resume_duplicates_removed": int((~resumes["is_canonical"]).sum()), "resume_near_dup_pairs": len(cv_pairs),
    }
    print("dedupe", report["dedupe"])
    k = dd["handcheck_pairs"]
    handcheck_sample(jobs, job_pairs, k, seed, "title", "company").to_csv(
        PROCESSED / "annotation/dedupe_handcheck_jobs.csv", index=False)
    handcheck_sample(resumes, cv_pairs, k // 4, seed).to_csv(
        PROCESSED / "annotation/dedupe_handcheck_resumes.csv", index=False)

    # 3) strata, splits (per cluster) and working set (canonical docs only)
    jobs["industry_group"] = top_k_or_other(jobs["industry"], 25)
    job_strata = jobs["industry_group"] + " | " + jobs["seniority"]
    cv_strata = resumes["source"] + " | " + resumes["category"]
    jobs["split"] = assign_splits(jobs, job_strata, dcfg["splits"], seed)
    resumes["split"] = assign_splits(resumes, cv_strata, dcfg["splits"], seed)

    canon_jobs = jobs[jobs["is_canonical"]]
    ws_jobs = stratified_sample(canon_jobs, job_strata, dcfg["working_set"]["jobs"], seed)
    jobs["in_working_set"] = jobs["doc_id"].isin(ws_jobs["doc_id"])
    canon_cv = resumes[resumes["is_canonical"]]
    ws_cv = pd.concat([
        canon_cv[canon_cv["source"] == "kaggle_resume_dataset"],
        stratified_sample(canon_cv[canon_cv["source"] == "resume_corpus"], cv_strata,
                          dcfg["working_set"]["resumes_corpus"], seed),
    ])
    resumes["in_working_set"] = resumes["doc_id"].isin(ws_cv["doc_id"])
    report["working_set"] = {"jobs": int(jobs["in_working_set"].sum()), "resumes": int(resumes["in_working_set"].sum())}
    report["splits"] = {"jobs": jobs["split"].value_counts().to_dict(), "resumes": resumes["split"].value_counts().to_dict()}
    print("working set", report["working_set"], "splits", report["splits"])

    # 4) leakage checks
    job_leak, job_flagged = leakage_report(jobs, job_sigs)
    cv_leak, cv_flagged = leakage_report(resumes, cv_sigs)
    report["leakage"] = {"jobs": job_leak, "resumes": cv_leak}
    print("leakage", report["leakage"])

    # 5) test sets from test split ∩ working set ∩ canonical, minus docs with a near-identical train doc
    pool_j = jobs[jobs["is_canonical"] & jobs["in_working_set"] & (jobs["split"] == "test")
                  & ~jobs["doc_id"].isin(job_flagged)]
    pool_cv = resumes[resumes["is_canonical"] & resumes["in_working_set"] & (resumes["split"] == "test")
                      & ~resumes["doc_id"].isin(cv_flagged)]
    kag, cor = pool_cv[pool_cv["source"] == "kaggle_resume_dataset"], pool_cv[pool_cv["source"] == "resume_corpus"]
    ext_jobs = round_robin(pool_j, pool_j["industry_group"], ts["extraction_jobs"], seed)
    ext_kag = round_robin(kag, kag["category"], ts["extraction_resumes_kaggle"], seed)
    ext_cor = round_robin(cor, cor["category"], ts["extraction_resumes_corpus"], seed)
    rank_kag = round_robin(kag.drop(ext_kag.index), kag["category"], ts["ranking_profiles_kaggle"], seed)
    rank_cor = round_robin(cor.drop(ext_cor.index), cor["category"], ts["ranking_profiles_corpus"], seed)
    extraction = pd.concat([ext_jobs.assign(doc_type="JobPosting"),
                            pd.concat([ext_kag, ext_cor]).assign(doc_type="CV")])
    profiles = pd.concat([rank_kag, rank_cor])
    extraction[["doc_id", "doc_type", "text"]].to_json(PROCESSED / "test_sets/extraction_test.jsonl", orient="records", lines=True)
    profiles[["doc_id", "category", "text"]].to_json(PROCESSED / "test_sets/ranking_profiles.jsonl", orient="records", lines=True)
    tasks = (label_studio_tasks(ext_jobs, "JobPosting") + label_studio_tasks(pd.concat([ext_kag, ext_cor]), "CV"))
    (PROCESSED / "annotation/extraction_tasks.json").write_text(json.dumps(tasks))
    test_ids = {"extraction_test": extraction["doc_id"].tolist(), "ranking_profiles": profiles["doc_id"].tolist()}
    (FROZEN / "test_sets.json").write_text(json.dumps(test_ids, indent=1))
    report["test_sets"] = {"extraction_docs": len(extraction), "extraction_jobs": len(ext_jobs),
                           "extraction_cvs": len(ext_kag) + len(ext_cor), "ranking_profiles": len(profiles)}

    # 6) weak labels (training only)
    job_weak_labels().to_parquet(PROCESSED / "weak_labels/job_weak_labels.parquet", index=False)
    for name, table in onet_tables().items():
        table.to_parquet(PROCESSED / f"weak_labels/{name}.parquet", index=False)
    resumes[["doc_id", "labels"]].to_parquet(PROCESSED / "weak_labels/resume_labels.parquet", index=False)

    # 7) save processed tables + frozen manifests
    jobs.to_parquet(PROCESSED / "jobs.parquet", index=False)
    resumes.to_parquet(PROCESSED / "resumes.parquet", index=False)
    manifest_cols = ["doc_id", "cluster_id", "is_canonical", "split", "in_working_set"]
    pd.concat([jobs[manifest_cols].assign(kind="job"), resumes[manifest_cols].assign(kind="resume")]).to_csv(
        FROZEN / "splits.csv.gz", index=False)

    report["seconds"] = round(time.time() - t0, 1)
    (RESULTS / "phase1_report.json").write_text(json.dumps(report, indent=2, default=int))
    print(f"done in {report['seconds']}s -> {PROCESSED}/, {FROZEN}/, {RESULTS}/phase1_report.json")


if __name__ == "__main__":
    main()
