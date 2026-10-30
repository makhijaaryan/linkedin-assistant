"""Run an extractor over a document set and write one JSONL per (extractor, run). Resumable.

  python -m extraction.run --set test --extractor crf
  python -m extraction.run --set test --extractor benchmark --runs 3         # every model in config benchmark_models
  python -m extraction.run --set test --extractor ollama/llama3.1:8b --runs 1
  python -m extraction.run --set graph --extractor route                    # config routes (jobs: ASU Voyager, CVs: Ollama)

Sets: test (150 gold docs), profiles (40 ranking CVs), graph (profiles + data/processed/graph_pool.txt jobs),
ids:<file> (one doc_id per line). Private docs (source not in config public_sources) are skipped for
hosted providers and recorded as skipped, never sent (D2).
Output: data/processed/extractions/<extractor>/run<k>.jsonl, one {"parsed", "stats"} row per doc.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from extraction import crf, llm
from router.router import Router

OUT = Path("data/processed/extractions")


def slug(extractor: str) -> str:
    return extractor.replace("/", "__").replace(":", "-")


def load_docs(name: str) -> pd.DataFrame:
    jobs = pd.read_parquet("data/processed/jobs.parquet", columns=["doc_id", "source", "title", "company", "text"])
    cvs = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "source", "text"])
    allx = pd.concat([jobs.assign(doc_type="JobPosting"), cvs.assign(doc_type="CV")]).set_index("doc_id")
    frozen = json.loads(Path("eval/frozen/test_sets.json").read_text())
    if name == "test":
        ids = frozen["extraction_test"]
    elif name == "profiles":
        ids = frozen["ranking_profiles"]
    elif name == "graph":
        pool = Path("data/processed/graph_pool.txt")
        ids = frozen["ranking_profiles"] + (pool.read_text().split() if pool.exists() else [])
    elif name.startswith("ids:"):
        ids = Path(name[4:]).read_text().split()
    else:
        raise ValueError(name)
    ids = list(dict.fromkeys(ids))
    return allx.loc[ids].reset_index()


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["parsed"]["doc_id"] for line in open(path) if line.strip()}


def run(docs: pd.DataFrame, extractor: str, run_idx: int, router: Router, workers: int = 1):
    """extractor: 'crf', 'route' (config routes per doc type) or 'provider/model'."""
    public = set(router.cfg["public_sources"])
    path = OUT / slug(extractor) / f"run{run_idx}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    skip_path = path.with_suffix(".skipped.txt")
    done = done_ids(path)
    todo = docs[~docs["doc_id"].isin(done)]
    provider, model = extractor.split("/", 1) if "/" in extractor else (None, None)

    def one(row):
        private = row.source not in public
        if extractor == "crf":
            return crf.extract_document(row.doc_id, row.doc_type, row.text)
        if provider and private and not router.cfg["providers"][provider]["private_ok"]:
            return None, {"skipped": "private"}
        return llm.extract_document(router, row.doc_id, row.doc_type, row.text, private=private, provider=provider,
                                    model=model, run=run_idx, title=getattr(row, "title", None) or None,
                                    company=getattr(row, "company", None) or None)

    t0, n = time.time(), 0
    with open(path, "a") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, row): row.doc_id for row in todo.itertuples()}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                parsed, stats = fut.result()
            except Exception as e:  # keep going; failures are listed and retried on the next run
                print(f"  FAIL {doc_id}: {type(e).__name__}: {str(e)[:200]}")
                continue
            if parsed is None:
                with open(skip_path, "a") as s:
                    s.write(doc_id + "\n")
                continue
            f.write(json.dumps({"parsed": parsed.model_dump(mode="json"), "stats": stats}) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                print(f"  {extractor} run{run_idx}: {n}/{len(todo)} docs, {time.time() - t0:.0f}s")
    print(f"{extractor} run{run_idx}: wrote {n} docs -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--extractor", default="crf", help="crf | route | benchmark | provider/model")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = 1 for ollama, 4 for hosted")
    a = ap.parse_args()
    router = Router()
    docs = load_docs(a.set)
    if a.limit:
        docs = docs.head(a.limit)
    if a.extractor == "benchmark":
        extractors = [f"{m['provider']}/{m['model']}" for m in router.cfg["benchmark_models"]]
    elif a.extractor == "route":
        extractors = ["route"]
    else:
        extractors = [a.extractor]
    for ex in extractors:
        runs = 1 if ex == "crf" else a.runs
        for k in range(runs):
            if ex == "route":
                for dt, task in (("JobPosting", "extract_job"), ("CV", "extract_cv")):
                    p, m = router.route(task)
                    sub = docs[docs["doc_type"] == dt]
                    if len(sub):
                        run(sub, f"{p}/{m}", k, router, a.workers or (1 if p == "ollama" else 4))
            else:
                w = a.workers or (1 if ex.startswith("ollama") or ex == "crf" else 4)
                run(docs, ex, k, router, w)


if __name__ == "__main__":
    main()
