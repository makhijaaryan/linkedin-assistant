"""RQ3 ablation: free-form (subject, predicate, object) extraction with no ontology, same model and documents.

  python -m extraction.spo --set graph --extractor groq/openai/gpt-oss-20b
Output: data/processed/extractions/spo__<extractor>/run0.jsonl, one {"doc_id", "triples", "stats"} per doc.
"""

import argparse
import json
from pathlib import Path

from extraction.chunk import chunk_text
from extraction.llm import parse_json
from extraction.prompts import SPO_SYSTEM
from extraction.run import OUT, load_docs, slug
from router.router import Router


def extract(router: Router, doc_id: str, text: str, *, private: bool, provider: str, model: str) -> dict:
    cfg = router.cfg["llm"]
    triples, stats = [], {"calls": 0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    for c in chunk_text(text, cfg["chunk_words"][0], cfg["chunk_words"][1], cfg["chunk_overlap"]):
        r = router.chat("extract_spo", [{"role": "system", "content": SPO_SYSTEM},
                                        {"role": "user", "content": f"Text:\n\"\"\"\n{c.text}\n\"\"\""}],
                        private=private, doc_id=doc_id, provider=provider, model=model, max_tokens=2500)
        stats["calls"] += 1
        stats["latency_s"] += r.latency_s
        stats["prompt_tokens"] += r.prompt_tokens
        stats["completion_tokens"] += r.completion_tokens
        out = parse_json(r.text) or {}
        triples += [t for t in out.get("triples", []) if isinstance(t, dict) and t.get("subject") and t.get("object")]
    return {"doc_id": doc_id, "triples": triples, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="graph")
    ap.add_argument("--extractor", default="groq/openai/gpt-oss-20b")
    a = ap.parse_args()
    router = Router()
    provider, model = a.extractor.split("/", 1)
    public = set(router.cfg["public_sources"])
    path = OUT / f"spo__{slug(a.extractor)}" / "run0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["doc_id"] for x in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for row in load_docs(a.set).itertuples():
            private = row.source not in public
            if row.doc_id in done or (private and not router.cfg["providers"][provider]["private_ok"]):
                continue
            try:
                f.write(json.dumps(extract(router, row.doc_id, row.text, private=private, provider=provider,
                                           model=model)) + "\n")
                f.flush()
            except Exception as e:
                print("  FAIL", row.doc_id, type(e).__name__, str(e)[:150])
    print("wrote", path)


if __name__ == "__main__":
    main()
