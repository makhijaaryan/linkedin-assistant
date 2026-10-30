"""R2 / RQ1 extraction benchmark: per-type P/R/F1 against the 150 gold docs, plus the pre-registered tests.

  python -m eval.extraction_eval                         # gold = Label Studio export (see GOLD)
  python -m eval.extraction_eval --dry-run-gold asu__gpt-oss-120b   # pipeline check only, NOT a result

Matching rule (D21, fixed before any gold exists): per doc and label, gold spans with the same normalized
text form one entity (any mention counts). A predicted entity matches a gold entity of the same label if
their normalized texts are equal, their token-F1 is >= 0.5, or a predicted span covers >= 50% of the
shorter of it and a gold mention. Matching is one-to-one. The 8 labels follow the annotation guide.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from eval import stats

GOLD = Path("data/processed/annotation/gold_extraction.json")
EXTRACTIONS = Path("data/processed/extractions")
RESULTS = Path("eval/results")
LABELS = ["Skill", "Requirement", "Experience", "Credential", "Project", "Company", "Occupation", "Location"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", (s or "").lower()).strip()


def token_f1(a: str, b: str) -> float:
    x, y = set(norm(a).split()), set(norm(b).split())
    if not x or not y:
        return 0.0
    k = len(x & y)
    return 2 * k / (len(x) + len(y))


def find_span(text: str, name: str, region: tuple | None = None):
    if not name:
        return None
    pat = r"\s+".join(map(re.escape, name.split()))
    if region and region[0] is not None:
        m = re.search(pat, text[region[0]:region[1]], re.IGNORECASE)
        if m:
            return region[0] + m.start(), region[0] + m.end()
    m = re.search(pat, text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def load_gold(path: Path = GOLD) -> dict:
    """Label Studio JSON export -> {doc_id: {"text", "doc_type", "entities": {label: {norm_text: [spans]}}}}."""
    gold = {}
    for task in json.loads(path.read_text()):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ents = defaultdict(lambda: defaultdict(list))
        for r in anns[-1]["result"]:
            if r.get("type") != "labels":
                continue
            v = r["value"]
            ents[v["labels"][0]][norm(v["text"])].append((v["start"], v["end"]))
        d = task["data"]
        gold[d["doc_id"]] = {"text": d["text"], "doc_type": d["doc_type"], "entities": ents}
    return gold


def pred_entities(parsed: dict, text: str) -> dict:
    """Map a ParsedCV/ParsedJob onto the 8 annotation labels: {label: {norm_text: [spans]}}."""
    ents = defaultdict(lambda: defaultdict(list))

    def ev_span(item):
        ev = item.get("evidence") or {}
        return (ev.get("char_start"), ev.get("char_end"))

    def add(label, name, span):
        if norm(name):
            ents[label][norm(name)]  # register even without a span (text-only match)
            if span and span[0] is not None:
                ents[label][norm(name)].append(tuple(span))

    def add_loc(loc):
        if loc:
            q = (loc.get("evidence") or {}).get("quote", "")
            name = ", ".join(x for x in (loc.get("city"), loc.get("state"), loc.get("country")) if x) or q
            add("Location", "Remote" if loc.get("is_remote") and not loc.get("city") else name, ev_span(loc))

    if "job" in parsed:
        job = parsed["job"]
        if job.get("title") and job["title"] != "Unknown":
            add("Occupation", job["title"], find_span(text, job["title"]))
        if job.get("company"):
            add("Company", job["company"], find_span(text, job["company"]))
        add_loc(job.get("location"))
        for r in parsed["requirements"]:
            span = ev_span(r)
            add("Requirement", r["text"], span)
            if r["requirement_type"] in ("Education", "Credential"):
                add("Credential", r["text"], span)
            for s in r.get("skills", []):
                add("Skill", s, find_span(text, s, span))
    else:
        cand = parsed.get("candidate") or {}
        if cand.get("headline"):
            add("Occupation", cand["headline"], find_span(text, cand["headline"]))
        add_loc(cand.get("location"))
        for s in parsed.get("skills", []):
            add("Skill", s["name"], find_span(text, s["name"], ev_span(s)) or ev_span(s))
        for x in parsed.get("experiences", []):
            span = ev_span(x)
            add("Experience", x["title"] + " " + (x.get("company") or ""), span)
            add("Occupation", x["title"], find_span(text, x["title"], span))
            if x.get("company"):
                add("Company", x["company"], find_span(text, x["company"], span))
            add_loc(x.get("location"))
            for s in x.get("skills_used", []):
                add("Skill", s, find_span(text, s, span))
        for c in parsed.get("credentials", []):
            add("Credential", c["name"], ev_span(c))
        for p in parsed.get("projects", []):
            add("Project", p["name"], ev_span(p))
            for s in p.get("skills_used", []):
                add("Skill", s, find_span(text, s, ev_span(p)))
    return ents


def _overlap(a, b) -> bool:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return inter > 0 and inter >= 0.5 * min(a[1] - a[0], b[1] - b[0])


def matches(p_text, p_spans, g_text, g_spans) -> bool:
    if p_text == g_text or token_f1(p_text, g_text) >= 0.5:
        return True
    return any(_overlap(a, b) for a in p_spans for b in g_spans)


def score_doc(gold_ents: dict, pred_ents: dict) -> dict:
    """{label: {"tp", "fp", "fn", "gold_hits": {gold_text: bool}}}"""
    out = {}
    for label in LABELS:
        g, p = gold_ents.get(label, {}), pred_ents.get(label, {})
        used, hits = set(), {}
        for gt, gs in g.items():
            hit = next((pt for pt, ps in p.items() if pt not in used and matches(pt, ps, gt, gs)), None)
            hits[gt] = hit is not None
            if hit is not None:
                used.add(hit)
        tp = sum(hits.values())
        out[label] = {"tp": tp, "fp": len(p) - len(used), "fn": len(g) - tp, "gold_hits": hits}
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load_runs(extractor_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(extractor_dir.glob("run*.jsonl")):
        rows = [json.loads(line) for line in open(f) if line.strip()]
        runs.append({r["parsed"]["doc_id"]: r for r in rows})
    return runs


def evaluate_extractor(gold: dict, runs: list[dict]) -> dict:
    docs = sorted(set(gold) & set.intersection(*(set(r) for r in runs)))
    per_run, doc_f1 = [], defaultdict(list)
    entity_hits = defaultdict(list)       # (doc, label, gold_text) -> [hit per run]
    for run in runs:
        tot = defaultdict(lambda: [0, 0, 0])
        for d in docs:
            sc = score_doc(gold[d]["entities"], pred_entities(run[d]["parsed"], gold[d]["text"]))
            t = [sum(sc[lab][k] for lab in LABELS) for k in ("tp", "fp", "fn")]
            doc_f1[d].append(100 * prf(*t)[2])
            for lab in LABELS:
                for k, i in (("tp", 0), ("fp", 1), ("fn", 2)):
                    tot[lab][i] += sc[lab][k]
                for gt, h in sc[lab]["gold_hits"].items():
                    entity_hits[(d, lab, gt)].append(h)
        per_type = {lab: dict(zip(("p", "r", "f1"), prf(*tot[lab]))) for lab in LABELS}
        micro = prf(*(sum(tot[lab][i] for lab in LABELS) for i in range(3)))
        per_run.append({"per_type": per_type, "micro_f1": micro[2]})
    summary = {lab: {m: [float(np.mean([r["per_type"][lab][m] for r in per_run])),
                         float(np.std([r["per_type"][lab][m] for r in per_run]))] for m in ("p", "r", "f1")}
               for lab in LABELS}
    micro = [r["micro_f1"] for r in per_run]
    return {"docs": docs, "n_docs": len(docs), "runs": len(runs), "per_type_mean_sd": summary,
            "micro_f1_mean_sd": [float(np.mean(micro)), float(np.std(micro))],
            "doc_f1": {d: float(np.mean(v)) for d, v in doc_f1.items()},
            "entity_hit": {"|".join(k): bool(np.mean(v) >= 0.5) for k, v in entity_hits.items()}}


def efficiency(runs: list[dict], docs: list[str], prices: dict, model: str, local: bool) -> dict:
    """Per-doc latency (sum over a doc's calls) and $/1K docs at list price, from extraction stats."""
    lat = [np.mean([r[d]["stats"].get("latency_s", 0) for r in runs]) for d in docs]
    toks = [(np.mean([r[d]["stats"].get("prompt_tokens", 0) for r in runs]),
             np.mean([r[d]["stats"].get("completion_tokens", 0) for r in runs])) for d in docs]
    price = [0, 0] if local else prices.get(model)
    cost = None if price is None else float(1000 * np.mean([(pt * price[0] + ct * price[1]) / 1e6 for pt, ct in toks]))
    q = lambda x: stats.bootstrap_ci(lat, stat=lambda v: np.quantile(v, x)) if lat else {}
    return {"latency_per_doc": dict(zip(docs, map(float, lat))), "p50_s": q(0.5), "p95_s": q(0.95),
            "usd_per_1k_docs": cost, "mean_prompt_tokens": float(np.mean([t[0] for t in toks])) if toks else 0,
            "mean_completion_tokens": float(np.mean([t[1] for t in toks])) if toks else 0}


def compare(results: dict, proprietary: str, baseline: str, delta: float) -> dict:
    """Family A (Holm): each open model vs. proprietary. Plus each LLM vs. CRF (+10 target)."""
    out, pvals = {}, {}
    prop = results.get(proprietary)
    for name, res in results.items():
        if name in (proprietary, baseline):
            continue
        row = {}
        if prop:
            common = sorted(set(res["doc_f1"]) & set(prop["doc_f1"]))
            a = [res["doc_f1"][d] for d in common]
            b = [prop["doc_f1"][d] for d in common]
            row["vs_proprietary"] = {**stats.noninferiority(a, b, delta), **stats.randomization_test(a, b)}
            ents = sorted(set(res["entity_hit"]) & set(prop["entity_hit"]))
            row["vs_proprietary"]["mcnemar"] = stats.mcnemar([res["entity_hit"][e] for e in ents],
                                                             [prop["entity_hit"][e] for e in ents])
            pvals[name] = row["vs_proprietary"]["p"]
            pl, ol = res.get("efficiency", {}).get("latency_per_doc", {}), prop.get("efficiency", {}).get("latency_per_doc", {})
            lc = sorted(set(pl) & set(ol))
            if lc:
                row["latency_vs_proprietary"] = stats.wilcoxon([pl[d] for d in lc], [ol[d] for d in lc])
        if baseline in results:
            common = sorted(set(res["doc_f1"]) & set(results[baseline]["doc_f1"]))
            ci = stats.paired_bootstrap([res["doc_f1"][d] for d in common],
                                        [results[baseline]["doc_f1"][d] for d in common])
            row["vs_crf"] = {**ci, "target_plus10_met": bool(ci["ci_low"] >= 10)}
        out[name] = row
    for name, p in stats.holm(pvals).items():
        out[name]["vs_proprietary"]["p_holm"] = p
    return out


def markdown(results: dict, comparisons: dict, dry_run: bool) -> str:
    lines = ["# Extraction benchmark (R2 / RQ1)", ""]
    if dry_run:
        lines += ["**DRY RUN: the 'gold' here is another model's output. Pipeline check only, not a result.**", ""]
    lines += ["| Extractor | Docs | Runs | Micro-F1 | " + " | ".join(LABELS) + " | p50 s/doc | $/1K docs |",
              "|---" * (6 + len(LABELS)) + "|"]
    for name, r in results.items():
        f1 = " | ".join(f"{100 * r['per_type_mean_sd'][lab]['f1'][0]:.1f}" for lab in LABELS)
        eff = r.get("efficiency", {})
        cost = eff.get("usd_per_1k_docs")
        lines.append(f"| {name} | {r['n_docs']} | {r['runs']} | {100 * r['micro_f1_mean_sd'][0]:.1f} ± "
                     f"{100 * r['micro_f1_mean_sd'][1]:.1f} | {f1} | {eff.get('p50_s', {}).get('estimate', 0):.1f} | "
                     f"{'n/a' if cost is None else f'{cost:.3f}'} |")
    lines += ["", "| Open model | ΔF1 vs proprietary [95% CI] | p (Holm) | Non-inferior (δ=3) | ΔF1 vs CRF [95% CI] | +10 met |",
              "|---|---|---|---|---|---|"]
    for name, c in comparisons.items():
        v, k = c.get("vs_proprietary"), c.get("vs_crf")
        a = f"{v['mean_diff']:+.1f} [{v['ci_low']:+.1f}, {v['ci_high']:+.1f}] | {v.get('p_holm', v['p']):.4f} | " \
            f"{'✓' if v['noninferior'] else '✗'}" if v else "– | – | –"
        b = f"{k['mean_diff']:+.1f} [{k['ci_low']:+.1f}, {k['ci_high']:+.1f}] | {'✓' if k['target_plus10_met'] else '✗'}" \
            if k else "– | –"
        lines.append(f"| {name} | {a} | {b} |")
    return "\n".join(lines) + "\n"


def gold_from_extractor(name: str, test_docs: dict) -> dict:
    """Dry run only: treat one extractor's run0 as 'gold' to exercise the pipeline end to end."""
    run0 = load_runs(EXTRACTIONS / name)[0]
    return {d: {"text": test_docs[d]["text"], "doc_type": test_docs[d]["doc_type"],
                "entities": pred_entities(run0[d]["parsed"], test_docs[d]["text"])}
            for d in run0 if d in test_docs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-gold", default="", help="extractor dir to use as pseudo-gold (pipeline check)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    test_docs = {json.loads(l)["doc_id"]: json.loads(l) for l in open("data/processed/test_sets/extraction_test.jsonl")}
    gold = gold_from_extractor(a.dry_run_gold, test_docs) if a.dry_run_gold else load_gold()
    prop = next(f"{m['provider']}__{m['model']}".replace("/", "__").replace(":", "-")
                for m in cfg["benchmark_models"] if m.get("proprietary"))
    local = {f"{m['provider']}__{m['model']}".replace("/", "__").replace(":", "-")
             for m in cfg["benchmark_models"] if cfg["providers"][m["provider"]]["private_ok"]}
    # only the configured benchmark (+ CRF, + the QLoRA student): older partial runs (Groq, Gemini; D31) stay on disk
    wanted = {f"{m['provider']}__{m['model']}".replace("/", "__").replace(":", "-") for m in cfg["benchmark_models"]}
    results = {}
    for d in sorted(p for p in EXTRACTIONS.iterdir() if p.is_dir() and not p.name.startswith("spo")):
        if d.name == a.dry_run_gold or not (d.name in wanted or d.name == "crf" or "qlora" in d.name):
            continue
        runs = [r for r in load_runs(d) if r]
        if not runs or not set(gold) & set.intersection(*(set(r) for r in runs)):
            continue
        res = evaluate_extractor(gold, runs)
        model = d.name.split("__", 1)[-1].replace("__", "/")
        res["efficiency"] = efficiency(runs, res["docs"], cfg.get("prices_per_1m") or {}, model,
                                       d.name in local or d.name == "crf")
        results[d.name] = res
    comparisons = compare(results, prop, "crf", cfg["stats"]["noninferiority_delta"]["extraction_f1"])
    RESULTS.mkdir(exist_ok=True, parents=True)
    tag = "_DRYRUN" if a.dry_run_gold else ""
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("entity_hit", "docs")} for k, v in results.items()}
    (RESULTS / f"extraction_benchmark{tag}.json").write_text(json.dumps(
        {"gold": "dry-run:" + a.dry_run_gold if a.dry_run_gold else str(GOLD), "results": slim,
         "comparisons": comparisons}, indent=1))
    md = markdown(results, comparisons, bool(a.dry_run_gold))
    (RESULTS / f"extraction_benchmark{tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
