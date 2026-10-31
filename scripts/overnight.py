"""Overnight orchestrator (2026-10-30): starts each stage as soon as its inputs exist, so the pipeline keeps moving
while nobody watches. Idempotent: a finished stage leaves a marker in .cache/overnight/ and is skipped on restart.

  nohup python scripts/overnight.py >> .cache/logs/overnight.log 2>&1 &

Stages: graph (rebuild + R3 sheets + ranking eval/LTR) -> explanations (local explainer, Ollama)
        spo (RQ3 eval) · hosted (interim extraction eval)
        qlora (export -> MLX train -> serve -> student on the 60-doc subset) -> local (qwen2.5, gemma2 on the subset)
        final (extraction eval, Table II, report)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/opt/anaconda3/envs/linkedin-assistant/bin/python"
EX = ROOT / "data/processed/extractions"
MARK = ROOT / ".cache/overnight"
MARK.mkdir(parents=True, exist_ok=True)
running: dict[str, subprocess.Popen] = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ids(path) -> set:
    p = ROOT / path
    return set(p.read_text().split()) if p.exists() else set()


def done_ids(extractor: str, run: int = 0) -> set:
    f = EX / extractor / f"run{run}.jsonl"
    return {json.loads(x)["parsed"]["doc_id"] for x in open(f) if x.strip()} if f.exists() else set()


def alive(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def sh(*cmd, name=None) -> bool:
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        log(f"FAILED ({r.returncode}): {name or cmd[2]}")
    return r.returncode == 0


def stage_done(name):
    return (MARK / name).exists()


def finish(name):
    (MARK / name).touch()
    log(f"== stage {name} done")


def graph_ready() -> bool:
    pool, pub = ids("data/processed/graph_pool.txt"), ids("data/processed/profiles_public.txt")
    r0 = done_ids("asu__gpt-oss-120b")
    pool_done = len(r0 & pool) / len(pool)
    return (pool_done >= 0.99 or (pool_done >= 0.95 and not alive("graph_pool.txt"))) and pub <= r0


def stage_graph():
    keep = sorted(ids("data/processed/profiles_public.txt") | ids("data/processed/graph_pool.txt"))
    (ROOT / "data/processed/graph_build_ids.txt").write_text("\n".join(keep))
    if not sh(PY, "-m", "graph.build", "--wipe", "--no-llm", "--extractions", "asu__gpt-oss-120b",
              "--only", "data/processed/graph_build_ids.txt"):
        return
    (ROOT / ".cache/graph_rebuilt").touch()
    for f in ("skill_mentions_500.csv", "edge_sample_100.csv"):
        if (ROOT / "data/processed/annotation" / f).exists():
            log(f"keeping existing {f}")
            break
    else:
        sh(PY, "-m", "graph.annotate", "--make", "--extractions", "asu__gpt-oss-120b")
    sh(PY, "-m", "eval.ranking_eval", "--labels", "llm_label", "--save-models")
    finish("graph")


def stage_explanations():
    # local explainer + critic run on Ollama next to the local benchmark (2 parallel slots)
    if not (ROOT / "data/processed/annotation/explanations_60.csv").exists():
        if not sh(PY, "-m", "eval.explanation_eval", "--make"):
            return
    finish("explanations")


def stage_spo():
    if sh(PY, "-m", "eval.spo_eval", "--spo", "spo__asu__gpt-oss-120b", "--ontology", "asu__gpt-oss-120b"):
        finish("spo")


def stage_hosted_eval():
    sh(PY, "-m", "eval.extraction_eval")
    sh(PY, "-m", "eval.report")
    finish("hosted")


def local_done(model_dir) -> bool:
    return ids("data/processed/test_local60.txt") <= done_ids(model_dir)


def stage_qlora():
    if not sh("scripts/qlora_mlx.sh", "export", name="qlora export"):
        return
    subprocess.run(["ollama", "stop", "llama3.1:8b"], cwd=ROOT)
    subprocess.run(["ollama", "stop", "qwen2.5:7b"], cwd=ROOT)
    if not sh("scripts/qlora_mlx.sh", "train", name="qlora train"):
        return
    server = subprocess.Popen(["scripts/qlora_mlx.sh", "serve"], cwd=ROOT,
                              stdout=open(ROOT / ".cache/logs/mlx_server.log", "a"), stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:8090/v1/models", timeout=5)
            break
        except Exception:
            time.sleep(5)
    sh(PY, "-u", "-m", "extraction.run", "--set", "ids:data/processed/test_local60.txt", "--extractor",
       "mlx/qwen2.5-7b-qlora", "--runs", "1", "--workers", "1")
    server.terminate()
    finish("qlora")


def stage_local_rest():
    for m in ("ollama/qwen2.5:7b", "ollama/gemma2:9b"):
        sh(PY, "-u", "-m", "extraction.run", "--set", "ids:data/processed/test_local60.txt", "--extractor", m,
           "--runs", "1", "--workers", "2")
    finish("local")


def stage_final():
    sh(PY, "-m", "eval.extraction_eval")
    sh(PY, "-m", "eval.report")
    sh(PY, "report/build.py")
    finish("final")


def start(name, fn):
    """Run a long stage in a child process so the loop keeps checking the others."""
    if name in running or stage_done(name):
        return
    log(f"== stage {name} start")
    running[name] = subprocess.Popen([PY, __file__, "--stage", name], cwd=ROOT)


STAGES = {"graph": stage_graph, "explanations": stage_explanations, "spo": stage_spo, "hosted": stage_hosted_eval,
          "qlora": stage_qlora, "local": stage_local_rest, "final": stage_final}

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--stage":
        STAGES[sys.argv[2]]()
        sys.exit(0)
    log("overnight orchestrator up")
    hosted_procs = ("asu/llama4-maverick-17b", "asu/gemma4-31b-it", "azure/gpt-5-mini", "--set test --extractor asu/gpt-oss")
    while not stage_done("final"):
        for name, p in list(running.items()):
            if p.poll() is not None:
                log(f"stage {name} exited ({p.returncode})")
                del running[name]
        if graph_ready():
            start("graph", stage_graph)
        if stage_done("graph"):
            start("explanations", stage_explanations)
        spo = EX / "spo__asu__gpt-oss-120b/run0.jsonl"
        spo_n = sum(1 for x in open(spo) if x.strip()) if spo.exists() else 0
        if stage_done("graph") and (spo_n >= 0.98 * len(ids("data/processed/spo_ids.txt")) or
                                    (spo_n and not alive("extraction.spo"))):
            start("spo", stage_spo)
        if not any(alive(p) for p in hosted_procs):
            start("hosted", stage_hosted_eval)
        teacher = ids("data/processed/distill/teacher_ids_600.txt")
        teacher_ok = len(done_ids("asu__gpt-oss-120b") & teacher) >= 0.98 * len(teacher) or (
            not alive("teacher_ids_600") and not alive("while kill -0"))
        if (teacher_ok and local_done("ollama__llama3.1-8b") and stage_done("explanations")
                and not alive("extraction.run --set ids:data/processed/test_local60.txt --extractor ollama")):
            start("qlora", stage_qlora)
        if stage_done("qlora"):
            start("local", stage_local_rest)
        if all(stage_done(s) for s in ("graph", "explanations", "spo", "hosted", "qlora", "local")):
            start("final", stage_final)
        time.sleep(120)
    log("all stages done")
