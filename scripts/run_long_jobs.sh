#!/usr/bin/env bash
# Long-running extraction jobs, in two queues that run side by side (all resumable; re-run to continue).
#   Hosted queue: RQ1 benchmark (3 ASU Voyager models + Azure gpt-5-mini baseline; 3 runs) -> graph-pool jobs
#                 -> RQ2 teacher set (D31). Public Kaggle docs only; private docs are skipped by the runner (D2).
#   Ollama queue: graph profiles' CVs (route extract_cv) -> local benchmark models, run0 (D25)
# Logs: .cache/logs/queue_{hosted,ollama}.log      Stop:  pkill -f run_long_jobs.sh; pkill -f extraction.run
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/linkedin-assistant/bin/python
mkdir -p .cache/logs

# One stream per hosted model, side by side (each model is its own backend; Voyager scales to ~4 parallel calls
# per model, measured 2026-10-30). gpt-oss-120b then does the graph pool and the RQ2 teacher set (D26, D31).
hosted_queue() {
  for m in asu/llama4-maverick-17b asu/gemma4-31b-it azure/gpt-5-mini; do
    $PY -u -m extraction.run --set test --extractor "$m" --runs 3 --workers 4 &
  done
  $PY -u -m extraction.run --set test --extractor asu/gpt-oss-120b --runs 3 --workers 4
  $PY -u -m extraction.run --set ids:data/processed/graph_pool.txt --extractor asu/gpt-oss-120b --workers 4
  $PY -u -m extraction.run --set ids:data/processed/distill/teacher_ids.txt --extractor asu/gpt-oss-120b --workers 4
  wait
}

ollama_queue() {
  $PY -u -m extraction.run --set profiles --extractor ollama/llama3.1:8b
  for m in ollama/llama3.1:8b ollama/qwen2.5:7b ollama/gemma2:9b; do
    $PY -u -m extraction.run --set test --extractor "$m" --runs 1
  done
}

case "${1:-both}" in
  hosted) hosted_queue >> .cache/logs/queue_hosted.log 2>&1 ;;
  ollama) ollama_queue >> .cache/logs/queue_ollama.log 2>&1 ;;
  both)
    hosted_queue >> .cache/logs/queue_hosted.log 2>&1 &
    ollama_queue >> .cache/logs/queue_ollama.log 2>&1 &
    wait ;;
esac
