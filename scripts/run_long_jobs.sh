#!/usr/bin/env bash
# Long-running extraction jobs, in two queues that run side by side (all resumable; re-run to continue).
#   Groq queue:   RQ1 benchmark (gpt-oss-20b, gpt-oss-120b; 3 runs) -> graph-pool jobs -> RQ2 teacher set
#   Ollama queue: graph profiles' CVs (route extract_cv) -> local benchmark models, run0 (D25)
# Logs: .cache/logs/queue_{groq,ollama}.log      Stop:  pkill -f run_long_jobs.sh; pkill -f extraction.run
# Free-tier limits (1K req/day/model on Groq) mean the Groq queue spans ~2 days; just re-run it the next day.
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/linkedin-assistant/bin/python
mkdir -p .cache/logs

groq_queue() {
  for m in groq/openai/gpt-oss-20b groq/openai/gpt-oss-120b; do
    $PY -u -m extraction.run --set test --extractor "$m" --runs 3 --workers 2
  done
  $PY -u -m extraction.run --set ids:data/processed/graph_pool.txt --extractor groq/openai/gpt-oss-20b --workers 2
  # RQ2 teacher labels (D26): ~3K calls on gpt-oss-120b, i.e. ~3 days of free quota
  $PY -u -m extraction.run --set ids:data/processed/distill/teacher_ids.txt --extractor groq/openai/gpt-oss-120b --workers 2
}

ollama_queue() {
  $PY -u -m extraction.run --set profiles --extractor ollama/llama3.1:8b
  for m in ollama/llama3.1:8b ollama/qwen2.5:7b ollama/gemma2:9b; do
    $PY -u -m extraction.run --set test --extractor "$m" --runs 1
  done
}

case "${1:-both}" in
  groq)   groq_queue   >> .cache/logs/queue_groq.log 2>&1 ;;
  ollama) ollama_queue >> .cache/logs/queue_ollama.log 2>&1 ;;
  both)
    groq_queue   >> .cache/logs/queue_groq.log 2>&1 &
    ollama_queue >> .cache/logs/queue_ollama.log 2>&1 &
    wait ;;
esac
