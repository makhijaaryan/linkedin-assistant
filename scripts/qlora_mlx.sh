#!/usr/bin/env bash
# RQ2 student on the laptop (D34): teacher export -> MLX LoRA on Qwen2.5-7B-Instruct-4bit -> local OpenAI-compatible
# server on :8090 (config provider `mlx`, private_ok). Needs Ollama idle (~8–10 GB). Usage:
#   scripts/qlora_mlx.sh export | train | serve
set -eu
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/linkedin-assistant/bin/python
D=.cache/qlora
case "$1" in
  export)
    $PY -m extraction.distill --export asu__gpt-oss-120b
    mkdir -p $D/data
    cp data/processed/distill/train.jsonl $D/data/train.jsonl
    cp data/processed/distill/val.jsonl $D/data/valid.jsonl
    wc -l $D/data/*.jsonl ;;
  train)
    n=$(wc -l < $D/data/train.jsonl)
    iters=$n                            # mlx counts micro-batches: one epoch at batch 1 (updates every 16)
    sed "s/^iters: .*/iters: $iters/; s/arguments: \[2.0e-4, [0-9]*, 1.0e-6\]/arguments: [2.0e-4, $iters, 1.0e-6]/" \
      extraction/qlora/lora_config.yaml > $D/lora_config.yaml
    echo "training $iters micro-batches ($(( iters / 16 )) optimizer steps) on $n examples"
    $PY -m mlx_lm lora --config $D/lora_config.yaml 2>&1 | tee .cache/logs/qlora_train.log
    touch .cache/qlora_done ;;
  serve)
    exec $PY -m mlx_lm server --model mlx-community/Qwen2.5-7B-Instruct-4bit --adapter-path $D/adapters --port 8090 ;;
esac
