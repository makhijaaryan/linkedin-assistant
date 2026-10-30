# LinkedIn Assistant — Explainable, Open-Weight LLM Job Matching

CSE 573 (Fall 2026), Group 3. See `IMPLEMENTATION_STEPS.md` for the roadmap and `DECISIONS.md` for the decision log.

## Setup (macOS arm64; Linux is similar)

Prerequisites: Homebrew, conda, Docker Desktop.

```bash
# 1. Python environment
conda create -n linkedin-assistant python=3.11 -y
conda activate linkedin-assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Secrets
cp .env.example .env        # then set NEO4J_PASSWORD, GROQ_API_KEY, OPENROUTER_API_KEY

# 3. Neo4j 5.26 + APOC + GDS (Louvain), in Docker
docker compose --env-file .env up -d     # browser at http://localhost:7474

# 4. Local LLMs
brew install ollama && brew services start ollama
ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull gemma2:9b

# 5. Kaggle credentials (for dataset download)
kaggle auth login

# 6. Verify everything
python scripts/check_env.py
```

QLoRA training (R5) needs CUDA, so run it on Colab/Kaggle with `pip install -r requirements-gpu.txt`.

## Running the pipeline

Run everything from the repo root, inside the `linkedin-assistant` env. Steps marked 👤 need a person.

| Phase | Command | Output |
|---|---|---|
| 1 Data & ontology | `python -m ontology.build` · `python -m ingest.build` | `data/processed/`, `eval/frozen/` |
| 2 CRF baseline | `python -m extraction.crf --train` · `python -m extraction.run --set test --extractor crf` | `data/processed/extractions/crf/` |
| 2 LLM benchmark | `python -m extraction.run --set test --extractor benchmark --runs 3` (or one `provider/model`) | `data/processed/extractions/<model>/run<k>.jsonl` |
| 2 👤 Gold labels | Label Studio (`eval/annotation/`), export JSON → `data/processed/annotation/gold_extraction.json` | |
| 2 Scores | `python -m eval.extraction_eval` | `eval/results/extraction_benchmark.{json,md}` |
| 4 Indexes | `python -m ranking.index` | `.cache/index/` (BM25, TF-IDF, CRF skills, family LR, K-means, dense) |
| 4 Pool | `python -m ranking.pool --make` | `relevance_pool.csv`, `graph_pool.txt` |
| 3 Graph | `python -m extraction.run --set graph --extractor route` · `python -m graph.build --wipe --extractions <dirs>` · `--snapshot v1` | Neo4j + `data/processed/graph_snapshots/` |
| 3 👤 R3 checks | `python -m graph.annotate --make`, fill the sheets, `--score` | `eval/results/graph_eval.json` |
| 5 👤 Judging | `streamlit run ui/judge.py` (800 pairs) | `label` column of `relevance_pool.csv` |
| 4–6 Ranking | `python -m eval.ranking_eval` | `eval/results/ranking_eval.{json,md}` + live LambdaMART models |
| 5 RQ3 | `python -m extraction.spo --set graph` · `python -m eval.spo_eval --spo <dir> --ontology <dir>` | `eval/results/spo_ablation.json` |
| 5 RQ2 | `python -m extraction.distill --make-set 1500` → teacher run → `--export <teacher>` → 👤 `notebooks/qlora_extraction.ipynb` on Colab | GGUF for Ollama |
| 6 👤 Faithfulness | `python -m eval.explanation_eval --make`, fill `supported`, `--score` | `eval/results/explanation_eval.json` |
| 7 App | `streamlit run ui/app.py` | Match, Outreach, Find candidates, Router, Dashboard |
| 9 👤 User study | `eval/user_study/PROTOCOL.md` | `eval/results/user_study.json` |
| 8 Summary | `python -m eval.report` | `eval/results/SUMMARY.md` |

Tests: `python -m pytest tests/` (offline; no network, Neo4j or LLM needed).

Privacy: uploaded CVs and `resume_corpus` are private and only ever go to local Ollama. The router raises `PrivacyError` otherwise (`config.yaml → public_sources`). Router logs hold no prompt text.

## Layout

| Folder | Thread |
|---|---|
| `data/` | raw / interim / processed / external (git-ignored) |
| `ontology/` | R1: ontology spec, extraction schema; `python -m ontology.build` → `ontology/generated/` |
| `ingest/` | R1: load, clean/mask, dedupe, splits, test sets; `python -m ingest.build` |
| `extraction/` | R2, R5: chunking, prompts, LLM extraction, CRF baseline, SPO ablation, distillation data |
| `graph/` | R3: O*NET normalization, Neo4j load + evidence audit, RELATED_TO + Louvain, R3 sheets |
| `ranking/` | R4: indexes, lexical, dense, RRF, graph features, LambdaMART, cross-encoder + MMR, live cascade, judging pool |
| `router/` | Model router (privacy guard, cache, retries/fallback) + call logs |
| `agents/` | R6: LangGraph harness (7 agents), explainer + critic, optimizer, outreach/scheduler |
| `ui/` | Streamlit app (`app.py`) and relevance judging tool (`judge.py`) |
| `eval/` | `stats.py`, extraction / ranking / SPO / explanation evaluation, user study kit, `report.py` |
| `notebooks/` | `qlora_extraction.ipynb` (Colab/Kaggle GPU) |
| `config.yaml` | model routes, seeds, stats settings |
