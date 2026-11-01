# LinkedIn Assistant — Explainable, Open-Weight LLM Job Matching

CSE 573 (Fall 2026), Group 3, Project 3. A job-matching assistant whose agent pipeline stays fixed while a model
router sends each sub-task to an open-weight model (local Ollama or hosted ASU Voyager); a proprietary model
(Azure `gpt-5-mini`) is only a measured baseline. Extracted facts go into a Neo4j graph built on an 11-type
ontology, where every edge points back to an evidence span in the source text. Ranking is a cascade (filters →
BM25 / TF-IDF / Jaccard → MiniLM dense + RRF → LambdaMART with graph features → cross-encoder → MMR), and
explanations may only state facts the graph can cite.

The full write-up is `report/REPORT_FINAL.md` (built from `report/REPORT.md` and `eval/results/`). Numbers like
D30 in code comments and the report refer to entries in the project's decision log.

## Results at a glance

From `eval/results/SUMMARY.md` (frozen test splits). Labels marked _Claude-judged_ were produced by Claude
under the annotation guide (D30, D39) with no independent human check; see "Labels" below.

| Question                                                                             | Result                                                                                                                                                                        |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RQ1: open-weight vs. proprietary extraction (micro-F1, 120 public test docs, 3 runs) | gemma4-31b-it **69.7**, llama4-maverick-17b 65.9, gpt-oss-120b 64.0 vs. gpt-5-mini 61.1 (all non-inferior at δ = 3); CRF baseline 27.0. Local Ollama models on 60 docs: 46–55 |
| RQ2: LoRA student vs. teacher                                                        | Not answered: laptop MLX run stopped early, student 14.6 F1 on 10 docs (D38); needs a GPU run                                                                                 |
| RQ3: ontology vs. free-form SPO                                                      | Ontology graph 4.4× smaller, same duplicate rate, no significant ranking difference                                                                                           |
| RQ4: ranking (nDCG@10, 25 public profiles, Claude-judged)                            | Cascade 0.811 vs. BM25 0.714 (+13.6%, p = 0.021); graph features not significant                                                                                              |
| RQ4: explanation faithfulness (852 sentences, Claude-judged)                         | Template 1.00, gpt-5-mini 0.98, gpt-oss-120b 0.92, local llama3.1-8b 0.89 (target 0.95)                                                                                       |
| Graph quality (Claude-judged)                                                        | Edge precision 84%, skill normalization 77% (modern) vs. 74.6% (classical)                                                                                                    |
| User study                                                                           | Not run (blind A/B protocol in `eval/user_study/`)                                                                                                                            |

## Setup (macOS arm64; Linux is similar)

Prerequisites: Homebrew, conda, Docker Desktop, ~40 GB free disk. Developed on an Apple M4 with 16 GB RAM: run at
most one 7–9B local model at a time next to Neo4j.

```bash
# 1. Python environment (Python 3.11)
conda create -n linkedin-assistant python=3.11 -y
conda activate linkedin-assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Secrets
cp .env.example .env          # fill in the keys listed below

# 3. Neo4j 5.26 + APOC + GDS, in Docker (browser at http://localhost:7474)
docker compose --env-file .env up -d

# 4. Local LLMs (Ollama)
brew install ollama && brew services start ollama
ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull gemma2:9b

# 5. Datasets (needs a Kaggle login: `kaggle auth login`)
bash scripts/download_data.sh

# 6. Check the machine
python scripts/check_env.py
```

Keys in `.env` (never commit it):

| Variable                          | Needed for                                                | Where                                                                         |
| --------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `NEO4J_PASSWORD`                  | the graph (everything after Phase 2)                      | any 8+ character password                                                     |
| `ASU_API_KEY`                     | hosted open-weight models (RQ1 benchmark, job extraction) | ASU Voyager, `https://openai.rc.asu.edu/v1`                                   |
| `AZURE_OPENAI_API_KEY`            | proprietary baseline (`gpt-5-mini`)                       | an Azure OpenAI resource; set its endpoint in `config.yaml → providers.azure` |
| `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` | live jobs in the app (optional)                           | developer.adzuna.com                                                          |
| `KAGGLE_USERNAME`, `KAGGLE_KEY`   | dataset download (or `~/.kaggle/kaggle.json`)             | kaggle.com                                                                    |

Without the hosted keys, everything that runs on Ollama still works (local benchmark, CV parsing, explanations,
the app on the Kaggle index). Model routes per task live in `config.yaml → routes`; a model swap is a one-line edit.

## Replicating the results

Run everything from the repo root inside the `linkedin-assistant` env, with Neo4j and Ollama running. Steps are
resumable: re-running a step skips finished documents. Times given were measured on the M4 laptop.

**What is in git and what is not.** The frozen splits and test-set ids (`eval/frozen/`), the ontology, all code and
the result files (`eval/results/`) are committed. Data, extractions, annotation sheets and caches (`data/`,
`.cache/`) are git-ignored and are rebuilt by the steps below. Re-scoring without re-labeling needs the annotation
sheets in `data/processed/annotation/`, which are shared separately.

| #   | Step                                                                                                        | Command                                                                                                                                                                                                                                    | Time                                                  |
| --- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| 1   | Ontology (writes `ontology/generated/ONTOLOGY.docx`)                                                        | `python -m ontology.build`                                                                                                                                                                                                                 | seconds                                               |
| 2   | Data: load, clean/mask, dedupe, working set, splits, leakage check, test sets                               | `python -m ingest.build`                                                                                                                                                                                                                   | ~6 min                                                |
| 3   | Ranking indexes (BM25, TF-IDF, CRF skills, family classifier, K-means, MiniLM)                              | `python -m ranking.index`                                                                                                                                                                                                                  |                                                       |
| 4   | Judging pool (20 jobs × 40 profiles) and graph pool                                                         | `python -m ranking.pool --make`                                                                                                                                                                                                            |                                                       |
| 5   | CRF baseline                                                                                                | `python -m extraction.crf --train` · `python -m extraction.run --set test --extractor crf`                                                                                                                                                 |                                                       |
| 6   | Hosted benchmark (RQ1), graph-pool jobs, RQ2 teacher set                                                    | `bash scripts/run_long_jobs.sh hosted`                                                                                                                                                                                                     | ~15–190 s per extraction call, depending on the model |
| 7   | Local benchmark (60-doc subset, D36) and private profile CVs                                                | `bash scripts/run_long_jobs.sh ollama`                                                                                                                                                                                                     | ~4 min/doc/model                                      |
| 8   | Gold labels for extraction                                                                                  | see "Labels" below → `data/processed/annotation/gold_extraction.json`                                                                                                                                                                      |                                                       |
| 9   | Extraction scores (RQ1 table)                                                                               | `python -m eval.extraction_eval`                                                                                                                                                                                                           |                                                       |
| 10  | Graph build (+ evidence audit, RELATED_TO, Louvain); the id list is the public profiles plus the graph pool | `cat data/processed/profiles_public.txt data/processed/graph_pool.txt \| sort -u > data/processed/graph_build_ids.txt` · `python -m graph.build --wipe --no-llm --extractions asu__gpt-oss-120b --only data/processed/graph_build_ids.txt` |                                                       |
| 11  | Graph quality sheets and scores (R3)                                                                        | `python -m graph.annotate --make` → label → `python -m graph.annotate --score --labels llm`                                                                                                                                                |                                                       |
| 12  | Ranking evaluation, ablations, LambdaMART models for the app (RQ4)                                          | `python -m eval.ranking_eval --labels llm_label --save-models`                                                                                                                                                                             |                                                       |
| 13  | SPO ablation (RQ3)                                                                                          | `python -m extraction.spo --set graph --extractor asu/gpt-oss-120b` · `python -m eval.spo_eval --spo spo__asu__gpt-oss-120b --ontology asu__gpt-oss-120b`                                                                                  |                                                       |
| 14  | RQ2 student                                                                                                 | `python -m extraction.distill --make-set 1500` → teacher (step 6) → GPU: `notebooks/qlora_extraction.ipynb` on Colab/Kaggle (`pip install -r requirements-gpu.txt`); laptop: `scripts/qlora_mlx.sh export \| train \| serve` (D34)         | GPU ~2 h; laptop ~8 h/epoch                           |
| 15  | Explanation faithfulness (60 pairs × 4 explainers)                                                          | `python -m eval.explanation_eval --make` → label `supported` → `--score`                                                                                                                                                                   |                                                       |
| 16  | Table II and the report                                                                                     | `python -m eval.report` · `python report/build.py`                                                                                                                                                                                         |                                                       |

Steps 6–16 can also run unattended: `nohup python scripts/overnight.py >> .cache/logs/overnight.log 2>&1 &`
starts each stage when its inputs exist and leaves markers in `.cache/overnight/`. `python scripts/progress.py`
prints a progress percentage per workstream. Background jobs log to `.cache/logs/`.

Seeds are fixed (`config.yaml → seed: 573`). Hosted models run at temperature 0 except `gpt-5-mini`, which only
accepts its default, so its three runs also measure sampling variance. LLM calls are cached by content hash in
`.cache/llm/`, so re-running an evaluation reproduces the same outputs without new calls.

### Labels

The project ran with one person, so test labels were produced by Claude in separate `llm_*` columns under the
written guide (`eval/annotation/ANNOTATION_GUIDE.md`); results built on them are marked "Claude-judged" (D30, D39),
and no independent human check was done. To use human labels instead, fill the human columns (`label` in
`relevance_pool.csv`, `is_duplicate`, `gold_taxonomy_id`, `is_correct`, `supported`) and re-run the scoring steps
without `--labels llm…`; human labels take precedence wherever both exist. Helper scripts used for labeling:
`scripts/gold.py`, `scripts/mentions.py`, `scripts/apply_labels.py`, `scripts/apply_dupes.py`, `scripts/view_profile.py`.
A Label Studio config for extraction gold is in `eval/annotation/label_studio_extraction.xml`, and
`streamlit run ui/judge.py` is a relevance-judging tool.

## Running the app

```bash
KMP_DUPLICATE_LIB_OK=TRUE streamlit run ui/app.py      # http://localhost:8501
```

Needs Neo4j and Ollama running and steps 2–3, 10 and 12 done (the indexes, the graph and the LambdaMART models).

| Page                 | What it does                                                                                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dashboard            | Profile summary, top matches, index and graph statistics                                                                                                                                                           |
| My profile           | Upload a CV (PDF/DOCX/TXT), paste text, or pick a public Kaggle sample; parse with the CRF (seconds) or the local LLM (1–3 min); review and correct skills                                                         |
| Job matches          | Search the Kaggle index (20,000 LinkedIn postings, Dec 2023 – Apr 2024) or **live Adzuna jobs**; filters; any ranking method; per-card skill overlap, job description, score breakdown and a cited "Why this job?" |
| Skill gap            | Requirement coverage, missing skills with bridges, CV rewrites checked by the critic                                                                                                                               |
| Knowledge graph      | Interactive subgraph of the CV and a job, with the evidence behind each edge                                                                                                                                       |
| Assistant & outreach | Questions answered only from cited graph facts; outreach draft and follow-up reminder                                                                                                                              |
| Find candidates      | Recruiter view: rank the 4,981 indexed CVs for a posting                                                                                                                                                           |
| Models & router      | Routes per task, live model swap, privacy guard, call log                                                                                                                                                          |
| Results              | Table II and the per-question results                                                                                                                                                                              |

**Live jobs (Adzuna).** On Job matches, choose _Live jobs (Adzuna)_. Keywords are suggested from the parsed CV
(recent role titles without seniority words, the headline, the predicted job family) and the CV's US state; edit
them, pick a _Date posted_ window, and press _Find matches_. Only the keywords and location are sent to Adzuna.
Each search is cached for 6 hours in `data/live/adzuna/` and every API call is logged to `calls.jsonl`
(`python -m ingest.adzuna "software engineer" --where Arizona` runs one from the command line). Adzuna returns a
~500-character snippet per posting, so live jobs have less text for skill matching and explanations than the
Kaggle postings; each card links to the full posting. Live jobs are not part of any evaluation.

## Privacy

Uploaded CVs and the `resume_corpus` CVs are private: they are parsed, explained and checked only on local Ollama.
The router raises `PrivacyError` if a private document is sent to a hosted provider (`config.yaml →
public_sources` lists the sources hosted models may see: the Kaggle postings and resumes, and Adzuna job ads).
Router logs hold no prompt text. One accidental exposure is disclosed in D35.

## Tests

```bash
python -m pytest tests/          # offline: no network, Neo4j or LLM needed
```

## Troubleshooting

- **Streamlit crashes with `OMP: Error #179` or a segfault on "Find matches" (macOS).** LightGBM and PyTorch load
  two OpenMP runtimes; `ranking/live.py` loads the LambdaMART models first, and `ui/app.py` imports it before
  anything else. Start Streamlit with `KMP_DUPLICATE_LIB_OK=TRUE`.
- **Local generation never finishes.** Long Ollama outputs are capped with `max_tokens`; if a call still hangs,
  `brew services restart ollama`.
- **Laptop swaps during QLoRA.** Quit Docker Desktop (Neo4j) while training, then restart it with
  `docker compose --env-file .env up -d`.
- **`python3` is 3.9 on macOS.** Use the conda env's Python.
- **A scratch script breaks imports.** Don't name scripts after standard-library modules (`inspect.py`, `bisect.py`).

## Layout

| Folder        | Contents                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ontology/`   | Ontology spec, Pydantic extraction schema, generator for `ONTOLOGY.docx`                                                 |
| `ingest/`     | Load, clean/mask, MinHash dedupe, splits, test sets, weak labels; `adzuna.py` live jobs                                  |
| `extraction/` | Chunking, prompts, LLM extraction runner, CRF baseline, SPO ablation, distillation data, QLoRA config                    |
| `graph/`      | O\*NET normalization, Neo4j build and evidence audit, RELATED_TO + Louvain, R3 sheets                                    |
| `ranking/`    | Indexes, lexical, dense, RRF, graph features, LambdaMART, cross-encoder + MMR, live cascade, Adzuna corpus, judging pool |
| `router/`     | Model router: privacy guard, cache, retries and fallback, call logs                                                      |
| `agents/`     | LangGraph harness (7 agents), explainer + critic, profile optimizer, outreach and scheduler                              |
| `ui/`         | Streamlit app (`app.py`, `views/`, `services.py`, `theme.css`; design adapted from Krish Naik's frontend) and `judge.py` |
| `eval/`       | Frozen splits, stats plan and tests, extraction / ranking / SPO / explanation evaluation, user-study kit, `report.py`    |
| `report/`     | `REPORT.md` template, `build.py`, `conclusion.md`, generated `REPORT_FINAL.md`                                           |
| `scripts/`    | Data download, environment check, long-job queues, overnight orchestrator, progress, QLoRA (MLX), labeling helpers       |
| `notebooks/`  | `qlora_extraction.ipynb` (Colab/Kaggle GPU)                                                                              |
| `config.yaml` | Providers, model routes, benchmark models, prices, seeds, stats settings                                                 |
