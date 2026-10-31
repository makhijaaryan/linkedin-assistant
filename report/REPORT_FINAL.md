# LinkedIn Assistant: An Explainable, Open-Weight LLM Framework for Job Matching

CSE 573 (Fall 2026), Group 3, Project 3. Generated from `eval/results/` by `report/build.py`.

## Abstract

Career assistants usually wire résumé parsing, job matching and knowledge-graph reasoning to one proprietary LLM.
We keep the agent pipeline fixed and let a model router send each sub-task to an open-weight model, served locally
(Ollama) or through a hosted gateway (ASU Voyager), with a proprietary model (Azure OpenAI `gpt-5-mini`) kept only as a
measured baseline. A minimal eleven-type ontology guides extraction, so every skill, requirement and company fact
becomes a typed Neo4j node linked to the text span it came from. Matching is a cascade from BM25, TF-IDF and Jaccard
through dense retrieval with Reciprocal Rank Fusion to LambdaMART with graph features, a cross-encoder and MMR, and
explanations may only state what the graph can cite. We report extraction F1, normalization accuracy, edge precision,
nDCG@10, explanation faithfulness and, as the headline result, the quality–cost–latency trade-off across model tiers.

## 1. Problem and research questions

Keyword filters miss paraphrased skills; embedding recommenders recover them but cannot say why a job fits. Prior
designs also lock into one proprietary model, send every CV to a token-priced API and never measure what an open model
would lose. We frame the task as evidence-grounded, two-sided matching under a model-agnostic harness, and ask:

- **RQ1** How much quality do open-weight models lose against a proprietary model, and what do they save in cost and
  latency?
- **RQ2** Can a LoRA-tuned small model match large zero-shot models on extraction?
- **RQ3** Does ontology-constrained extraction beat free-form subject–predicate–object (SPO) extraction on graph
  consistency and ranking?
- **RQ4** Do graph features improve ranking, and can explanations be fully grounded?

## 2. Data

Kaggle LinkedIn Job Postings 2023–24 (123,842 postings with text), the Kaggle Resume Dataset (2,484 CVs, 24 categories) and the MIT-licensed `resume_corpus` (29,783 CVs) are cleaned, contact-masked (58,063 postings and 23,167 CVs had contacts masked) and deduplicated with MinHash LSH on title + company + text Jaccard ≥ 0.9: 20,440 duplicate postings and 2,295 duplicate CVs removed. Splits are by duplicate cluster (train/dev/test 80/10/10), with 0 clusters crossing splits; test documents with a near neighbour in train (Jaccard ≥ 0.8) are excluded from the gold sets. The working set is 20,000 postings and 4,981 CVs; skills are normalized to O\*NET 31.0 (Lightcast access required a sales contact, D14).

**Labels.** The project ran with one team member, so test labels follow D30: Claude labels every public (Kaggle) item
in separate columns, and the human owner spot-checks a fixed random sample of each set; results are reported as
"Claude-judged" with the agreement figures in Section 6. CVs from `resume_corpus` are treated as private (D24) and are
never shown to a hosted model or to Claude, so the ranking evaluation uses the 25 public profiles (500 judged pairs) and
the extraction gold covers the 120 public test documents (75 postings, 45 Kaggle CVs). Long postings are annotated on
their first 6,000 characters, and predictions found only past that point are not scored.

## 3. System

**Ontology.** Eleven node types (Candidate, JobPosting, Company, Occupation, Skill, Requirement, Experience,
Credential, Project, Location, Evidence) and relations HAS_SKILL, HELD, USED, AT, POSTED_BY, FOR_ROLE, HAS_REQUIREMENT,
TARGETS, RELATED_TO, TYPICAL_FOR and SUPPORTED_BY, frozen as v1.0 and exported as a Pydantic schema and the course's
Word template. Every extracted edge must be SUPPORTED_BY an Evidence node holding the exact source span.

**Model router.** One OpenAI-compatible client for every provider; `config.yaml` maps each task to a provider/model,
so a model swap is one line. The router enforces the privacy rule (a `private=True` call may only reach a provider
marked `private_ok`, i.e. local Ollama or the local MLX server), caches by content hash, retries with back-off, never
silently falls back when a benchmark pins a model, and logs tokens, latency and list-price cost for every call.

**Extraction.** Documents are cut into 200–500-word chunks with 10% overlap; each chunk is extracted to ontology JSON,
validated against the schema with one repair retry, merged, and every item's evidence quote is located back in the
source to get character offsets. The classical baseline is a two-layer BIO CRF trained on silver labels from O\*NET
gazetteers. The SPO ablation asks the same model for free-form triples on the same documents.

**Graph.** Mentions are normalized to O\*NET skills and occupations: rules plus character-n-gram TF-IDF (classical)
versus embedding nearest neighbours with an LLM tie-break (modern). The loader writes idempotent MERGEs, infers
RELATED_TO edges between skills (marked `inferred`), and runs Louvain communities (GDS). An audit checks that no
extracted edge lacks evidence.

**Matching cascade.** Filters (state, work mode, seniority) → BM25, TF-IDF and skill Jaccard → a MiniLM bi-encoder
fused with the lexical lists by RRF (k = 60) → LambdaMART over lexical, dense, coverage, experience-gap and graph
features (skill paths, RELATED_TO bridges to unmet requirements) → a cross-encoder on the top 20 → MMR (λ = 0.7) for a
diverse top 10. The same cascade ranks candidates for a job.

**Explanations and agents.** Seven LangGraph agents (Parser, KG-Build, Matching, Critic/Explainer, Profile Optimizer,
Research & Outreach, Scheduler) call the router. For a (CV, job) pair, Cypher returns met requirements with the CV
quote, required gaps, RELATED_TO bridges, the years check and shared occupations; each fact carries edge IDs. The
template explainer verbalizes these facts directly; the LLM explainer must cite a fact for every sentence, and a critic
model rejects any sentence its cited edges do not support. Outreach is drafted only, and a follow-up reminder (.ics) is
created only after explicit approval.

**User interface.** A ten-page Streamlit app (layout and visual design adapted from Krish Naik's team frontend):
profile upload and correction, job matches with removable filter chips and cited "Why this job?" panels, a skill-gap
plan with critic-checked CV rewrites, an interactive knowledge-graph view with evidence inspector, grounded Q&A and
outreach, a recruiter view, the router with live model swap and privacy guard, a results dashboard and the annotation
spot-check tool.

## 4. Experimental setup

**Models (RQ1).** Local (Ollama, laptop: Apple M4, 16 GB): `llama3.1:8b`, `qwen2.5:7b`, `gemma2:9b`. Hosted
open-weight (ASU Voyager): `gpt-oss-120b`, `llama4-maverick-17b`, `gemma4-31b-it`. Proprietary (Azure OpenAI):
`gpt-5-mini`. Temperature 0 except `gpt-5-mini`, which only accepts the default (1); reasoning models use low effort.
Hosted models run 3 times on the 120 public test documents; local models run once on all 150 (D25). Costs use list or
gateway-indicative prices per 1M tokens (Voyager itself is free).

**Statistics** (pre-registered in `eval/STATS_PLAN.md`). Paired bootstrap CIs (10,000 resamples), randomization tests,
McNemar on entity-level hits, Holm correction within each family, Wilson intervals for proportions, Wilcoxon for
latency, and non-inferiority margins of 3 F1 points and 0.02 nDCG@10.

## 5. Results

### 5.1 Targets (Table II)

| Thread | Metric | Target | Result |
|---|---|---|---|
| R1 | Dedupe precision (200 job pairs) (Claude-judged, D30) | ≥ 0.95 | 1.000 [0.981, 1.000] |
| R1 | Split leakage (clusters across splits) | 0 | 0 |
| R2 / RQ1 | Per-type F1 on 150 docs | ≥ 0.80; +10 over CRF | PENDING: gold labels (Label Studio export -> data/processed/annotation/gold_extraction.json) |
| R3 | Top-1 normalization (500) | ≥ 0.90 | PENDING: fill gold_taxonomy_id in skill_mentions_500.csv |
| R3 | Edge precision (100) | ≥ 0.90 | PENDING: fill is_correct in edge_sample_100.csv |
| R3 / RQ3 | Ontology vs SPO | report | PENDING: run extraction.spo, then eval.spo_eval |
| R4 / RQ4 | nDCG@10, MRR, P@5 on 40 profiles | +15% nDCG over BM25 | PENDING: 800 relevance labels (streamlit run ui/judge.py) |
| R5 / RQ2 | LoRA F1 vs teacher | within 3 pts | PENDING: teacher labels, then scripts/qlora_mlx.sh (D34) and the student's benchmark run |
| R6 / RQ4 | Explanation faithfulness (60) | ≥ 0.95 | PENDING: fill supported in explanations_60.csv |
| R6 | Blind A/B Likert (15-20 users) | mean ≥ 4.0 | PENDING: run the study (eval/user_study/PROTOCOL.md) |

### 5.2 RQ1: open-weight vs. proprietary extraction, cost and latency

*Pending: extraction benchmark.*



### 5.3 RQ4: ranking and ablations

*Pending: ranking evaluation.*

### 5.4 RQ3: ontology vs. free-form SPO

*Pending: SPO extraction and `eval.spo_eval`.*

### 5.5 R3: normalization and edge precision

*Pending: mention and edge labels.*

### 5.6 RQ2: LoRA student

*Pending: teacher labels → MLX LoRA training → the student's benchmark run (D34).*

### 5.7 R6: explanations

*Pending: human faithfulness labels on the 60 explanations.*

## 6. Label quality and limitations

*Pending: the owner's spot-checks on the Annotation page.*

- **Claude-judged labels.** Test labels for relevance, gold extraction, dedupe, normalization and edges were produced
  by Claude under a written guide and checked by the project owner on fixed random samples (agreement above). They
  share blind spots with LLM extractors, so absolute scores may be optimistic for LLM systems; comparisons between
  systems are less affected than absolute numbers.
- **Private CVs.** `resume_corpus` profiles (15 of 40) and test documents (30 of 150) have no Claude labels; the
  ranking and extraction results above therefore cover public Kaggle data only.
- **Proprietary baseline.** `gpt-5-mini` replaced the planned Gemini/OpenRouter baseline (free tiers were too small)
  and cannot run at temperature 0, so its three runs also measure sampling variance.
- **Local hardware.** Local inference on a 16 GB laptop (~13–17 tokens/s) limited how many local models finished the
  full benchmark; the QLoRA student was trained with MLX on a 4-bit base instead of NF4 on a CUDA GPU (D34).

## 7. Conclusion

*Pending: conclusion, written once all results are in.*

## References

See the proposal's reference list [1]–[19]; additionally MLX-LM (Apple, 2024) for LoRA on Apple silicon.
