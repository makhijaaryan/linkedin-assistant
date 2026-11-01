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
in separate columns under the written annotation guide, and results are reported as "Claude-judged" (no human
spot-check was done; see Section 6). CVs from `resume_corpus` are treated as private (D24): they
are never sent to a hosted model and are not labeled by Claude (one accidental exposure of dedupe-pair fragments is
disclosed in D35), so the ranking evaluation uses the 25 public profiles (500 judged pairs) and
the extraction gold covers the 120 public test documents (75 postings, 45 Kaggle CVs). Long postings are annotated on
their first 6,000 characters, and predictions found only past that point are not scored.

**Live postings (not evaluated).** The app can also search current postings through the Adzuna API (D40). Only the
search keywords and location leave the laptop; postings are cached and logged. Adzuna returns a ~500-character
snippet per posting rather than the full text, so live jobs are a demonstration of the pipeline on fresh data and
are not part of any reported result.

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

**User interface.** A ten-page Streamlit app with top navigation, a landing page and a light/dark theme (card
design adapted from Krish Naik's team frontend): profile upload and correction, job matches with removable filter chips and cited "Why this job?" panels, a skill-gap
plan with critic-checked CV rewrites, an interactive knowledge-graph view with evidence inspector, grounded Q&A and
outreach, a recruiter view, the router with live model swap and privacy guard, and a results dashboard. Job matches
can search either the Kaggle index or live Adzuna postings; for live search, keywords are suggested from the parsed
CV (recent role titles without seniority words, the headline, the predicted job family) and the user can edit them
and restrict the posting date before anything is sent.

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
| R2 / RQ1 | Micro-F1, best open (asu__gemma4-31b-it) | ≥ 0.80 | 0.697 |
| R2 | ΔF1 vs CRF (points) | ≥ +10 (CI low) | +42.6 [+39.5, +45.7] |
| R3 | Top-1 normalization (modern) (Claude-judged, D30) | ≥ 0.90 | 0.770 [0.731, 0.805] |
| R3 | Top-1 normalization (classical) (Claude-judged, D30) | ≥ 0.90 | 0.746 [0.706, 0.782] |
| R3 | Edge precision (100) (Claude-judged, D30) | ≥ 0.90 | 0.840 [0.756, 0.899] |
| R3 / RQ3 | Duplicate-node rate SPO vs ontology | report | 0.090 vs 0.091 (z-test p = 0.6913) |
| R4 / RQ4 | nDCG@10 cascade vs BM25 (25 profiles) (Claude-judged, D30) | +15% | 0.811 vs 0.714 (+13.6%, p = 0.0210) |
| R4 ablation | ablate_no_graph | report | Δ -0.010 [-0.033, +0.010], p_holm = 1.0000 |
| R4 ablation | ablate_no_rrf | report | Δ -0.007 [-0.027, +0.011], p_holm = 1.0000 |
| R4 ablation | ablate_no_cross_encoder | report | Δ -0.002 [-0.015, +0.011], p_holm = 1.0000 |
| R5 / RQ2 | LoRA student F1 vs teacher (mlx__qwen2.5-7b-qlora) | within 3 pts | 14.6 vs 64.0 (-49.4) |
| R6 / RQ4 | Faithfulness (llm:asu/gpt-oss-120b) (Claude-judged, D30) | ≥ 0.95 | 0.915 [0.866, 0.947] |
| R6 / RQ4 | Faithfulness (llm:azure/gpt-5-mini) (Claude-judged, D30) | ≥ 0.95 | 0.983 [0.956, 0.993] |
| R6 / RQ4 | Faithfulness (llm:ollama/llama3.1:8b) (Claude-judged, D30) | ≥ 0.95 | 0.889 [0.836, 0.926] |
| R6 / RQ4 | Faithfulness (template) (Claude-judged, D30) | ≥ 0.95 | 1.000 [0.985, 1.000] |
| R6 | Blind A/B Likert (15-20 users) | mean ≥ 4.0 | PENDING: run the study (eval/user_study/PROTOCOL.md) |

### 5.2 RQ1: open-weight vs. proprietary extraction, cost and latency

| Extractor | Docs | Runs | Micro-F1 | Micro-F1, shared 60 docs | Skill | Requirement | Experience | Credential | Project | Company | Occupation | Location | p50 s/doc | $/1K docs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| asu__gemma4-31b-it | 120 | 3 | 69.7 ± 0.1 | 68.0 (n=60) | 66.1 | 72.8 | 91.4 | 86.7 | 62.7 | 44.8 | 87.8 | 47.6 | 74.0 | 3.119 |
| asu__gpt-oss-120b | 120 | 3 | 64.0 ± 0.2 | 62.3 (n=60) | 58.4 | 74.1 | 86.9 | 81.0 | 40.7 | 43.1 | 85.3 | 35.5 | 96.8 | 1.577 |
| asu__llama4-maverick-17b | 120 | 3 | 65.9 ± 0.1 | 65.1 (n=60) | 62.1 | 68.5 | 89.7 | 82.1 | 44.2 | 45.3 | 86.8 | 36.8 | 36.6 | 5.598 |
| azure__gpt-5-mini | 120 | 3 | 61.1 ± 0.7 | 60.5 (n=60) | 55.3 | 68.4 | 90.7 | 82.4 | 44.0 | 41.1 | 85.1 | 38.0 | 23.9 | 8.738 |
| crf | 120 | 1 | 27.0 ± 0.0 | 27.5 (n=60) | 8.0 | 33.2 | 76.2 | 53.9 | 0.0 | 12.2 | 48.8 | 32.9 | 0.0 | 0.000 |
| mlx__qwen2.5-7b-qlora | 10 | 1 | 14.6 ± 0.0 | 14.6 (n=10) | 8.2 | 0.0 | 27.9 | 28.6 | 100.0 | 6.9 | 31.8 | 0.0 | 339.0 | n/a |
| ollama__gemma2-9b | 60 | 1 | 55.2 ± 0.0 | 55.2 (n=60) | 48.3 | 64.0 | 70.4 | 72.1 | 75.0 | 48.6 | 75.9 | 32.7 | 293.3 | n/a |
| ollama__llama3.1-8b | 60 | 1 | 53.4 ± 0.0 | 53.4 (n=60) | 48.0 | 58.7 | 63.4 | 64.4 | 57.1 | 47.8 | 78.2 | 31.4 | 222.4 | n/a |
| ollama__qwen2.5-7b | 60 | 1 | 46.1 ± 0.0 | 46.1 (n=60) | 39.0 | 53.9 | 57.3 | 65.8 | 16.7 | 49.5 | 76.7 | 33.9 | 226.2 | n/a |

| Open model | ΔF1 vs proprietary [95% CI] | p (Holm) | Non-inferior (δ=3) | ΔF1 vs CRF [95% CI] | +10 met |
|---|---|---|---|---|---|
| asu__gemma4-31b-it | +6.9 [+5.2, +8.5] | 0.0007 | ✓ | +42.6 [+39.5, +45.7] | ✓ |
| asu__gpt-oss-120b | +2.0 [+0.6, +3.3] | 0.0070 | ✓ | +37.7 [+34.8, +40.6] | ✓ |
| asu__llama4-maverick-17b | +3.4 [+2.1, +4.8] | 0.0007 | ✓ | +39.1 [+36.2, +42.1] | ✓ |
| mlx__qwen2.5-7b-qlora | -51.8 [-64.5, -35.4] | 0.0057 | ✗ | -16.2 [-29.2, -0.2] | ✗ |
| ollama__gemma2-9b | -4.2 [-7.4, -1.1] | 0.0117 | ✗ | +30.5 [+25.5, +35.3] | ✓ |
| ollama__llama3.1-8b | -7.2 [-10.8, -3.9] | 0.0008 | ✗ | +27.5 [+22.2, +32.4] | ✓ |
| ollama__qwen2.5-7b | -12.5 [-15.3, -9.9] | 0.0007 | ✗ | +22.1 [+17.7, +26.4] | ✓ |


**Quality vs. cost and latency per document** (list or gateway-indicative prices; local models cost 0, latency is end-to-end per document including all chunks):

| extractor                |   micro-F1 |   p50 s/doc |   p95 s/doc |   $ / 1K docs |
|:-------------------------|-----------:|------------:|------------:|--------------:|
| asu__gemma4-31b-it       |       69.7 |        74   |       229.6 |       3.11889 |
| asu__llama4-maverick-17b |       65.9 |        36.6 |       139.7 |       5.59759 |
| asu__gpt-oss-120b        |       64   |        96.8 |       342.6 |       1.57705 |
| azure__gpt-5-mini        |       61.1 |        23.9 |        71.5 |       8.73762 |
| ollama__gemma2-9b        |       55.2 |       293.3 |      1164.4 |     nan       |
| ollama__llama3.1-8b      |       53.4 |       222.4 |      1207.1 |     nan       |
| ollama__qwen2.5-7b       |       46.1 |       226.2 |      1011.2 |     nan       |
| crf                      |       27   |         0   |         0   |       0       |
| mlx__qwen2.5-7b-qlora    |       14.6 |       339   |       853.1 |     nan       |

### 5.3 RQ4: ranking and ablations


| System | nDCG@10 | MRR@10 | P@5 |
|---|---|---|---|
| dense | 0.838 | 0.973 | 0.912 |
| ablate_no_graph | 0.833 | 0.980 | 0.968 |
| ablate_no_rrf | 0.830 | 0.950 | 0.952 |
| ablate_no_cross_encoder | 0.825 | 0.953 | 0.952 |
| ltr_full | 0.823 | 0.953 | 0.952 |
| cascade_mmr | 0.811 | 0.953 | 0.960 |
| cross_encoder | 0.806 | 1.000 | 0.912 |
| tfidf | 0.784 | 0.947 | 0.912 |
| rrf | 0.783 | 0.950 | 0.928 |
| bm25 | 0.714 | 0.930 | 0.840 |
| classical | 0.532 | 0.925 | 0.656 |
| jaccard | 0.417 | 0.607 | 0.464 |

Cascade vs BM25: Δ nDCG@10 = +0.097 [+0.022, +0.180], relative +13.6%, p = 0.0210, d_z = 0.47; +15% target not met.

| Ablation (vs full LambdaMART) | Δ nDCG@10 [95% CI] | p (Holm) |
|---|---|---|
| ablate_no_graph | -0.010 [-0.033, +0.010] | 1.0000 |
| ablate_no_rrf | -0.007 [-0.027, +0.011] | 1.0000 |
| ablate_no_cross_encoder | -0.002 [-0.015, +0.011] | 1.0000 |


### 5.4 RQ3: ontology vs. free-form SPO

On 485 documents, the free-form SPO graph has a duplicate-node rate of 0.090 vs. 0.091 for the ontology graph (two-proportion z-test p = 0.6913). With each graph's features in LambdaMART, nDCG@10 is 0.819 (ontology) vs. 0.840 (SPO); Δ = -0.021 [-0.048, +0.003], Wilcoxon p = 0.1266.

### 5.5 R3: normalization and edge precision

Top-1 skill normalization on 500 mentions: modern 0.770 [0.731, 0.805], classical 0.746 [0.706, 0.782] (McNemar p = 0.1272). Edge precision on sampled extracted edges: 0.840 [0.756, 0.899]. Labels: Claude-judged (D30).

### 5.6 RQ2: LoRA student

Micro-F1 — mlx__qwen2.5-7b-qlora: 14.6; asu__gpt-oss-120b: 64.0; ollama__qwen2.5-7b: 46.1; crf: 27.0.

### 5.7 R6: explanations

Sentences shown after the critic, judged against the graph facts they cite (Claude-judged, D30/D39). Fully faithful explanations vs. the proprietary explainer (McNemar, Holm): llm:asu/gpt-oss-120b p_holm = 0.004; llm:ollama/llama3.1:8b p_holm = 0.001.

|                        |   faithfulness | 95% CI         |   hallucination rate |   fully faithful explanations |
|:-----------------------|---------------:|:---------------|---------------------:|------------------------------:|
| llm:asu/gpt-oss-120b   |          0.915 | [0.866, 0.947] |                0.085 |                         0.719 |
| llm:azure/gpt-5-mini   |          0.983 | [0.956, 0.993] |                0.017 |                         0.931 |
| llm:ollama/llama3.1:8b |          0.889 | [0.836, 0.926] |                0.111 |                         0.648 |
| template               |          1.000 | [0.985, 1.000] |                0.000 |                         1.000 |

## 6. Label quality and limitations

**No independent human check.** All test labels in this report (relevance, gold extraction, dedupe, normalization, edges and explanation faithfulness) were produced by Claude (D30); no human spot-check of them was done.

- **Claude-judged labels.** Test labels for relevance, gold extraction, dedupe, normalization, edges and explanation
  faithfulness were produced by Claude under a written guide, with no independent human check. They
  share blind spots with LLM extractors, so absolute scores may be optimistic for LLM systems; comparisons between
  systems are less affected than absolute numbers.
- **Private CVs.** `resume_corpus` profiles (15 of 40) and test documents (30 of 150) have no Claude labels; the
  ranking and extraction results above therefore cover public Kaggle data only.
- **Proprietary baseline.** `gpt-5-mini` replaced the planned Gemini/OpenRouter baseline (free tiers were too small)
  and cannot run at temperature 0, so its three runs also measure sampling variance.
- **Local hardware.** Local inference on a 16 GB laptop (~13–17 tokens/s) limited how many local models finished the
  full benchmark; the QLoRA student was trained with MLX on a 4-bit base instead of NF4 on a CUDA GPU (D34).

## 7. Conclusion

All numbers below come from the frozen test splits and are Claude-judged where marked (D30), with no independent human check. The blind A/B user study has not been run.

**RQ1, open-weight vs. proprietary.** The three hosted open-weight models were non-inferior to the proprietary baseline (Azure gpt-5-mini, micro-F1 61.1) on 120 test documents: gemma4-31b-it 69.7 (+6.9 [+5.2, +8.5], p = 0.0007), llama4-maverick-17b 65.9 (+3.4) and gpt-oss-120b 64.0 (+2.0), all within the pre-registered δ = 3 margin on the non-inferiority side and all about +38 to +43 F1 over the CRF baseline. At list price they were also cheaper per 1K documents (gpt-oss-120b $1.58, gemma4 $3.12, llama4 $5.60 vs. $8.74 for gpt-5-mini). They were slower: median seconds per document were 37 (llama4), 74 (gemma4) and 97 (gpt-oss-120b) vs. 24 for gpt-5-mini. The best open model (69.7) did not reach the proposal's 0.80 F1 target, so that target is not met.

**Local models.** On a fixed 60-document public subset, the local Ollama models were below the proprietary baseline and failed non-inferiority: gemma2-9b 55.2 (-4.2), llama3.1-8b 53.4 (-7.2), qwen2.5-7b 46.1 (-12.5). All three still beat the CRF baseline by 22 to 31 F1 points. They took 3.7 to 4.9 minutes per document on this laptop, so they are a privacy option for CVs, not a throughput option.

**RQ2, LoRA student.** Not answered. The laptop trained at about 30 s per example, validation loss started rising after example 100, and the example-100 checkpoint produced runaway outputs (evidence quotes that copy most of the document and hit the token cap). It was run on 10 documents only and scored F1 14.6, below the CRF baseline. The comparison in Table II ("14.6 vs 64.0") sets the student on 10 documents against the teacher on 120, so it is not like-for-like. A proper RQ2 run needs the full teacher set trained for a full epoch on a GPU (the Colab/Kaggle notebook in `extraction/qlora/`).

**RQ3, ontology vs. free-form SPO.** The ontology-guided graph is 4.4 times smaller (5,222 vs. 22,968 nodes) with the same duplicate-node rate (9.1% vs. 9.0%, p = 0.69). Ranking quality did not differ significantly (nDCG@10 0.819 ontology vs. 0.840 SPO, p = 0.13, 25 profiles), and the point estimate favors SPO. The ontology's benefit here is a compact, cited graph, not better ranking.

**RQ4, ranking and graph features.** The cascade beat BM25 by +13.6% nDCG@10 (0.811 vs. 0.714, p = 0.021) on 25 public profiles, below the +15% target. Dense retrieval alone (0.838) scored higher than the full cascade (0.811), and removing the graph features changed nothing significant (-0.010, p_holm = 1.0), so in this evaluation graph features did not earn their place under the "beat the classical baseline" rule. Skill normalization reached 77.0% with the modern method vs. 74.6% classical (McNemar p = 0.13), and edge precision was 84.0% [75.6, 89.9]; both are below the 90% targets. Graph audit checks passed: no extracted edge lacks Evidence.

**RQ4, grounded explanations.** On 60 public (profile, job) pairs, 852 explanation sentences shown after the critic were judged against the graph facts they cite (Claude-judged). The template baseline was fully faithful (244/244), and gpt-5-mini reached 98.3% [95.6, 99.3]. The open-weight explainers did not reach the 0.95 target: gpt-oss-120b 91.5% [86.6, 94.7] and the local llama3.1-8b 88.9% [83.6, 92.6]. Most unsupported sentences attribute the total years of experience to a specific field, or treat a skill that is only related as matched. At the explanation level, 93% of gpt-5-mini explanations were fully supported vs. 72% for gpt-oss-120b and 65% for llama3.1-8b (McNemar p_holm = 0.004 and 0.001). Every shown sentence carried a citation, so the critic catches missing citations but not overclaims.

**Limitations.** Relevance, gold extraction, dedupe, skill-mention, edge and explanation-faithfulness labels were produced by Claude (D30), with no independent human check; ranking uses 25 public profiles only because the 15 `resume_corpus` profiles stay private; the local tier uses 60 documents, the student 10; the blind A/B study is pending.

## References

See the proposal's reference list [1]–[19]; additionally MLX-LM (Apple, 2024) for LoRA on Apple silicon.
