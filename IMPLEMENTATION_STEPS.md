# LinkedIn Assistant — Step-by-Step Implementation Guide

Source: *FA26 Group 3 Proposal — "LinkedIn Assistant: An Explainable, Open-Weight LLM Framework for Job Matching"*

Owners: **AM** = Aryan, **KN** = Krish, **AP** = Ashmit, **HR** = Himanshi, **AU** = Aditya

> **Main rule:** a modern component stays in the system only if it beats its classical baseline on the frozen test splits.

---

## Phase 0 — Foundations (do first; all members)

1. **Repo:** set up one repo with these folders: `data/`, `ontology/`, `extraction/`, `graph/`, `ranking/`, `agents/`, `router/`, `ui/`, `eval/`, `notebooks/`. Add `pyproject.toml`/`requirements.txt`, `.env.example` (Groq/OpenRouter keys, Neo4j URI), pre-commit hooks, and a `.gitignore` that covers data and secrets.
2. **Shared config:** one `config.yaml` holds the model routes, paths, and random seeds. Fix the seeds so results can be reproduced.
3. **Infra:** Neo4j (Docker or Aura Free), Ollama with Llama/Qwen/Gemma pulled, Groq and OpenRouter API keys, and a Kaggle/Colab GPU for QLoRA.
4. **Build the model router early (AU), before Week 2 if possible.** Use one OpenAI-compatible client that routes each call by task, privacy, and cost. Log tokens, latency, $, model, and prompt hash to JSONL/SQLite, and add a cache keyed by content hash. R7 depends on this logging from the start.
5. **Privacy guardrail:** real CVs go to Ollama only, and hosted APIs see only anonymized Kaggle text. Enforce this in the router code, not just by convention.
6. **Process:** hold a weekly sync. Each thread keeps its results table in `eval/results/` so the final report is easy to put together.
7. **Shared stats helpers:** write `eval/stats.py` with unit tests. It provides a paired bootstrap, a paired randomization test, McNemar's test, Holm–Bonferroni correction, Wilson CIs, and the non-inferiority check. Every thread uses these helpers (see the [Statistical Significance Testing Plan](#statistical-significance-testing-plan-addresses-proposal-feedback)).

## Phase 1 — R1 Data & Ontology (M1, Oct 14) — KN, AM

1. **Download the data:**
   * Job postings: Kaggle LinkedIn Postings 2023-24, LinkedIn Job Posts Insights, and the HF `datastax` listings.
   * Resumes: Kaggle Resume Dataset, GitHub `resume_corpus`, and the 5 team CVs (with consent).
   * Taxonomies: Lightcast Open Skills, OSMT, and O*NET.
2. **Clean the text** and mask contact details (emails, phones, URLs, and names in CV headers).
3. **Deduplicate** on (title, company) plus MinHash LSH over the text. Check 200 sampled pairs by hand. **Target: precision ≥ 0.95.**
4. **Pick a 20K stratified working set** (by industry, seniority, and location) for full LLM extraction. All postings still get BM25 and embeddings.
5. **Freeze the train/dev/test splits** and write a leakage check: no shared doc IDs or near-duplicates across splits. **Target: no overlap.**
6. **Freeze the 11-type ontology** and its relations (Fig. 2) as JSON Schema. Every field carries `source_span`, `confidence`, and `extractor_version`. Generate Pydantic models from the schema.
7. **Annotation:** write an annotation guide and set up a tool (Label Studio or Doccano).
8. **Start the test sets now:**
   * 150 extraction docs, each annotated by two people (report κ).
   * 40 CV profiles for relevance judging.
   * 500 skill mentions for normalization.
9. **Collect weak labels:** job-skill tables, resume categories, and O*NET occupation-skill lists.
10. **Freeze the stats plan:** write `eval/STATS_PLAN.md` at the same time as the splits. It records α, the non-inferiority margins δ, the test for each comparison, and the families used for Holm correction. Fixing it now means the analysis is chosen before any results come in.

## Phase 2 — R2 Parsing Benchmark (Oct 21) — AP (+AU for router)

1. **Classical baseline:** split the text into sections with regex (Experience, Education, Skills, …), then run a spaCy CRF NER model trained on weak/silver labels.
2. **Modern path:**
   1. Chunk the text into 200–500 words with 10% overlap.
   2. Run an ontology-constrained JSON prompt on each chunk.
   3. Validate the output against the schema, with one repair retry.
   4. Merge the chunk results.
3. **Compare models:** 3–4 open models across Ollama, Groq, and OpenRouter, plus the proprietary baseline (anonymized data only). Send every call through the router.
4. **Evaluate** on the 150-doc test set with per-type P/R/F1. **Target: F1 ≥ 0.80 and ≥ +10 points over the CRF.**
   * **Open vs. proprietary (RQ1):** run each model 3 times at temperature 0. Compare per-doc F1 with a paired bootstrap (10K resamples) plus an approximate randomization test. Compare entity-level errors with McNemar's test. Call a model non-inferior if the lower bound of the 95% CI of ΔF1 (open − proprietary) is above −3 points. Apply Holm correction across models.
   * **LLM vs. CRF:** the +10 target counts as met only if the lower bound of the 95% CI of ΔF1 is ≥ +10.
5. **Output:** Parsed-CV and Parsed-Job JSONL files, plus the cache.

## Phase 3 — R3 Text → Graph (M2, Oct 28) — AM, KN

1. **Classical normalization:** rules plus a char-n-gram TF-IDF alias table that maps skills to Lightcast and O*NET IDs.
2. **Modern normalization:** embedding kNN, with an LLM tie-break when the top candidates are close.
3. **Inferred edges:** the LLM proposes `RELATED_TO` and `TYPICAL_FOR` edges, marked `inferred=true`. Run Louvain to find communities.
4. **Neo4j load:** use idempotent `MERGE`, with constraints and indexes on IDs. **Every edge must be `SUPPORTED_BY` an Evidence node.** Add a Cypher check that fails if any edge has no evidence.
5. **Targets:**
   * Top-1 accuracy ≥ 0.90 on the 500 skill mentions.
   * Precision ≥ 0.90 on 100 sampled edges.
6. **Version the Neo4j snapshot** (dump file plus a tag).

## Phase 4 — R4 Classical Ranking End-to-End (M2, Oct 28) — HR

1. **Filters:** location, work mode, and seniority.
2. **Lexical ranking:** BM25, TF-IDF cosine, and Jaccard over normalized skills, keeping the top 200.
3. **Extras:** logistic regression to predict the occupation family, and K-means for diversity.
4. **M2 gate:** a CV goes in and a ranked job list comes out, using only classical components. Smoke-test it on the 5 team CVs.

## Phase 5 — Modern Retrieval, LoRA, Ablation, Judging (Nov 4) — AP, HR, all

1. **Dense + RRF (HR):** embed CVs and postings with a Sentence-BERT bi-encoder, index them in FAISS, and fuse the dense list with the lexical lists using RRF (k = 60).
2. **Relevance judging (all):** for each of the 40 profiles, pool the top 20 jobs from all systems and judge each pair 0/1/2 (800 pairs). Have two people judge a subset and report agreement.
3. **QLoRA (AP):**
   1. A large open model labels ~4K docs (the teacher).
   2. Hand-correct 100 of them, kept apart from the test set.
   3. QLoRA-tune a 7–8B model on Colab/Kaggle.
   4. Convert it to GGUF and serve it through Ollama.

   **Target: within 3 F1 points of the zero-shot teacher**, with latency reported. Test this as non-inferiority: the lower bound of the 95% paired-bootstrap CI of ΔF1 (LoRA − teacher) must be above −3. Compare latency with a Wilcoxon signed-rank test.
4. **SPO ablation (AP):** run free-form SPO extraction on the same documents. Compare it with the ontology path on duplicate-node rate and ΔnDCG@10 (RQ3). Test the duplicate-node rate with a two-proportion z-test plus a bootstrap CI, and per-profile nDCG@10 with a Wilcoxon signed-rank test (report rank-biserial r).

## Phase 6 — Learning-to-Rank & Explanations (M3, Nov 11) — HR, AM

1. **LambdaMART (HR):** LightGBM with `lambdarank`, trained on the judged pairs with cross-validation grouped by profile. Features:
   * Lexical scores: BM25, TF-IDF, Jaccard.
   * Dense score and RRF rank.
   * Skill coverage and experience gap.
   * Graph features: shortest skill path, and `RELATED_TO` bridges to unmet requirements.
2. **Cross-encoder + MMR:** rerank the top 20 with the cross-encoder, then use MMR to pick a diverse top 10.
3. **Both directions:** also run the cascade from a job to its best candidates.
4. **Evaluate** with nDCG@10, MRR, and P@5 per profile. **Target: +15% nDCG@10 over BM25.** Run ablations without graph features, without RRF, and without the cross-encoder.
   * **Significance:** use a paired randomization test (Fisher, 10K permutations, via `ranx`), with a paired t-test as a check. Report a 95% bootstrap CI and Cohen's d_z. Apply Holm correction across the ablation family.
   * **Open vs. proprietary:** use the same tests when the router's LLM choice (e.g., for tie-breaking or edge inference) changes the ranking. Non-inferiority margin: δ = 0.02 nDCG@10.
5. **Explainer (AM):**
   1. Build the template baseline first: it lists matched and missing skills.
   2. Use Cypher to build an evidence bundle: satisfied requirements with their evidence, gaps, and bridging paths.
   3. The LLM writes sentences only from that bundle, and each sentence carries an edge ID.
6. **Verifier/Critic (AM):** reject any sentence with no citation or with an edge ID that is not in the bundle, then regenerate. **Target: faithfulness ≥ 0.95** on 60 explanations. Also report the hallucination rate. Give each model a Wilson 95% CI for faithfulness, and compare open models with the proprietary baseline on the same explanations using McNemar's test.

## Phase 7 — R6 Agents, Router Config, UI (Nov 18) — AU (+all)

1. **LangGraph state graph with 7 agents:** Parser, KG-Build, Matching, Profile Optimizer, Research & Outreach, Scheduler, and Critic. Every agent calls the router, so changing a model is a one-line config change. Prove it with a live swap.
2. **Profile Optimizer:** suggests CV rewrites that only rephrase existing claims (the Critic checks them against the evidence), plus upskilling pointers for skill gaps.
3. **Research & Outreach:** writes company briefs and outreach drafts. **Scheduler:** adds Calendar follow-ups. The user must approve every external action in the UI.
4. **Streamlit UI:**
   * Upload a CV and chat.
   * See ranked jobs with cited explanations.
   * See the skill-gap plan.
   * Approve outreach.
   * View the cost/latency dashboard.
5. **Latency:** **target p50 ranking latency < 3 s.** Precompute the indexes and use the cache. Compare model tiers' per-call latency with a paired Wilcoxon signed-rank test, and give bootstrap CIs for p50/p95.

## Phase 8 — R7 Full Evaluation (M4, Nov 25) — all

1. **Re-run every thread** on the frozen test splits and fill in Table II.
2. **Headline RQ1 matrix:** model × task, showing quality against $/1K docs and p50/p95 latency. Cover Ollama, Groq, OpenRouter, the LoRA model, and the proprietary baseline. Plot it as a Pareto chart. Each cell shows Δ vs. proprietary ± 95% CI, the Holm-adjusted p-value, and a non-inferiority flag (✓/✗).
3. **Answer RQ1–RQ4,** each linked to its evidence. Write up failure cases too.

## Phase 9 — User Study & Deliverables (Final, Dec 2) — AU leads, all

1. **Blind A/B user study:** compare the template explainer with the LLM explainer, or the classical ranking with the full cascade. Recruit 15–20 users and use a 1–5 Likert scale. **Target: mean ≥ 4.0.** Collect consent and anonymize responses. Each user rates both conditions, so compare them with a Wilcoxon signed-rank test (report rank-biserial r). Check the ≥ 4.0 target with a one-sample Wilcoxon test against 4.0 and a bootstrap CI of the mean.
2. **Deliverables:**
   * Report in IEEE format, extending the proposal. Its evaluation section includes the significance testing plan below.
   * Slides.
   * Demo video.
   * Repo with a README and reproduction scripts.

---

## Statistical Significance Testing Plan (addresses proposal feedback)

> **Feedback received:** *"To make the evaluation stronger, consider adding details on which statistical significance tests will be used to compare the performance of the open weight models against the proprietary baseline."*

### General protocol (frozen in Week 1 as `eval/STATS_PLAN.md`)

- **Significance level:** α = 0.05, two-sided. Every result reports a **95% CI and an effect size**, not just a p-value.
- **Paired design:** every model runs on the exact same test items. The unit of analysis is the item: a document (150), a CV profile (40), or an explanation (60).
- **Multiple comparisons:** apply **Holm–Bonferroni** within each family. The families are (a) open models × task vs. proprietary for RQ1, and (b) the R4 ablation set.
- **LLM non-determinism:** use temperature 0 and **3 runs per model**. Report mean ± SD across runs, and use the per-item mean in the tests.
- **Non-inferiority (RQ1, RQ2):** "no significant difference" does not mean the models are equivalent. Margins are fixed in advance: **δ = 3 F1 points** for extraction and **δ = 0.02 nDCG@10** for ranking. An open model is *non-inferior* if the lower bound of the 95% bootstrap CI for (open − proprietary) is above −δ.
- **Tools:**
  * `scipy.stats`: `wilcoxon`, `ttest_rel`, `permutation_test`, `bootstrap`.
  * `statsmodels`: `mcnemar`, `multipletests(method="holm")`.
  * `ranx`: paired IR significance tests.

  Shared helpers live in `eval/stats.py`.

### Test for each comparison

| Comparison | Metric / unit | Test | CI & effect size |
|---|---|---|---|
| **RQ1/R2:** each open model vs. proprietary (extraction) | Micro-F1 per doc (150) | Paired bootstrap (10K) + approximate randomization test | 95% CI of ΔF1; non-inferiority δ = 3 |
| **RQ1/R2:** entity-level errors | Correct/incorrect per gold entity | McNemar's test (exact if discordant pairs < 25) | Odds ratio of discordant pairs |
| **R2:** LLM vs. CRF (+10 target) | F1 per doc | Paired bootstrap | Target met only if CI lower bound ≥ +10 |
| **RQ2/R5:** LoRA vs. zero-shot teacher | F1 per doc | Paired bootstrap, non-inferiority (δ = 3) | 95% CI of ΔF1 |
| **RQ3:** ontology vs. SPO | Duplicate-node rate; nDCG@10 per profile | Two-proportion z-test + bootstrap; Wilcoxon signed-rank | CI of Δ; rank-biserial r |
| **RQ4/R4:** cascade vs. BM25, ablations, open vs. proprietary | nDCG@10, MRR, P@5 per profile (40) | Paired randomization test (10K perms, `ranx`) + paired t-test as a check | 95% bootstrap CI; Cohen's d_z |
| **RQ4/R6:** explanation faithfulness, open vs. proprietary | Faithful yes/no per explanation (60) | McNemar's test; Wilson CI vs. the ≥ 0.95 target | Wilson 95% CI per model |
| **RQ1/R7:** latency | Seconds per call | Wilcoxon signed-rank (paired per doc); bootstrap CI for p50/p95 | Hodges–Lehmann shift |
| **RQ1/R7:** cost | $/1K docs | From router logs; bootstrap CI over docs | Cost ratio vs. proprietary |
| **R6:** user study A/B (within-subject) | 1–5 Likert per user | Wilcoxon signed-rank; one-sample Wilcoxon vs. 4.0 | Rank-biserial r; bootstrap CI of mean |
| Annotation agreement | Items labeled by two annotators | Cohen's κ (extraction, relevance) | κ with 95% CI |

### Power and sample size

- With 40 paired profiles, a paired test at α = 0.05 has about 80% power to detect an effect of d_z ≈ 0.45. Smaller differences may be missed.
- Report an undetected difference as "not detected," with its CI. Only claim two models are equivalent when the non-inferiority test supports it.
- If time allows, add more judged profiles.

### Reporting

The headline RQ1 matrix shows, for each model × task cell:

* quality Δ vs. proprietary ± 95% CI,
* the Holm-adjusted p-value,
* a non-inferiority flag (✓/✗),
* cost and latency.

---

## Success Practices & Risk Mitigation

- **Classical first:** every thread ships its baseline before its modern version, so there is always something to show.
- **Freeze early:** fix the splits, ontology, and annotation guide in Week 1. Change them only through a logged decision.
- **Pre-register the stats plan:** choose tests, margins, and correction families before seeing any results, and never switch tests afterwards.
- **Cost control:** dedupe before any LLM call, cache by content hash, keep full extraction to the 20K working set, and distill into the LoRA model.
- **Rate limits (Groq/OpenRouter):** retry with backoff, batch requests, and have the router fall back to another tier.
- **Annotation bottleneck:** split the 150 docs, 800 pairs, and 60 explanations across all 5 members from Week 1, and track progress in a shared sheet.
- **GPU limits for QLoRA:** use 4-bit, short sequence lengths, and gradient checkpointing. If that fails, fall back to a 3–4B model.
- **Integration risk:** agree on the JSONL and graph contracts (Pydantic) between modules early, and run CI tests on a small fixture set.
- **Schedule check (today, Oct 28, is M2):** confirm the R3 graph load and the classical R4 pipeline both run end to end. If anything slips, cut Phase 7 extras (Scheduler, outreach) before cutting evaluation.

## Milestone Checklist

- [ ] **M1 (Oct 14):** data deduped and split, ontology JSON Schema frozen, annotation guide ready, stats plan frozen
- [ ] **Oct 21:** CRF vs. LLM extraction results, router logging live
- [ ] **M2 (Oct 28):** Neo4j graph loaded with evidence, classical ranking end to end
- [ ] **Nov 4:** QLoRA model served, dense + RRF, SPO ablation, 800 judgments done
- [ ] **M3 (Nov 11):** LambdaMART, cross-encoder, explainer + verifier
- [ ] **Nov 18:** 7 agents, router config, Streamlit UI
- [ ] **M4 (Nov 25):** full evaluation, ablations, cost matrix, all significance tests run with Holm correction
- [ ] **Dec 2 (Final):** user study, report, slides, demo video

## Evaluation Targets (from Table II)

| Thread | Metric | Target | Significance test |
|---|---|---|---|
| R1 | Dedupe precision (200 pairs); split leakage | ≥ 0.95; no overlap | Wilson 95% CI |
| R2 / RQ1 | Per-type F1 on 150 docs | ≥ 0.80; ≥ +10 over CRF | Paired bootstrap + randomization; McNemar; non-inferiority (δ = 3) vs. proprietary |
| R3 | Top-1 normalization acc. (500); edge precision (100) | ≥ 0.90 each | Wilson 95% CI |
| R3 / RQ3 | Ontology vs. SPO: duplicate-node rate, ΔnDCG@10 | Report both | Two-proportion z-test; Wilcoxon signed-rank |
| R4 / RQ4 | nDCG@10, MRR, P@5 on 40 profiles | +15% nDCG over BM25 | Paired randomization test (`ranx`), Holm |
| R4 ablation | −graph, −RRF, −cross-encoder | Report each | Paired randomization test, Holm |
| R5 / RQ2 | LoRA F1 & latency vs. zero-shot teacher | Within 3 pts | Non-inferiority via paired bootstrap CI; Wilcoxon (latency) |
| R6 / RQ4 | Explanation faithfulness (60) | ≥ 0.95 | Wilson CI; McNemar (open vs. proprietary) |
| R6 | Blind A/B Likert (15–20 users) | Mean ≥ 4.0 | Wilcoxon signed-rank; one-sample Wilcoxon vs. 4.0 |
| R7 / RQ1 | $/1K docs, p50/p95 latency | < 3 s ranking | Wilcoxon signed-rank; bootstrap CIs |
