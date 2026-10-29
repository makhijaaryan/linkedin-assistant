# Statistical Analysis Plan (pre-registered, frozen 2026-10-28)

Written before any results, in response to the proposal feedback: *"consider adding details on which statistical significance tests will be used to compare the performance of the open weight models against the proprietary baseline."* Changing anything here after results exist requires a logged decision. Settings are mirrored in `config.yaml` → `stats`.

## Protocol
- **Significance level:** α = 0.05, two-sided. Every result reports a **95% CI and an effect size**, not just a p-value.
- **Paired design:** all models run on the same frozen test items (`eval/frozen/test_sets.json`). The unit of analysis is the item: document (150), CV profile (40), or explanation (60).
- **Multiple comparisons:** Holm–Bonferroni within each family. Family A is open models × task vs. the proprietary baseline (RQ1). Family B is the R4 ablations.
- **LLM non-determinism:** temperature 0 and 3 runs per model. Report mean ± SD across runs, and test on the per-item mean.
- **Non-inferiority (RQ1, RQ2):** margins are δ = 3 F1 points (extraction) and δ = 0.02 nDCG@10 (ranking). A model is *non-inferior* if the lower bound of the 95% paired-bootstrap CI of (open − proprietary) is above −δ. A non-significant difference is never reported as equivalence.
- **Resampling:** 10,000 bootstrap resamples and 10,000 permutations, with seed 573.

## Tests by comparison

| Comparison | Metric / unit | Test | CI & effect size |
|---|---|---|---|
| Each open model vs. proprietary (extraction) | Micro-F1 per doc | Paired bootstrap + approximate randomization | 95% CI of ΔF1; non-inferiority δ = 3 |
| Entity-level errors, open vs. proprietary | Correct/incorrect per gold entity | McNemar (exact if < 25 discordant) | Discordant-pair odds ratio |
| LLM vs. CRF (target +10 F1) | F1 per doc | Paired bootstrap | Target met only if the CI lower bound is ≥ +10 |
| LoRA vs. zero-shot teacher | F1 per doc | Paired bootstrap, non-inferiority δ = 3 | 95% CI of ΔF1 |
| Ontology vs. free-form SPO | Duplicate-node rate; nDCG@10 per profile | Two-proportion z-test + bootstrap; Wilcoxon signed-rank | CI of Δ; rank-biserial r |
| Cascade vs. BM25 (target +15%); ablations; open vs. proprietary | nDCG@10, MRR, P@5 per profile | Paired randomization test (`ranx`), paired t-test as a check | 95% bootstrap CI; Cohen's d_z |
| Explanation faithfulness, open vs. proprietary | Faithful yes/no per explanation | McNemar; Wilson CI vs. ≥ 0.95 | Wilson 95% CI |
| Latency | Seconds per call (paired per doc) | Wilcoxon signed-rank; bootstrap CI for p50/p95 | Hodges–Lehmann shift |
| Cost | $/1K docs | From router logs; bootstrap CI over docs | Ratio vs. proprietary |
| User study A/B (within-subject) | 1–5 Likert per user | Wilcoxon signed-rank; one-sample Wilcoxon vs. 4.0 | Rank-biserial r; bootstrap CI of mean |
| Annotation agreement | Doubly-labeled items | Cohen's κ (solo fallback: intra-annotator κ on a 30-doc re-label) | κ with 95% CI |
| Dedupe precision (target ≥ 0.95) | 200 hand-checked pairs | Wilson CI | Wilson 95% CI |

## Power
With 40 paired profiles, a paired test at α = 0.05 has about 80% power to detect an effect of d_z ≈ 0.45. Smaller differences are reported as "not detected" with their CI, and only the non-inferiority result can support an equivalence claim.
