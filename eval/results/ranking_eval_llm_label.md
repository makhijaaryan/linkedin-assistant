# Ranking evaluation (labels: llm_label, 25 profiles)

| System | nDCG@10 | MRR@10 | P@5 |
|---|---|---|---|
| dense | 0.838 | 0.973 | 0.912 |
| ablate_no_graph | 0.833 | 0.980 | 0.968 |
| ablate_no_cross_encoder | 0.830 | 0.973 | 0.960 |
| ablate_no_rrf | 0.821 | 0.970 | 0.944 |
| ltr_full | 0.819 | 0.973 | 0.944 |
| cross_encoder | 0.806 | 1.000 | 0.912 |
| cascade_mmr | 0.806 | 0.973 | 0.944 |
| tfidf | 0.784 | 0.947 | 0.912 |
| rrf | 0.783 | 0.950 | 0.928 |
| bm25 | 0.714 | 0.930 | 0.840 |
| classical | 0.532 | 0.925 | 0.656 |
| jaccard | 0.417 | 0.607 | 0.464 |

Cascade vs BM25: Δ nDCG@10 = +0.092 [+0.017, +0.174], relative +12.8%, p = 0.0277, d_z = 0.45; +15% target not met.

| Ablation (vs full LambdaMART) | Δ nDCG@10 [95% CI] | p (Holm) |
|---|---|---|
| ablate_no_graph | -0.015 [-0.035, +0.005] | 0.4896 |
| ablate_no_rrf | -0.003 [-0.024, +0.018] | 0.8003 |
| ablate_no_cross_encoder | -0.012 [-0.027, +0.003] | 0.4896 |
