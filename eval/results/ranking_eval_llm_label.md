# Ranking evaluation (labels: llm_label, 25 profiles)

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
