"""Pre-registered significance tests (eval/STATS_PLAN.md). All paired, item-level, seed 573.

Every function returns a plain dict so results drop straight into the JSON reports.
"""

import numpy as np
from scipy import stats as st
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

SEED, B = 573, 10_000


def _rng(seed=SEED):
    return np.random.default_rng(seed)


def paired_bootstrap(a, b, n: int = B, seed: int = SEED, alpha: float = 0.05) -> dict:
    """CI of mean(a - b) by resampling items."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = _rng(seed).integers(0, len(d), (n, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(d))}


def randomization_test(a, b, n: int = B, seed: int = SEED) -> dict:
    """Approximate (paired) randomization test: randomly swap each pair's labels; two-sided p for mean diff."""
    d = np.asarray(a, float) - np.asarray(b, float)
    signs = _rng(seed).choice([-1, 1], (n, len(d)))
    null = np.abs((signs * d).mean(1))
    p = (np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1)
    return {"p": float(p)}


def noninferiority(a, b, delta: float, **kw) -> dict:
    """Non-inferior if the lower 95% bound of mean(a - b) is above -delta (STATS_PLAN)."""
    ci = paired_bootstrap(a, b, **kw)
    return {**ci, "delta": delta, "noninferior": bool(ci["ci_low"] > -delta)}


def mcnemar(correct_a, correct_b) -> dict:
    """Paired correct/incorrect per item; exact binomial when there are < 25 discordant pairs."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    table = [[int(np.sum(a & b)), n01], [n10, int(np.sum(~a & ~b))]]
    res = _mcnemar(table, exact=(n01 + n10) < 25)
    return {"a_only": n01, "b_only": n10, "p": float(res.pvalue),
            "odds_ratio": float((n01 + 0.5) / (n10 + 0.5))}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni within one family: {name: p} -> {name: adjusted p}."""
    if not pvalues:
        return {}
    names = list(pvalues)
    adj = multipletests([pvalues[k] for k in names], method="holm")[1]
    return {k: float(v) for k, v in zip(names, adj)}


def wilson(successes: int, n: int, alpha: float = 0.05) -> dict:
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson") if n else (float("nan"),) * 2
    return {"p_hat": successes / n if n else float("nan"), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def wilcoxon(a, b) -> dict:
    """Wilcoxon signed-rank on paired values, with rank-biserial r and the Hodges-Lehmann shift."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"p": 1.0, "rank_biserial_r": 0.0, "hodges_lehmann": 0.0, "n": int(len(d))}
    res = st.wilcoxon(nz)
    ranks = st.rankdata(np.abs(nz))
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    walsh = (nz[:, None] + nz[None, :])[np.triu_indices(len(nz))] / 2
    return {"p": float(res.pvalue), "rank_biserial_r": float(r), "hodges_lehmann": float(np.median(walsh)),
            "n": int(len(d))}


def one_sample_wilcoxon(x, mu: float) -> dict:
    x = np.asarray(x, float)
    res = wilcoxon(x, np.full_like(x, mu))
    res.update(bootstrap_ci(x))
    return res


def bootstrap_ci(x, stat=np.mean, n: int = B, seed: int = SEED, alpha: float = 0.05) -> dict:
    x = np.asarray(x, float)
    idx = _rng(seed).integers(0, len(x), (n, len(x)))
    vals = np.apply_along_axis(stat, 1, x[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(stat(x)), "ci_low": float(lo), "ci_high": float(hi)}


def paired_t(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    res = st.ttest_rel(a, b)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return {"t": float(res.statistic), "p": float(res.pvalue), "cohen_dz": float(dz)}


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> dict:
    z, p = proportions_ztest([x1, x2], [n1, n2])
    return {"p1": x1 / n1, "p2": x2 / n2, "z": float(z), "p": float(p)}


def kappa(labels_a, labels_b, n: int = 2000, seed: int = SEED) -> dict:
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    k = cohen_kappa_score(a, b)
    idx = _rng(seed).integers(0, len(a), (n, len(a)))
    boots = [cohen_kappa_score(a[i], b[i]) for i in idx]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return {"kappa": float(k), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(a))}
