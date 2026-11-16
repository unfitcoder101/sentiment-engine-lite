"""Paired stationary bootstrap on a daily difference series d_t = IC_C(t) - IC_B(t)."""
import numpy as np


def stationary_bootstrap_means(d, mean_block=20, n_resamples=2000, seed=12345):
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = np.empty((n_resamples, n), dtype=int)
    idx[:, 0] = rng.integers(0, n, n_resamples)
    restart = rng.random((n_resamples, n)) < 1.0 / mean_block
    fresh = rng.integers(0, n, (n_resamples, n))
    for j in range(1, n):
        idx[:, j] = np.where(restart[:, j], fresh[:, j], (idx[:, j - 1] + 1) % n)
    return d[idx].mean(axis=1)


def paired_bootstrap(d, mean_block=20, n_resamples=2000, seed=12345):
    """One-sided p = (1 + #{centred resampled mean >= observed}) / (B + 1); unadjusted 95% interval."""
    d = np.asarray(d, float)
    obs = float(d.mean())
    means = stationary_bootstrap_means(d, mean_block, n_resamples, seed)
    p = (1 + int(((means - obs) >= obs).sum())) / (n_resamples + 1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": obs, "p": float(p), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(d)), "mean_block": mean_block, "n_resamples": n_resamples, "seed": seed}


# ---------- Phase 5 additions ----------
from statistics import NormalDist

_IDX_CACHE = {}


def _indices(n, mean_block, n_resamples, seed):
    """Same index scheme as stationary_bootstrap_means, cached so all comparisons share resamples."""
    key = (n, mean_block, n_resamples, seed)
    if key not in _IDX_CACHE:
        rng = np.random.default_rng(seed)
        idx = np.empty((n_resamples, n), dtype=int)
        idx[:, 0] = rng.integers(0, n, n_resamples)
        restart = rng.random((n_resamples, n)) < 1.0 / mean_block
        fresh = rng.integers(0, n, (n_resamples, n))
        for j in range(1, n):
            idx[:, j] = np.where(restart[:, j], fresh[:, j], (idx[:, j - 1] + 1) % n)
        if len(_IDX_CACHE) > 8:
            _IDX_CACHE.clear()
        _IDX_CACHE[key] = idx
    return _IDX_CACHE[key]


def compare(d, mean_block=20, n_resamples=2000, seed=12345):
    """paired_bootstrap plus the bootstrap standard error. Interval is unadjusted."""
    d = np.asarray(d, float)
    obs = float(d.mean())
    means = d[_indices(len(d), mean_block, n_resamples, seed)].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": obs, "p": (1 + int(((means - obs) >= obs).sum())) / (n_resamples + 1),
            "ci_low": float(lo), "ci_high": float(hi), "se": float(means.std(ddof=1)), "n": int(len(d)),
            "mean_block": mean_block, "n_resamples": n_resamples, "seed": seed, "interval": "unadjusted"}


def holm(pvals):
    """Holm-Bonferroni: adjusted p(i) = min(1, max over j<=i of (m-j+1)*p(j)), returned in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        out[i] = min(1.0, running)
    return out


def mde(se, alpha=0.05, m=3, power_z=0.8416):
    """Approximate minimum detectable delta: SE * (z_{1-alpha/m} + z_power). An approximation only."""
    return se * (NormalDist().inv_cdf(1 - alpha / m) + power_z)
