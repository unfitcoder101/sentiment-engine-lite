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
