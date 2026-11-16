"""Rank metrics written out by hand: Spearman via average-tie ranks, AUC via the rank statistic."""
import math

import numpy as np


def avg_rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    b = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])      # tie-group boundaries
    r = np.empty(len(a))
    r[order] = np.repeat((b[:-1] + b[1:] + 1) / 2.0, np.diff(b))
    return r


def spearman(a, b):
    ra, rb = avg_rank(a), avg_rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def auc(score, label):
    """P(score of a random 1 > score of a random 0), ties count half."""
    label = np.asarray(label) == 1
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((avg_rank(score)[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def top_k(score, k_share):
    k = max(1, math.ceil(k_share * len(score) - 1e-9))
    return np.argsort(-np.asarray(score, float), kind="mergesort")[:k]


def daily_secondary(t, score, z, r, l, min_n, k_share, cost):
    """Means over dates (t must be sorted) with at least min_n stocks."""
    t = np.asarray(t)
    b = np.flatnonzero(np.r_[True, t[1:] != t[:-1], True])
    out = {"ic_z": [], "ic_r": [], "auc": [], "precision_at_k": [], "net_topk": []}
    for i, j in zip(b[:-1], b[1:]):
        if j - i < min_n:
            continue
        s, rr, ll = score[i:j], r[i:j], l[i:j]
        top = top_k(s, k_share)
        out["ic_z"].append(spearman(s, z[i:j]))
        out["ic_r"].append(spearman(s, rr))
        out["auc"].append(auc(s, ll))
        out["precision_at_k"].append(float(ll[top].mean()))
        out["net_topk"].append(float(rr[top].mean() - np.median(rr) - cost))
    res = {k: (float(np.nanmean(v)) if v else None) for k, v in out.items()}
    res["n_dates"] = len(out["ic_z"])
    return res
