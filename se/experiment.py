"""Baseline (B) vs baseline + sentiment (C) vs shuffled-sentiment placebo (C2). Synthetic data only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import add_ranks, compute_features
from .labels import H, compute_labels
from .stats import paired_bootstrap

BASE_COLS = ["atr_pct", "ret_5", "ret_20", "vol_10"]
SENT_COLS = ["S1_r", "S5_r", "NVS_r", "ABN_r", "has_news"]
EMBARGO = 5
MIN_IC_N = 20
N_PLACEBO = 20
PLACEBO_SEEDS = list(range(1000, 1000 + N_PLACEBO))
ALPHA = 0.05          # flag threshold, fixed before running
TEST_YEAR = 2022


def sentiment_features(assignments, symbols, n_days):
    """Articles without lexicon coverage carry no score, so they are left out of the features."""
    a = assignments[assignments["bucket"] != "no lexicon coverage"]
    return add_ranks(compute_features(a, symbols, n_days))


def build_panel(bars, feats, cal):
    lab = compute_labels(bars)
    lab["t"] = lab["date"].map(cal.index)
    by = lab.groupby("symbol")["close"]
    lab["ret_5"], lab["ret_20"] = by.pct_change(5), by.pct_change(20)
    lab["vol_10"] = by.pct_change().groupby(lab["symbol"]).transform(lambda s: s.rolling(10).std())
    panel = lab.merge(feats[["symbol", "t"] + SENT_COLS], on=["symbol", "t"])
    return panel.dropna(subset=BASE_COLS).sort_values(["t", "symbol"]).reset_index(drop=True)


def split(panel, cal):
    first_test = next(i for i, d in enumerate(cal.days) if d.year == TEST_YEAR)
    train = panel[(panel["t"] + H + 1 < first_test - EMBARGO) & panel["l_exec"].notna()]
    test = panel[(panel["t"] >= first_test) & panel["z_exec"].notna()]
    return train, test


def fit_score(train, test, cols):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
    model.fit(train[cols].to_numpy(), train["l_exec"].to_numpy().astype(int))
    return model.decision_function(test[cols].to_numpy())   # a ranking score, not a probability


def daily_ic(t, score, z):
    """Spearman rank correlation of score vs Z_exec per date (dates with >= MIN_IC_N stocks)."""
    t = pd.Series(np.asarray(t))
    a = pd.Series(np.asarray(score, float)).groupby(t).rank()
    b = pd.Series(np.asarray(z, float)).groupby(t).rank()
    a, b = a - a.groupby(t).transform("mean"), b - b.groupby(t).transform("mean")
    ic = (a * b).groupby(t).sum() / np.sqrt((a * a).groupby(t).sum() * (b * b).groupby(t).sum())
    return ic[t.groupby(t).size() >= MIN_IC_N]


def shuffle_block(panel, seed):
    """Permute the whole sentiment block across stocks within each date."""
    rng = np.random.default_rng(seed)
    vals = panel[SENT_COLS].to_numpy().copy()
    for idx in panel.groupby("t").indices.values():
        vals[idx] = vals[rng.permutation(idx)]
    out = panel.copy()
    out[SENT_COLS] = vals
    return out


def verdict(world, p, placebo_pass):
    """Legacy signature; the text comes from the single verdict function."""
    from .verdicts import verdict_text
    return verdict_text("synthetic_validation", world, p < ALPHA, placebo_pass, False)


def run_world(world, bars, feats, cal, boot_seed=12345):
    panel = build_panel(bars, feats, cal)
    train, test = split(panel, cal)
    ic_b = daily_ic(test["t"], fit_score(train, test, BASE_COLS), test["z_exec"])
    ic_c = daily_ic(test["t"], fit_score(train, test, BASE_COLS + SENT_COLS), test["z_exec"])
    boot = paired_bootstrap((ic_c - ic_b).to_numpy(), seed=boot_seed)
    placebo = []
    for seed in PLACEBO_SEEDS:
        tr2, te2 = split(shuffle_block(panel, seed), cal)
        ic_2 = daily_ic(te2["t"], fit_score(tr2, te2, BASE_COLS + SENT_COLS), te2["z_exec"])
        placebo.append(float((ic_2 - ic_b).mean()))
    passed = bool(boot["delta"] >= max(placebo))
    return {"world": world, "ic_b": float(ic_b.mean()), "ic_c": float(ic_c.mean()), "delta_ic": boot["delta"],
            "p_value": boot["p"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "placebo_deltas": placebo, "placebo_max": max(placebo), "placebo_pass": passed,
            "n_test_days": boot["n"], "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
            "alpha": ALPHA, "bootstrap": {k: boot[k] for k in ("mean_block", "n_resamples", "seed")},
            "verdict": verdict(world, boot["p"], passed)}
