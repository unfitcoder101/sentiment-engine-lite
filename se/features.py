"""Sentiment features per (stock, trading-day index t). Only days <= t are ever read."""
import numpy as np
import pandas as pd

WARMUP = 64      # first 64 trading days excluded for everyone
BASE = 60        # baseline window t-64..t-5
MIN_BASE = 10
MIN_RANK = 5
FEATURES = ["S1", "S5", "NVS", "ABN"]


def _win(a, lo_off, hi_off):
    """Sum over days t-lo_off..t-hi_off (NaN where t < lo_off)."""
    n = a.shape[1]
    c = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
    out = np.full(a.shape, np.nan)
    t = np.arange(lo_off, n)
    out[:, lo_off:] = c[:, t - hi_off + 1] - c[:, t - lo_off]
    return out


def compute_features(assignments, symbols, n_days):
    """assignments: DataFrame with symbol, t (trading-day index), x (score), w (weight)."""
    si = {s: i for i, s in enumerate(symbols)}
    wx, w, n = (np.zeros((len(symbols), n_days)) for _ in range(3))
    a = assignments[(assignments["t"] >= 0) & (assignments["t"] < n_days)]
    idx = (a["symbol"].map(si).to_numpy(int), a["t"].to_numpy(int))
    np.add.at(wx, idx, (a["w"] * a["x"]).to_numpy(float))
    np.add.at(w, idx, a["w"].to_numpy(float))
    np.add.at(n, idx, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(n > 0, wx / w, np.nan)
        n5, nb = _win(n, 4, 0), _win(n, 64, 5)
        s5 = np.where(n5 >= 1, _win(wx, 4, 0) / _win(w, 4, 0), np.nan)
        nvs = np.log1p(n5) - np.log1p(5 * nb / BASE)
        abn = np.where(nb >= MIN_BASE, s5 - _win(wx, 64, 5) / _win(w, 64, 5), np.nan)
    ts = np.arange(WARMUP, n_days)
    cols = {"S1": s1, "S5": s5, "NVS": nvs, "ABN": abn, "has_news": (n5 >= 1).astype(float), "N1": n, "N5": n5}
    df = pd.DataFrame({"symbol": np.repeat(symbols, len(ts)), "t": np.tile(ts, len(symbols))})
    for k, v in cols.items():
        df[k] = v[:, WARMUP:].ravel()
    return df


def frac_rank(values):
    """(rank-0.5)/n among non-missing (average ties); missing -> 0.5; < MIN_RANK non-missing -> all 0.5."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    n = int(ok.sum())
    out = np.full(v.shape, 0.5)
    if n >= MIN_RANK:
        out[ok] = (pd.Series(v[ok]).rank(method="average").to_numpy() - 0.5) / n
    return out


def add_ranks(df):
    df = df.copy()
    for f in FEATURES:
        df[f + "_r"] = df.groupby("t")[f].transform(lambda s: frac_rank(s.to_numpy()))
    return df
