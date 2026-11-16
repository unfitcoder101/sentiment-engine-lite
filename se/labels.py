"""Execution-based labels: open(t+1) -> open(t+11) return scaled by ATR at signal date t.

This is its own label definition; no equivalence with any close-to-close label is claimed.
"""
import numpy as np
import pandas as pd

H = 10
ATR_WINDOW = 14
ATR_FLOOR = 0.001
MIN_GROUP = 20
TOP_SHARE = 0.30


def true_range(high, low, close):
    high, low, close = (np.asarray(v, float) for v in (high, low, close))
    prev = np.concatenate([[np.nan], close[:-1]])
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def atr(high, low, close, window=ATR_WINDOW):
    tr = pd.Series(true_range(high, low, close))
    return tr.rolling(window, min_periods=window).mean().to_numpy()


def atr_pct(high, low, close, window=ATR_WINDOW):
    return atr(high, low, close, window) / np.asarray(close, float)


def r_exec(open_, h=H):
    o = pd.Series(np.asarray(open_, float))
    return (o.shift(-(h + 1)) / o.shift(-1) - 1).to_numpy()


def z_exec(r, atr_pct_t):
    return np.asarray(r, float) / np.maximum(np.asarray(atr_pct_t, float), ATR_FLOOR)


def average_ranks(values):
    return pd.Series(np.asarray(values, float)).rank(method="average").to_numpy()


def l_exec_group(z):
    """Top-30% label within one (market, date) group; NaN if < MIN_GROUP valid stocks."""
    z = np.asarray(z, float)
    ok = ~np.isnan(z)
    n = int(ok.sum())
    out = np.full(z.shape, np.nan)
    if n < MIN_GROUP:
        return out
    pct = average_ranks(z) / n
    out[ok] = (pct[ok] >= (1 - TOP_SHARE) - 1e-12).astype(float)
    return out


def compute_labels(bars, market="US", h=H, window=ATR_WINDOW):
    """bars: DataFrame symbol,date,open,high,low,close with one row per calendar day per symbol."""
    parts = []
    for _, g in bars.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = g.copy()
        px = g[["open", "high", "low", "close"]].astype(float)
        px.loc[~(px > 0).all(axis=1)] = np.nan  # missing or non-positive bar
        g["atr_pct"] = atr_pct(px["high"], px["low"], px["close"], window)
        g["r_exec"] = r_exec(px["open"], h)
        g["z_exec"] = z_exec(g["r_exec"], g["atr_pct"])
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df["market"] = market
    df["l_exec"] = df.groupby(["market", "date"])["z_exec"].transform(lambda z: l_exec_group(z.to_numpy()))
    return df
