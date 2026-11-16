"""Synthetic prices and news. Everything here is invented; nothing is market data.

Two worlds share every random draw: "null" (news has no link to prices) and "planted"
(hidden tone u adds BETA per day of drift on days t+1..t+5 after the assigned day t).
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .dedup import expand_and_dedup
from .scorer import NEG, POS, score
from .timeutil import assign, weekday_calendar

SEED = 7
BETA = 0.002          # fixed before running anything
N_SYMBOLS = 30
MARKET = "US"
START, END = date(2020, 1, 1), date(2022, 12, 31)
AS_OF = datetime(2023, 1, 1, tzinfo=timezone.utc)
HOLIDAYS = ["2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03", "2020-09-07",
            "2020-11-26", "2020-12-25", "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31",
            "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24", "2022-01-17", "2022-02-21", "2022-04-15",
            "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26"]
FILLER = ["shares", "company", "quarter", "outlook", "report", "analysts", "results", "guidance",
          "market", "update", "sector", "demand", "revenue", "orders", "management", "forecast"]


def _headline(rng, syms, u):
    p = 1.0 / (1.0 + np.exp(-1.5 * u))
    k = 0 if rng.random() < 0.15 else int(rng.integers(1, 4))
    words = [POS[rng.integers(len(POS))] if rng.random() < p else NEG[rng.integers(len(NEG))] for _ in range(k)]
    body = [FILLER[i] for i in rng.integers(0, len(FILLER), int(rng.integers(3, 6)))] + words
    body = [body[i] for i in rng.permutation(len(body))]
    return " ".join(syms) + " " + " ".join(body)


def _news(rng, symbols):
    n_cal = (END - START).days + 1
    midnight = datetime(START.year, START.month, START.day, tzinfo=timezone.utc)
    arts, bad = [], 0
    for s in symbols:
        counts = rng.poisson(rng.uniform(0.5, 3.0) / 7.0, n_cal)
        for day in np.nonzero(counts)[0]:
            for _ in range(counts[day]):
                ts = midnight + timedelta(days=int(day), seconds=int(rng.integers(0, 86400)))
                u, r = float(rng.normal()), rng.random()
                a = {"symbols": [s], "u": u, "kind": "timestamp", "precision": "timestamp",
                     "published": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}
                if r < 0.05:
                    a.update(kind="date-only", precision="date", published=ts.date().isoformat())
                elif r < 0.09:
                    others = [o for o in rng.choice(symbols, int(rng.integers(1, 3)), replace=False) if o != s]
                    a.update(kind="multi-ticker", symbols=[s] + [str(o) for o in others])
                elif r < 0.10:
                    a["kind"] = "invalid"
                    a["published"] = [None, ts.strftime("%Y-%m-%dT%H:%M:%S"), "2023-06-01T12:00:00Z"][bad % 3]
                    bad += 1
                a["headline"] = _headline(rng, a["symbols"], u)
                arts.append(a)
                if 0.10 <= r < 0.13:  # syndicated copy a few hours later, different case/punctuation
                    later = ts + timedelta(hours=float(rng.uniform(1, 40)))
                    arts.append(dict(a, kind="duplicate", headline=a["headline"].upper() + "!",
                                     published=later.strftime("%Y-%m-%dT%H:%M:%SZ")))
    for i, a in enumerate(arts):
        a["id"] = i
    return arts


def process_articles(articles, cal, market=MARKET, as_of=AS_OF):
    """Assign, de-duplicate and score. Annotates articles in place; returns assignment DataFrame."""
    valid = []
    for a in articles:
        day, reason = assign(a["published"], market, cal, a["precision"], as_of)
        sc = score(a["headline"], a["symbols"])
        a.update(assigned=day, reason=reason, x=sc["x"], bucket=sc["bucket"], matched=sc["matched"], dropped_dup=False)
        if day is not None:
            p = a["published"]
            ts = datetime.fromisoformat((p + "T00:00:00+00:00") if a["precision"] == "date" else p.replace("Z", "+00:00"))
            valid.append(dict(a, ts=ts))
    kept, dropped = expand_and_dedup(valid)
    for d in dropped:
        articles[d["id"]]["dropped_dup"] = True
    return pd.DataFrame([{"id": k["id"], "symbol": k["symbol"], "t": cal.index[k["assigned"]], "x": k["x"],
                          "w": k["weight"], "u": k["u"], "bucket": k["bucket"]} for k in kept])


def _shift(a, j):
    out = np.zeros_like(a)
    out[:, j:] = a[:, :-j]
    return out


def _bars(draws, mbar, beta, symbols, days):
    sig = draws["sig"][:, None]
    d1 = beta * _shift(mbar, 1)                                   # effect on day t+1
    later = beta * sum(_shift(mbar, j) for j in range(2, 6))      # days t+2..t+5
    gap = 0.4 * sig * draws["gap"] + 0.5 * d1                     # half of day t+1 goes overnight
    intra = 0.9 * sig * draws["intra"] + 0.5 * d1 + later
    close = draws["p0"][:, None] * np.cumprod((1 + gap) * (1 + intra), axis=1)
    prev = np.concatenate([draws["p0"][:, None], close[:, :-1]], axis=1)
    open_ = prev * (1 + gap)
    high = np.maximum(open_, close) * (1 + 0.3 * sig * np.abs(draws["hi"]))
    low = np.minimum(open_, close) * (1 - 0.3 * sig * np.abs(draws["lo"]))
    n = len(days)
    return pd.DataFrame({"symbol": np.repeat(symbols, n), "date": np.tile(np.array(days, dtype=object), len(symbols)),
                         "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(), "close": close.ravel()})


def generate(seed=SEED, beta=BETA):
    rng = np.random.default_rng(seed)
    cal = weekday_calendar(START, END, HOLIDAYS)
    symbols = [f"SYN{i:02d}" for i in range(1, N_SYMBOLS + 1)]
    articles = _news(rng, symbols)
    assignments = process_articles(articles, cal)
    shape = (len(symbols), len(cal.days))
    draws = {"sig": rng.uniform(0.01, 0.03, len(symbols)), "p0": rng.uniform(20, 200, len(symbols)),
             "gap": rng.normal(size=shape), "intra": rng.normal(size=shape),
             "hi": rng.normal(size=shape), "lo": rng.normal(size=shape)}
    si = {s: i for i, s in enumerate(symbols)}
    usum, cnt = np.zeros(shape), np.zeros(shape)
    idx = (assignments["symbol"].map(si).to_numpy(int), assignments["t"].to_numpy(int))
    np.add.at(usum, idx, assignments["u"].to_numpy(float))
    np.add.at(cnt, idx, 1.0)
    mbar = np.divide(usum, cnt, out=np.zeros(shape), where=cnt > 0)   # mean hidden tone per (stock, day)
    bars = {"null": _bars(draws, mbar, 0.0, symbols, cal.days), "planted": _bars(draws, mbar, beta, symbols, cal.days)}
    return {"cal": cal, "symbols": symbols, "articles": articles, "assignments": assignments, "bars": bars,
            "seed": seed, "beta": beta}
