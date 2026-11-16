import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from se import dedup, features, labels, timeutil
from se.timeutil import TradingCalendar, assign, cutoff_utc, demo_2023_calendar

CAL = demo_2023_calendar()
UTC = timezone.utc


# ---------- timeutil ----------
@pytest.mark.parametrize("value,precision,expected,reason", [
    ("2023-03-14T19:30:00Z", "timestamp", "2023-03-14", "same_day"),
    ("2023-03-14T19:30:01Z", "timestamp", "2023-03-15", "after_cutoff"),
    ("2023-01-10T20:30:00Z", "timestamp", "2023-01-10", "same_day"),
    ("2023-03-18T15:00:00Z", "timestamp", "2023-03-20", "weekend"),
    ("2023-03-17T21:00:00Z", "timestamp", "2023-03-20", "after_cutoff"),
    ("2023-03-14", "date", "2023-03-15", "date_only"),
    ("2023-03-17", "date", "2023-03-20", "date_only"),
    ("2023-11-22T21:00:00Z", "timestamp", "2023-11-24", "after_cutoff"),
    ("2023-11-23T15:00:00Z", "timestamp", "2023-11-24", "holiday"),
])
def test_us_cutoff_cases(value, precision, expected, reason):
    got, why = assign(value, "US", CAL, precision)
    assert got == date.fromisoformat(expected) and why == reason


def test_cutoff_microsecond():
    at = datetime(2023, 3, 14, 19, 30, 0, 0, tzinfo=UTC)
    assert assign(at, "US", CAL)[0] == date(2023, 3, 14)
    assert assign(at + timedelta(microseconds=1), "US", CAL)[0] == date(2023, 3, 15)


def test_invalid_inputs():
    assert assign("2023-03-14T19:30:00", "US", CAL) == (None, "naive_timestamp")
    assert assign(datetime(2023, 3, 14, 12), "US", CAL) == (None, "naive_timestamp")
    assert assign(None, "US", CAL) == (None, "missing")
    as_of = datetime(2023, 3, 14, tzinfo=UTC)
    assert assign("2023-03-15T12:00:00Z", "US", CAL, as_of=as_of) == (None, "after_as_of")


def test_nse_and_calendar():
    # 09:30 UTC = 15:00 IST exactly at the NSE cutoff
    assert assign("2023-03-14T09:30:00Z", "NSE", CAL) == (date(2023, 3, 14), "same_day")
    assert assign("2023-03-14T09:30:01Z", "NSE", CAL)[0] == date(2023, 3, 15)
    assert CAL.next_trading_day(date(2023, 3, 14)) == date(2023, 3, 15)  # strictly after
    assert not CAL.is_trading_day(date(2023, 11, 23))
    assert all(r["match"] for r in timeutil.run_cutoff_cases())


# ---------- dedup ----------
def test_normalise():
    assert dedup.normalise("  SYN01:  Shares—RALLY, again!! ", ["SYN01"]) == "shares rally again"


def test_dedup_48h_window():
    t0 = datetime(2023, 3, 1, tzinfo=UTC)
    arts = [{"id": h, "symbols": ["AAA"], "headline": "AAA beats forecasts", "ts": t0 + timedelta(hours=h)}
            for h in (0, 47, 49, 96)]
    kept, dropped = dedup.expand_and_dedup(arts)
    assert [k["id"] for k in kept] == [0, 49]
    assert [d["id"] for d in dropped] == [47, 96]


def test_dedup_exact_48h_dropped_and_multi_ticker_weight():
    t0 = datetime(2023, 3, 1, tzinfo=UTC)
    arts = [{"id": 0, "symbols": ["AAA"], "headline": "x y", "ts": t0},
            {"id": 1, "symbols": ["AAA"], "headline": "X, Y!", "ts": t0 + timedelta(hours=48)},
            {"id": 2, "symbols": ["AAA", "BBB", "CCC", "DDD"], "headline": "AAA BBB CCC DDD merge", "ts": t0}]
    kept, _ = dedup.expand_and_dedup(arts)
    assert sorted(k["id"] for k in kept) == [0, 2, 2, 2, 2]
    assert all(k["weight"] == 0.25 for k in kept if k["id"] == 2)


# ---------- labels ----------
def test_tr_atr_exact():
    h, l, c = [101, 103, 102, 104], [99, 101, 100, 102], [100, 102, 101, 103]
    tr = labels.true_range(h, l, c)
    assert math.isnan(tr[0]) and list(tr[1:]) == [3, 2, 3]
    assert labels.atr(h, l, c, 3)[3] == pytest.approx(2.666667, abs=1e-6)
    assert labels.atr_pct(h, l, c, 3)[3] == pytest.approx(0.025890, abs=1e-6)
    assert math.isnan(labels.atr(h, l, c, 3)[2])


def test_r_exec_z_exec_exact():
    o = np.full(12, 100.0)
    o[1], o[11] = 101.0, 108.9
    r = labels.r_exec(o)
    assert r[0] == pytest.approx(0.078218, abs=1e-6)
    assert math.isnan(r[1])
    assert labels.z_exec(r[0], 0.02) == pytest.approx(3.910891, abs=1e-6)
    assert labels.z_exec(0.01, 0.0004) == pytest.approx(10.0)  # floor 0.001


def test_l_exec_groups_and_ties():
    assert np.nansum(labels.l_exec_group(np.arange(20.0))) == 7
    assert np.isnan(labels.l_exec_group(np.arange(19.0))).all()
    with_nan = np.append(np.arange(20.0), np.nan)
    out = labels.l_exec_group(with_nan)
    assert math.isnan(out[-1]) and np.nansum(out) == 7
    assert list(labels.average_ranks([1, 2, 2, 3])) == [1, 2.5, 2.5, 4]


def test_compute_labels_bad_price_is_nan():
    rng = np.random.default_rng(0)
    days = CAL.days[:45]
    rows = []
    for s in range(22):
        c = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
        for d, ci in zip(days, c):
            rows.append((f"S{s:02d}", d, ci * 0.999, ci * 1.01, ci * 0.99, ci))
    bars = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close"])
    bars.loc[(bars.symbol == "S00") & (bars.date == days[20]), "close"] = 0.0
    lab = compute = labels.compute_labels(bars)
    row = lab[(lab.symbol == "S00") & (lab.date == days[20])].iloc[0]
    assert math.isnan(row.z_exec) and math.isnan(row.l_exec)
    ok = lab[lab.date == days[20]]
    assert ok.l_exec.notna().sum() == 21 and ok.l_exec.sum() == 7  # ceil-free: pct>=0.7 of 21
    assert lab[lab.date == days[-5]].l_exec.isna().all()  # no t+11 bar


# ---------- features ----------
def _toy():
    rows = [("A", t, 0.1, 1.0) for t in range(10, 20)]          # baseline: 10 articles, mean 0.1
    rows += [("A", 66, -0.5, 1.0), ("A", 69, 1.0, 1.0), ("A", 69, 0.0, 0.5)]
    f = features.compute_features(pd.DataFrame(rows, columns=["symbol", "t", "x", "w"]), ["A", "B"], 70)
    return f.set_index(["symbol", "t"])


def test_feature_toys():
    f = _toy()
    a = f.loc[("A", 69)]
    assert a.S1 == pytest.approx(1.0 / 1.5)
    assert a.S5 == pytest.approx(0.5 / 2.5)
    assert a.NVS == pytest.approx(math.log(4) - math.log(1 + 5 * 10 / 60))
    assert a.ABN == pytest.approx(0.2 - 0.1)
    assert a.has_news == 1
    b = f.loc[("B", 69)]
    assert math.isnan(b.S1) and math.isnan(b.S5) and math.isnan(b.ABN)
    assert b.NVS == 0 and b.has_news == 0
    assert f.index.get_level_values("t").min() == 64  # warm-up excluded
    assert math.isnan(f.loc[("A", 65)].S5) and f.loc[("A", 66)].S1 == -0.5


def test_abn_needs_10_baseline_articles():
    rows = [("A", t, 0.1, 1.0) for t in range(10, 19)] + [("A", 69, 1.0, 1.0)]
    f = features.compute_features(pd.DataFrame(rows, columns=["symbol", "t", "x", "w"]), ["A"], 70)
    assert math.isnan(f[f.t == 69].ABN.iloc[0])


def test_frac_rank():
    out = features.frac_rank([0.3, np.nan, 0.1, 0.1, 0.5, 0.9])
    assert out == pytest.approx([0.5, 0.5, 0.2, 0.2, 0.7, 0.9])
    assert list(features.frac_rank([1, 2, 3, 4, np.nan])) == [0.5] * 5


def _pipeline(arts, syms):
    rows = []
    for a in arts:
        d, _ = assign(a["ts"], "US", CAL)
        if d is not None:
            rows.append((a["symbol"], CAL.index[d], a["x"], 1.0))
    df = pd.DataFrame(rows, columns=["symbol", "t", "x", "w"])
    return features.add_ranks(features.compute_features(df, syms, len(CAL.days)))


def test_future_deletion_invariance():
    rng = np.random.default_rng(3)
    syms = [f"S{i}" for i in range(8)]
    start = datetime(2023, 1, 1, tzinfo=UTC)
    arts = [{"symbol": syms[rng.integers(8)], "x": float(rng.normal()),
             "ts": start + timedelta(seconds=int(rng.integers(0, 364 * 86400)))} for _ in range(3000)]
    t = 150
    cut = cutoff_utc(CAL.days[t], "US")
    past = [a for a in arts if a["ts"] <= cut]
    assert 0 < len(past) < len(arts)
    full, trimmed = _pipeline(arts, syms), _pipeline(past, syms)
    pd.testing.assert_frame_equal(full[full.t <= t].reset_index(drop=True),
                                  trimmed[trimmed.t <= t].reset_index(drop=True))
    assert not full[full.t == t + 1].equals(trimmed[trimmed.t == t + 1])
