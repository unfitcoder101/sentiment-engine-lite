import numpy as np
import pandas as pd
import pytest

from se import experiment, scorer, stats, synthetic


@pytest.fixture(scope="module")
def data():
    return synthetic.generate()


# ---------- scorer ----------
def test_scorer_buckets():
    s = scorer.score("SYN01 shares rally on strong demand", ["SYN01"])
    assert (s["P"], s["N"], s["tokens"], s["bucket"]) == (2, 0, 5, "positive")
    assert s["x"] == pytest.approx(0.4) and s["probabilities"] is None
    assert scorer.score("shares slump")["bucket"] == "negative"
    assert scorer.score("rally then slump")["bucket"] == "neutral (zero-net)"
    none = scorer.score("quarterly report update")
    assert none["bucket"] == "no lexicon coverage" and none["x"] == 0
    assert len(scorer.POS) == 20 and len(scorer.NEG) == 20 and not set(scorer.POS) & set(scorer.NEG)


# ---------- stats ----------
def test_bootstrap_reproducible():
    d = np.random.default_rng(1).normal(0.01, 0.05, 240)
    a, b = stats.paired_bootstrap(d, seed=5), stats.paired_bootstrap(d, seed=5)
    assert a == b
    assert stats.paired_bootstrap(d, seed=6)["ci_low"] != a["ci_low"]
    assert a["ci_low"] < a["delta"] < a["ci_high"]


def test_bootstrap_p_values():
    rng = np.random.default_rng(2)
    strong = stats.paired_bootstrap(rng.normal(0.05, 0.05, 240))
    assert strong["p"] == pytest.approx(1 / 2001)
    assert stats.paired_bootstrap(rng.normal(0.0, 0.05, 240) - 0.01)["p"] > 0.5


# ---------- synthetic ----------
def test_synthetic_worlds(data):
    null, planted = data["bars"]["null"], data["bars"]["planted"]
    assert len(data["symbols"]) == 30 and len(null) == 30 * len(data["cal"].days)
    for b in (null, planted):
        assert (b["low"] > 0).all()
        assert (b["high"] >= b[["open", "close"]].max(axis=1)).all()
        assert (b["low"] <= b[["open", "close"]].min(axis=1)).all()
    assert not np.allclose(null["close"], planted["close"])
    vol = null.groupby("symbol")["close"].apply(lambda c: c.pct_change().std())
    assert vol.between(0.008, 0.032).all()
    kinds = {a["kind"] for a in data["articles"]}
    assert kinds == {"timestamp", "date-only", "multi-ticker", "duplicate", "invalid"}
    reasons = {a["reason"] for a in data["articles"]}
    assert {"same_day", "after_cutoff", "weekend", "holiday", "date_only",
            "missing", "naive_timestamp", "after_as_of"} <= reasons
    assert any(a["dropped_dup"] for a in data["articles"])
    per_week = len(data["articles"]) / 30 / (1096 / 7)
    assert 0.5 < per_week < 3.2


def test_synthetic_reproducible_and_null_ignores_beta(data):
    again = synthetic.generate(beta=0.01)
    pd.testing.assert_frame_equal(again["bars"]["null"], data["bars"]["null"])
    assert [a["headline"] for a in again["articles"]] == [a["headline"] for a in data["articles"]]


# ---------- experiment ----------
def test_split_purge_and_shuffle(data):
    cal = data["cal"]
    feats = experiment.sentiment_features(data["assignments"], data["symbols"], len(cal.days))
    panel = experiment.build_panel(data["bars"]["null"], feats, cal)
    train, test = experiment.split(panel, cal)
    first_test = int(test["t"].min())
    assert cal.days[first_test].year == 2022 and cal.days[first_test - 1].year == 2021
    assert train["t"].max() + 11 < first_test - 5
    assert panel["t"].min() >= 64
    sh = experiment.shuffle_block(panel, 1)
    pd.testing.assert_frame_equal(sh[experiment.BASE_COLS + ["z_exec"]], panel[experiment.BASE_COLS + ["z_exec"]])
    assert not sh[experiment.SENT_COLS].equals(panel[experiment.SENT_COLS])
    assert sh.groupby("t")["S5_r"].sum().to_numpy() == pytest.approx(panel.groupby("t")["S5_r"].sum().to_numpy())


def test_daily_ic_and_verdicts():
    t = np.repeat([0, 1, 2], [20, 20, 19])
    z = np.arange(59.0)
    ic = experiment.daily_ic(t, z, z)
    assert list(ic.index) == [0, 1] and ic.to_numpy() == pytest.approx([1, 1])
    assert experiment.daily_ic(t, -z, z).to_numpy() == pytest.approx([-1, -1])
    pre = "SYNTHETIC SOFTWARE CHECK: "
    assert experiment.verdict("planted", 0.001, True) == pre + "planted effect detected"
    assert experiment.verdict("planted", 0.30, True) == pre + "not detected"
    assert experiment.verdict("planted", 0.001, False) == pre + "placebo not excluded"
    assert experiment.verdict("null", 0.30, False) == pre + "no effect flagged"
