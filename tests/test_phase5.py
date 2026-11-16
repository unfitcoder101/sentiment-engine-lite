import copy
import shutil

import numpy as np
import pytest

from se import config, gates, metrics, scorer, stats, study, synthetic
from se.verdicts import find_forbidden, verdict_text


@pytest.fixture(scope="module")
def data():
    return synthetic.generate()


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def ctx(data, cfg):
    c = study.prepare(data, "null", cfg)
    return c, study.freeze_samples(c)


# ---------- mock model ----------
def test_mock_model_scorer():
    a, b = scorer.mock_model_score("SYN01 shares rally on strong demand", ["SYN01"]), None
    b = scorer.mock_model_score("SYN01 shares rally on strong demand", ["SYN01"])
    assert a == b and a["scorer_label"] == "MOCK MODEL (not FinBERT)"
    s = a["output_scores"]
    assert sum(s.values()) == pytest.approx(1.0) and a["x"] == pytest.approx(s["positive"] - s["negative"])
    assert a["label"] == max(s, key=s.get) and a["label"] in ("positive", "negative", "neutral")
    assert "price move" in a["note"] and a != scorer.mock_model_score("SYN01 shares slump", ["SYN01"])
    e = np.array([scorer._hash_normals(f"h{i}") for i in range(4000)]).ravel()
    assert abs(e.mean()) < 0.05 and e.std() == pytest.approx(1.0, abs=0.05)


# ---------- metrics and statistics (hand cases) ----------
def test_spearman_auc_hand_cases():
    assert metrics.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert list(metrics.avg_rank([1, 2, 2, 3])) == [1, 2.5, 2.5, 4]
    assert metrics.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert metrics.auc([1, 1], [0, 1]) == pytest.approx(0.5)      # tie counts half


def test_topk_precision_and_net_return_hand_case():
    score = np.arange(10.0)                       # best = last row
    r = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10])
    l = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1.0])
    out = metrics.daily_secondary(np.zeros(10), score, r, r, l, 10, 0.10, 0.004)
    assert out["precision_at_k"] == 1.0 and out["n_dates"] == 1          # K = 1 of 10
    assert out["net_topk"] == pytest.approx(0.10 - 0.045 - 0.004)
    assert out["ic_r"] == pytest.approx(1.0) and out["auc"] == pytest.approx(1.0)
    assert len(metrics.top_k(np.arange(30.0), 0.10)) == 3


def test_holm_example_and_mde():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.6, 0.7]) == pytest.approx([1.0, 1.0, 1.0])
    assert stats.mde(0.01) == pytest.approx(0.01 * (2.12805 + 0.8416), abs=1e-6)


def test_compare_matches_paired_bootstrap():
    d = np.random.default_rng(4).normal(0.01, 0.05, 300)
    a, b = stats.compare(d), stats.paired_bootstrap(d)
    assert (a["delta"], a["p"], a["ci_low"], a["ci_high"]) == (b["delta"], b["p"], b["ci_low"], b["ci_high"])
    assert a["se"] > 0 and a["interval"] == "unadjusted"


# ---------- purge / folds ----------
def test_purge_hand_calculation():
    # first test bar 100, embargo 5, label window ends at t+11: need t + 11 < 95, so the last allowed t is 83
    t = np.arange(70, 100)
    assert t[study.train_allowed(t, 100, 5)].max() == 83
    assert not study.train_allowed(84, 100, 5) and study.train_allowed(83, 100, 5)


def test_folds_are_expanding_and_purged(ctx):
    c, _ = ctx
    f21, f22 = c["folds"]
    t, year = c["panel"]["t"].to_numpy(), c["panel"]["year"].to_numpy()
    assert (f21["year"], f22["year"]) == (2021, 2022) and set(f21["train_idx"]) < set(f22["train_idx"])
    for f in (f21, f22):
        assert t[f["train_idx"]].max() + 11 < f["first_test_t"] - 5
        assert c["cal"].days[f["first_test_t"]].year == f["year"] and c["cal"].days[f["first_test_t"] - 1].year == f["year"] - 1
        assert set(year[f["test_idx"]]) == {f["year"]} and t[f["train_idx"]].min() == 64


# ---------- frozen samples and the five abort checks ----------
def test_frozen_samples_definition(ctx):
    c, frozen = ctx
    p = c["panel"]
    assert p[study.BASE_COLS + ["r_exec", "z_exec", "l_exec"]].notna().all().all() and p["t"].min() >= 64
    assert frozen["counts"]["P1"] == int(p["year"].isin([2021, 2022]).sum())
    assert 0 < frozen["counts"]["S2"] < frozen["counts"]["P1"] < frozen["counts"]["PANEL"]
    assert len(set(frozen["hashes"].values())) == 5 and all(len(h) == 64 for h in frozen["hashes"].values())
    assert study.freeze_samples(study.prepare(c["data"], "null", c["cfg"]))["hashes"] == frozen["hashes"]


def test_check1_training_keys(ctx):
    c, frozen = ctx
    bad = dict(c, folds=[dict(c["folds"][0], train_idx=c["folds"][0]["train_idx"][:-1]), c["folds"][1]])
    x = c["panel"][study.BASE_COLS].to_numpy(float)
    with pytest.raises(study.FrozenSampleError, match="training keys"):
        study.run_arm("B", x, bad, frozen, {"checked": 0})


def test_check2_evaluated_keys(ctx):
    c, frozen = ctx
    bad = dict(c, folds=[c["folds"][0], dict(c["folds"][1], test_idx=c["folds"][1]["test_idx"][:-1])])
    with pytest.raises(study.FrozenSampleError, match="frozen P1"):
        study.run_arm("A", None, bad, frozen, {"checked": 0}, direct_score=c["panel"]["atr_pct"].to_numpy())


def test_check3_s2_keys_cannot_be_redefined(ctx):
    c, frozen = ctx
    atr = c["panel"]["atr_pct"].to_numpy()
    redefined = study.shuffle_within_date(c["blocks"]["primary"], c["panel"]["t"].to_numpy(), 1)[:, 4] == 1
    with pytest.raises(study.FrozenSampleError, match="frozen S2"):
        study.run_arm("A", None, c, frozen, {"checked": 0}, direct_score=atr, s2_mask=redefined)


def test_checks_4_and_5_across_arms(ctx):
    c, frozen = ctx
    counter = {"checked": 0}
    atr = c["panel"]["atr_pct"].to_numpy()
    a = study.run_arm("A_highvol", None, c, frozen, counter, direct_score=atr)
    b = study.run_arm("A_lowvol", None, c, frozen, counter, direct_score=-atr)
    study.check_across([a, b], counter)
    assert counter["checked"] == 6
    assert a["ic"].to_numpy() == pytest.approx(-b["ic"].to_numpy())
    with pytest.raises(study.FrozenSampleError, match="dates differ"):
        study.check_across([a, dict(b, ic=b["ic"].iloc[1:])], counter)
    with pytest.raises(study.FrozenSampleError, match="row count"):
        study.check_across([a, dict(b, n_rows=b["n_rows"] - 1)], counter)


def test_lagged_block_and_shuffle(ctx):
    c, _ = ctx
    p, prim, lag = c["panel"], c["blocks"]["primary"], c["blocks"]["lagged"]
    first = p["t"].to_numpy() < 64 + 20
    assert (lag[first, :4] == 0.5).all() and (lag[first, 4] == 0).all()
    i = int(np.flatnonzero((p["symbol"] == "SYN05").to_numpy() & (p["t"] == 200).to_numpy())[0])
    j = int(np.flatnonzero((p["symbol"] == "SYN05").to_numpy() & (p["t"] == 180).to_numpy())[0])
    assert list(lag[i]) == list(prim[j])
    sh = study.shuffle_within_date(prim, p["t"].to_numpy(), 1000)
    day = (p["t"] == 300).to_numpy()
    assert not np.array_equal(sh, prim) and sorted(map(tuple, sh[day])) == sorted(map(tuple, prim[day]))


# ---------- full runs ----------
@pytest.fixture(scope="module")
def runs(data):
    return {w: study.run_study(data, w, write=False, light=True) for w in ("null", "planted")}


def test_run_is_reproducible_and_complete(data, runs):
    again = study.run_study(data, "planted", write=False, light=True)
    r = runs["planted"]
    assert again["samples"]["hashes"] == r["samples"]["hashes"] and again["verdict"] == r["verdict"]
    assert again["comparisons"] == r["comparisons"]
    assert r["assertions"]["failed"] == 0 and r["assertions"]["arms_checked"] == 28
    assert r["assertions"]["checked"] == 2 * 28 + 2 * 26 + 2 * 27
    assert [c["name"] for c in r["comparisons"]["primary"]] == ["C vs B", "C vs C1", "C vs C_LM"]
    assert all(c["p_holm"] >= c["p"] for c in r["comparisons"]["primary"])
    assert r["arms"]["C"]["P1"]["ic_z"] == pytest.approx(r["arms"]["C"]["ic"])
    assert set(r["arms"]["C"]["by_year"]) == {"2021", "2022"} or set(r["arms"]["C"]["by_year"]) == {2021, 2022}
    assert r["arms"]["A_lowvol"]["ic"] == pytest.approx(-r["arms"]["A_highvol"]["ic"])
    assert r["prereg"]["match"] is True
    for w in runs:
        assert runs[w]["verdict"].startswith("SYNTHETIC SOFTWARE CHECK: ") and not find_forbidden(runs[w]["verdict"])
        assert runs[w]["samples"]["hashes"]["S2"] == r["samples"]["hashes"]["S2"]      # same news in both worlds


def test_verdict_follows_criteria(runs):
    for w, r in runs.items():
        assert r["detected"] == (r["criteria"]["1_holm_p_C_vs_B"] and r["criteria"]["2_min_delta_ic"])
        assert r["verdict"] == verdict_text("synthetic_validation", w, r["detected"], r["placebos_passed"], r["criteria_met"])
        c2 = r["placebo"]["C2"]
        assert c2["passed"] == (c2["observed"] >= c2["max"] and c2["mean"] < 0.005) and len(c2["deltas"]) == 20


def test_verdict_texts_and_forbidden_words():
    pre, texts = "SYNTHETIC SOFTWARE CHECK: ", set()
    assert verdict_text("synthetic_validation", "planted", True, True, True) == pre + "criteria met"
    assert verdict_text("synthetic_validation", "planted", True, True, False) == pre + "planted effect detected"
    assert verdict_text("synthetic_validation", "planted", False, True, False) == pre + "not detected"
    assert verdict_text("synthetic_validation", "planted", True, False, False) == pre + "placebo not excluded"
    assert verdict_text("synthetic_validation", "null", False, False, False) == pre + "no effect flagged"
    assert verdict_text("exploratory", "planted", True, True, True) == "EXPLORATORY: criteria met"
    assert verdict_text("exploratory", "null", False, False, False) == "EXPLORATORY: criteria not met"
    for mode in ("synthetic_validation", "exploratory"):
        for world in ("null", "planted"):
            for flags in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
                texts.add(verdict_text(mode, world, *map(bool, flags)))
    assert all(not find_forbidden(t) for t in texts)
    assert find_forbidden("a Significant edge that predicts profit, validated") == \
        ["edge", "predicts", "profit", "significant", "validated"]
    assert find_forbidden("knowledge of predictive value") == []
    with pytest.raises(ValueError):
        verdict_text("confirmatory", "null", True, True, True)


# ---------- pre-registration and run modes ----------
def _tmp_root(tmp_path):
    for rel in (config.CONFIG_REL, config.PREREG_DOC_REL, config.PREREG_REL, "gates.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(config.ROOT / rel, tmp_path / rel)
    return tmp_path


def test_prereg_mismatch_aborts_synthetic_validation(data, tmp_path):
    root = _tmp_root(tmp_path)
    assert config.prereg_status(root)["match"] and config.prereg_status()["match"]
    with open(root / config.CONFIG_REL, "a") as fh:
        fh.write("# edited after freezing\n")
    st = config.prereg_status(root)
    assert not st["match"] and "config_sha256" in st["reason"]
    with pytest.raises(config.PreregMismatchError):
        study.run_study(data, "null", root=root, write=False, light=True)
    r = study.run_study(data, "null", mode="exploratory", root=root, write=False, light=True)
    assert r["prereg"]["match"] is False and r["verdict"].startswith("EXPLORATORY: criteria ")
    (root / config.PREREG_REL).unlink()
    assert "missing" in config.prereg_status(root)["reason"]
    assert config.prereg_freeze(root)["config_sha256"] == config.prereg_status(root)["config_sha256"]
    assert config.prereg_status(root)["match"]


def test_config_must_agree_with_code(cfg):
    config.check_against_code(cfg)
    bad = copy.deepcopy(cfg)
    bad["labels"]["horizon"] = 5
    with pytest.raises(config.ConfigError, match="labels.horizon"):
        config.check_against_code(bad)
    assert cfg["robustness_seeds"] == list(range(1, 11)) and cfg["data"]["beta"] == 0.002


def test_confirmatory_always_refused_under_real_gates(data):
    with pytest.raises(gates.ConfirmatoryGateError) as e:
        study.run_study(data, "planted", mode="confirmatory", write=False)
    assert len(e.value.reasons) >= 3 and any("study:confirmatory" in r for r in e.value.reasons)
    real = gates.load()
    for dk in [k for k in real["gates"] if k.startswith("data:")]:
        for sk in [k for k in real["gates"] if k.startswith("scorer:")]:
            reasons = gates.confirmatory_reasons(real, dk, sk, True, True)
            assert any(sk in r for r in reasons) and any("study:confirmatory" in r for r in reasons)
    assert len(gates.confirmatory_reasons(real, "data:fnspid", "scorer:finbert_prosus", False, False)) == 5


def test_confirmatory_guard_with_synthetic_all_clear_entries():
    clear = {"gates": {"data:x": {"licence": "verified_permitted"}, "scorer:y": {"contamination": "verified_clear"},
                       "study:confirmatory": {"status": "allowed"}}}
    assert gates.check_confirmatory(clear, "data:x", "scorer:y", True, True) is True
    for kwargs, word in [(dict(prereg_match=False, frozen_samples_exist=True), "pre-registration"),
                         (dict(prereg_match=True, frozen_samples_exist=False), "frozen samples")]:
        with pytest.raises(gates.ConfirmatoryGateError, match=word):
            gates.check_confirmatory(clear, "data:x", "scorer:y", **kwargs)
    worse = copy.deepcopy(clear)
    worse["gates"]["scorer:y"]["contamination"] = "unresolved"
    with pytest.raises(gates.ConfirmatoryGateError, match="contamination"):
        gates.check_confirmatory(worse, "data:x", "scorer:y", True, True)
