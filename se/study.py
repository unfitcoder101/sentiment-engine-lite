"""Frozen-sample experiment framework (extends se/experiment.py). Synthetic data only.

Order of work in one run: build PANEL from prices and labels -> freeze key lists and hashes ->
fit every arm inside each fold -> five abort-on-failure checks -> statistics -> one verdict text.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as config_mod
from . import gates as gates_mod
from . import metrics, stats
from .config import PreregMismatchError
from .experiment import BASE_COLS, daily_ic
from .features import WARMUP, add_ranks, compute_features
from .labels import H, compute_labels
from .verdicts import MODES, verdict_text

SENT = ["S1_r", "S5_r", "NVS_r", "ABN_r", "has_news"]
ARM_LABELS = {
    "A_lowvol": "Low-volatility rule (score = -ATR%)", "A_highvol": "High-volatility rule (score = +ATR%)",
    "B": "Baseline price features", "C": "B + sentiment (MOCK MODEL)", "C_LM": "B + sentiment (demo lexicon)",
    "C1": "B + news volume only", "C3": "Placebo: sentiment from 20 days earlier", "D": "Sentiment only",
}
PRIMARY = [("C", "B"), ("C", "C1"), ("C", "C_LM")]
EXPLORATORY = [("C3", "B"), ("D", "B"), ("C_LM", "B"), ("C1", "B"), ("A_lowvol", "B"), ("A_highvol", "B")]


class FrozenSampleError(RuntimeError):
    pass


def sha_keys(keys):
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def train_allowed(t, first_test_t, embargo=5, h=H):
    """Purge + embargo: the label window (ends at t+h+1) must finish `embargo` bars before the first test bar."""
    return np.asarray(t) + h + 1 < first_test_t - embargo


def build_base_panel(bars, cal):
    """PANEL: membership depends only on prices and labels."""
    lab = compute_labels(bars)
    lab["t"] = lab["date"].map(cal.index)
    by = lab.groupby("symbol")["close"]
    lab["ret_5"], lab["ret_20"] = by.pct_change(5), by.pct_change(20)
    lab["vol_10"] = by.pct_change().groupby(lab["symbol"]).transform(lambda s: s.rolling(10).std())
    p = lab[lab["t"] >= WARMUP].dropna(subset=BASE_COLS + ["r_exec", "z_exec", "l_exec"])
    p = p.sort_values(["t", "symbol"]).reset_index(drop=True)
    p["key"] = [f"{s}|{d.isoformat()}" for s, d in zip(p["symbol"], p["date"])]
    p["year"] = [d.year for d in p["date"]]
    return p


def scorer_feats(assignments, symbols, n_days, scorer, unit_weights=False):
    a = assignments
    if scorer == "demo_lexicon":
        a = a[a["bucket"] != "no lexicon coverage"]      # no score exists for these
        x = a["x"]
    elif scorer == "mock_model":
        x = a["x_mock"]
    else:
        raise ValueError(f"unknown scorer {scorer!r}")
    df = pd.DataFrame({"symbol": a["symbol"].to_numpy(), "t": a["t"].to_numpy(), "x": x.to_numpy(),
                       "w": 1.0 if unit_weights else a["w"].to_numpy()})
    return add_ranks(compute_features(df, symbols, n_days))


def sentiment_block(panel, feats, lag=0):
    """Rows aligned to the panel. lag > 0 takes the block from `lag` trading days earlier."""
    f = feats[["symbol", "t"] + SENT]
    if lag:
        f = f.assign(t=f["t"] + lag)
    x = panel[["symbol", "t"]].merge(f, on=["symbol", "t"], how="left")[SENT].to_numpy(float)
    miss = np.isnan(x[:, 0])
    x[miss, :4], x[miss, 4] = 0.5, 0.0
    return x


def shuffle_within_date(x, t, seed):
    rng = np.random.default_rng(seed)
    out = x.copy()
    b = np.flatnonzero(np.r_[True, t[1:] != t[:-1], True])
    for i, j in zip(b[:-1], b[1:]):
        out[i:j] = x[i:j][rng.permutation(j - i)]
    return out


def make_folds(panel, cal, test_years, embargo):
    t, year, folds = panel["t"].to_numpy(), panel["year"].to_numpy(), []
    for y in test_years:
        first = next(i for i, d in enumerate(cal.days) if d.year == y)
        folds.append({"year": y, "first_test_t": first, "train_idx": np.flatnonzero(train_allowed(t, first, embargo)),
                      "test_idx": np.flatnonzero(year == y)})
    return folds


def prepare(data, world, cfg):
    cal, symbols, asg = data["cal"], data["symbols"], data["assignments"]
    n_days = len(cal.days)
    panel = build_base_panel(data["bars"][world], cal)
    f_primary = scorer_feats(asg, symbols, n_days, cfg["scorers"]["primary"])
    blocks = {"primary": sentiment_block(panel, f_primary),
              "lexicon": sentiment_block(panel, scorer_feats(asg, symbols, n_days, cfg["scorers"]["lexicon_baseline"])),
              "lagged": sentiment_block(panel, f_primary, lag=cfg["placebo"]["c3_lag"])}
    panel["in_S2"] = blocks["primary"][:, 4] == 1      # frozen column: original, unshuffled, unlagged has_news
    folds = make_folds(panel, cal, cfg["folds"]["test_years"], cfg["folds"]["embargo"])
    return {"panel": panel, "blocks": blocks, "folds": folds, "cal": cal, "data": data, "world": world, "cfg": cfg}


def freeze_samples(ctx, out_dir=None):
    """Compute the key lists ONCE (before any placebo), hash them, optionally write them."""
    panel, keys = ctx["panel"], ctx["panel"]["key"].to_numpy()
    p1 = np.sort(np.concatenate([f["test_idx"] for f in ctx["folds"]]))
    lists = {"PANEL": keys, "P1": keys[p1], "S2": keys[p1][panel["in_S2"].to_numpy()[p1]]}
    for f in ctx["folds"]:
        lists[f"TRAIN_{f['year']}"] = keys[f["train_idx"]]
    frozen = {"hashes": {k: sha_keys(v) for k, v in lists.items()}, "counts": {k: int(len(v)) for k, v in lists.items()},
              "files": {}}
    if out_dir is not None:
        d = Path(out_dir) / "samples"
        d.mkdir(parents=True, exist_ok=True)
        for k, v in lists.items():
            (d / f"{k}.txt").write_text("\n".join(v) + "\n")
            frozen["files"][k] = f"samples/{k}.txt"
    return frozen


def _check(ok, message, counter):
    counter["checked"] += 1
    if not ok:
        raise FrozenSampleError(message)


def _fit_predict(x_tr, y_tr, x_te, c=1.0):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=1000))
    model.fit(x_tr, y_tr.astype(int))
    return model.decision_function(x_te)       # a ranking score, not a probability


def run_arm(name, x, ctx, frozen, counter, direct_score=None, s2_mask=None):
    """Fit inside each fold, score P1, and enforce checks 1-3 against the frozen hashes."""
    panel, keys = ctx["panel"], ctx["panel"]["key"].to_numpy()
    y, score = panel["l_exec"].to_numpy(), np.full(len(panel), np.nan)
    for f in ctx["folds"]:
        if direct_score is None:
            tr = f["train_idx"]
            _check(sha_keys(keys[tr]) == frozen["hashes"][f"TRAIN_{f['year']}"],
                   f"{name}: training keys differ from the frozen fold {f['year']} keys", counter)
            score[f["test_idx"]] = _fit_predict(x[tr], y[tr], x[f["test_idx"]], ctx["cfg"]["model"]["C"])
        else:
            score[f["test_idx"]] = direct_score[f["test_idx"]]
    ev = np.flatnonzero(~np.isnan(score))
    _check(sha_keys(keys[ev]) == frozen["hashes"]["P1"], f"{name}: evaluated keys differ from frozen P1", counter)
    mask = panel["in_S2"].to_numpy() if s2_mask is None else s2_mask
    s2 = ev[mask[ev]]
    _check(sha_keys(keys[s2]) == frozen["hashes"]["S2"], f"{name}: S2 keys differ from frozen S2", counter)
    ic = daily_ic(panel["t"].to_numpy()[ev], score[ev], panel["z_exec"].to_numpy()[ev])
    return {"name": name, "score": score, "eval_idx": ev, "s2_idx": s2, "ic": ic, "n_rows": int(len(ev))}


def check_across(arms, counter):
    """Checks 4-5: identical daily-IC date lists and identical row counts across arms."""
    ref = arms[0]
    for a in arms[1:]:
        _check(list(a["ic"].index) == list(ref["ic"].index),
               f"{a['name']}: daily-IC dates differ from {ref['name']}", counter)
        _check(a["n_rows"] == ref["n_rows"], f"{a['name']}: row count differs from {ref['name']}", counter)


def _secondary(ctx, arm, which):
    panel, m = ctx["panel"], ctx["cfg"]["metrics"]
    idx = arm["eval_idx"] if which == "P1" else arm["s2_idx"]
    return metrics.daily_secondary(
        panel["t"].to_numpy()[idx], arm["score"][idx], panel["z_exec"].to_numpy()[idx], panel["r_exec"].to_numpy()[idx],
        panel["l_exec"].to_numpy()[idx], m["min_stocks_p1"] if which == "P1" else m["min_stocks_s2"],
        m["top_k_share"], m["cost"])


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run_study(data, world, mode="synthetic_validation", cfg=None, root=None, write=True, light=False,
              data_key="data:synthetic", scorer_key=None):
    """One world, one mode. `light` skips ablations, sensitivity and most secondary metrics (robustness runs)."""
    root = Path(root or config_mod.ROOT)
    cfg = cfg or config_mod.load()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    config_mod.check_against_code(cfg)
    prereg = config_mod.prereg_status(root)
    run_id = f"{mode}-seed{data['seed']}-{world}"
    out_dir = root / "runs" / run_id
    if mode == "confirmatory":      # always refused under the real gates.yaml
        gates_mod.check_confirmatory(gates_mod.load(root / "gates.yaml"), data_key,
                                     scorer_key or "scorer:" + cfg["scorers"]["primary"], prereg["match"],
                                     (out_dir / "samples" / "P1.txt").exists())
        raise NotImplementedError("no confirmatory pipeline exists in this prototype")
    if mode == "synthetic_validation" and not prereg["match"]:
        raise PreregMismatchError("synthetic_validation run aborted: " + prereg["reason"])

    st, pl = cfg["statistics"], cfg["placebo"]
    ctx = prepare(data, world, cfg)
    frozen = freeze_samples(ctx, out_dir if write else None)       # before any placebo
    panel, counter = ctx["panel"], {"checked": 0}
    base, prim, t = panel[BASE_COLS].to_numpy(float), ctx["blocks"]["primary"], panel["t"].to_numpy()
    atr = panel["atr_pct"].to_numpy()
    designs = {"B": base, "C": np.hstack([base, prim]), "C_LM": np.hstack([base, ctx["blocks"]["lexicon"]]),
               "C1": np.hstack([base, prim[:, [2, 4]]]), "C3": np.hstack([base, ctx["blocks"]["lagged"]]), "D": prim}
    arms = {"A_lowvol": run_arm("A_lowvol", None, ctx, frozen, counter, direct_score=-atr),
            "A_highvol": run_arm("A_highvol", None, ctx, frozen, counter, direct_score=atr)}
    for name, x in designs.items():
        arms[name] = run_arm(name, x, ctx, frozen, counter)
    c2_seeds = [pl["c2_seed_start"] + i for i in range(pl["c2_n"])]
    c2 = [run_arm(f"C2_{s}", np.hstack([base, shuffle_within_date(prim, t, s)]), ctx, frozen, counter) for s in c2_seeds]
    abl = {}
    if not light:
        for i, f in enumerate(["S1", "S5", "NVS", "ABN"]):
            abl[f"C_no_{f}"] = run_arm(f"C_no_{f}", np.hstack([base, np.delete(prim, i, axis=1)]), ctx, frozen, counter)
        d = ctx["data"]
        w1 = sentiment_block(panel, scorer_feats(d["assignments"], d["symbols"], len(ctx["cal"].days),
                                                 cfg["scorers"]["primary"], unit_weights=True))
        abl["C_weights_1"] = run_arm("C_weights_1", np.hstack([base, w1]), ctx, frozen, counter)
    check_across(list(arms.values()) + c2 + list(abl.values()), counter)

    def cmp(a, b, block=None):
        return stats.compare((a["ic"] - b["ic"]).to_numpy(), block or st["mean_block"], st["n_resamples"], st["seed"])

    prim_cmp = [dict(cmp(arms[a], arms[b]), name=f"{a} vs {b}", family="primary (Holm, 3 comparisons)") for a, b in PRIMARY]
    for c, adj in zip(prim_cmp, stats.holm([c["p"] for c in prim_cmp])):
        c["p_holm"] = adj
    expl_cmp = [dict(cmp(arms[a], arms[b]), name=f"{a} vs {b}", family="exploratory") for a, b in EXPLORATORY]
    cb, c3b = prim_cmp[0], expl_cmp[0]
    c2_deltas = [float((a["ic"] - arms["B"]["ic"]).mean()) for a in c2]
    c2_pass = bool(cb["delta"] >= max(c2_deltas) and float(np.mean(c2_deltas)) < pl["c2_mean_max"])
    c3_pass = bool(c3b["delta"] <= pl["c3_ratio_max"] * cb["delta"] and c3b["p"] > pl["c3_p_min"])

    sec_names = ["B", "C"] if light else list(arms)
    arm_out = {}
    for name in arms:
        a = arms[name]
        o = {"label": ARM_LABELS[name], "ic": float(a["ic"].mean()), "n_rows": a["n_rows"], "n_dates": int(len(a["ic"]))}
        if name in sec_names:
            o["P1"], o["S2"] = _secondary(ctx, a, "P1"), _secondary(ctx, a, "S2")
            years = np.array([ctx["cal"].days[int(i)].year for i in a["ic"].index])
            o["by_year"] = {int(y): float(a["ic"].to_numpy()[years == y].mean()) for y in sorted(set(years))}
        arm_out[name] = o

    alpha = st["alpha"]
    net_c, net_b = arm_out["C"]["P1"]["net_topk"], arm_out["B"]["P1"]["net_topk"]
    criteria = {
        "1_holm_p_C_vs_B": bool(cb["p_holm"] <= alpha),
        "2_min_delta_ic": bool(cb["delta"] >= cfg["success"]["min_delta_ic"]),
        "3_C_beats_C1_and_C_LM": bool(all(c["p_holm"] <= alpha and c["delta"] > 0 for c in prim_cmp[1:])),
        "4_both_placebos_passed": bool(c2_pass and c3_pass),
        "5_net_topk_C_not_lower_than_B": bool(net_c >= net_b),
    }
    detected = criteria["1_holm_p_C_vs_B"] and criteria["2_min_delta_ic"]
    placebos, met = criteria["4_both_placebos_passed"], all(criteria.values())
    result = {
        "run_id": run_id, "mode": mode, "world": world, "data_origin": "SYNTHETIC", "seed": data["seed"],
        "beta": data["beta"] if world == "planted" else 0.0, "primary_scorer": cfg["scorers"]["primary"],
        "lexicon_baseline": cfg["scorers"]["lexicon_baseline"], "prereg": prereg,
        "samples": frozen, "assertions": {"checked": counter["checked"], "failed": 0,
                                         "arms_checked": len(arms) + len(c2) + len(abl)},
        "folds": [{"year": f["year"], "first_test_t": f["first_test_t"], "n_train": int(len(f["train_idx"])),
                   "n_test": int(len(f["test_idx"])), "max_train_t": int(t[f["train_idx"]].max())} for f in ctx["folds"]],
        "arms": arm_out, "comparisons": {"primary": prim_cmp, "exploratory": expl_cmp},
        "placebo": {"C2": {"seeds": c2_seeds, "deltas": c2_deltas, "max": max(c2_deltas), "mean": float(np.mean(c2_deltas)),
                           "observed": cb["delta"], "passed": c2_pass},
                    "C3": {"delta": c3b["delta"], "p": c3b["p"], "limit": pl["c3_ratio_max"] * cb["delta"], "passed": c3_pass}},
        "criteria": criteria, "detected": bool(detected), "placebos_passed": bool(placebos), "criteria_met": bool(met),
        "verdict": verdict_text(mode, world, detected, placebos, met),
        "power": {"bootstrap_se": cb["se"], "mde": stats.mde(cb["se"], alpha, len(PRIMARY), cfg["power"]["target_power_z"]),
                  "note": "Approximation: SE * (z(1 - 0.05/3) + 0.8416). Not an exact power calculation."},
        # flat keys kept for the Stage 3 dashboard
        "ic_b": arm_out["B"]["ic"], "ic_c": arm_out["C"]["ic"], "delta_ic": cb["delta"], "p_value": cb["p"],
        "ci_low": cb["ci_low"], "ci_high": cb["ci_high"], "placebo_deltas": c2_deltas, "placebo_max": max(c2_deltas),
        "placebo_pass": c2_pass, "n_test_days": cb["n"], "n_test_rows": arms["B"]["n_rows"],
        "n_train_rows": int(len(ctx["folds"][-1]["train_idx"])), "alpha": alpha,
        "bootstrap": {k: st[k] for k in ("mean_block", "n_resamples", "seed")},
    }
    if not light:
        result["sensitivity"] = {f"block_{b}": {k: v for k, v in cmp(arms["C"], arms["B"], b).items()
                                                if k in ("delta", "p", "ci_low", "ci_high", "se")}
                                 for b in st["sensitivity_blocks"]}
        result["sensitivity"]["note"] = "Reported only; not used in any decision."
        result["ablations"] = [dict(name=n, ic=float(a["ic"].mean()), family="exploratory",
                                    delta_vs_B=float((a["ic"] - arms["B"]["ic"]).mean()),
                                    **{k: v for k, v in cmp(a, arms["C"]).items() if k in ("delta", "ci_low", "ci_high")})
                               for n, a in abl.items()]
    result = _jsonable(result)
    if write:
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, allow_nan=False))
    return result


def run_robustness(cfg=None, seeds=None):
    """Both worlds for the pre-declared seeds. Rates use: flagged (criteria 1-2) with both placebos passed."""
    from . import synthetic
    cfg = cfg or config_mod.load()
    rows = []
    for s in (seeds or cfg["robustness_seeds"]):
        data = synthetic.generate(seed=s, beta=cfg["data"]["beta"])
        for world in ("null", "planted"):
            r = run_study(data, world, cfg=cfg, write=False, light=True)
            c = r["comparisons"]["primary"][0]
            rows.append({"seed": s, "world": world, "delta_ic": c["delta"], "p": c["p"], "p_holm": c["p_holm"],
                         "detected": r["detected"], "placebos_passed": r["placebos_passed"],
                         "criteria_met": r["criteria_met"], "flagged": r["detected"] and r["placebos_passed"],
                         "verdict": r["verdict"]})

    def rate(world, key):
        sel = [x for x in rows if x["world"] == world]
        return sum(bool(x[key]) for x in sel) / len(sel)

    return {"seeds": list(seeds or cfg["robustness_seeds"]), "rows": rows,
            "null_false_alarm_rate": rate("null", "flagged"), "planted_detection_rate": rate("planted", "flagged"),
            "null_raw_flag_rate": rate("null", "detected"), "planted_criteria_met_rate": rate("planted", "criteria_met"),
            "definition": "flagged = Holm-adjusted p of C vs B <= 0.05 and delta IC >= 0.01, with both placebos passed. "
                          "Ten seeds give only a coarse rate (steps of 0.1)."}
