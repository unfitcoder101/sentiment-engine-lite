"""Generate synthetic data, run both worlds, write web/data.json. All data is SYNTHETIC."""
import json
import math
import time
from pathlib import Path

import numpy as np

from se import db, experiment, scorer, study, synthetic, timeutil

BANNER = ("Educational research prototype. Data origin: SYNTHETIC. "
          "No evidence of predictive value. Not investment advice.")

LEXICON_NOTE = ("Small hand-written demo word list. NOT Loughran-McDonald. Headlines are generated "
                "from the same words, so scoring is circular by construction.")


def _clean(v, nd=4):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def main():
    t0 = time.time()
    data = synthetic.generate()
    cal, symbols, arts, asg = data["cal"], data["symbols"], data["articles"], data["assignments"]
    feats = experiment.sentiment_features(asg, symbols, len(cal.days))
    results = {w: study.run_study(data, w) for w in ("null", "planted")}
    robust = study.run_robustness()
    feats_by = {s: study.scorer_feats(asg, symbols, len(cal.days), s) for s in ("mock_model", "demo_lexicon")}
    db_meta = {"banner": BANNER, "data_origin": "SYNTHETIC", "seed": data["seed"], "beta": data["beta"],
               "market": "US", "n_symbols": len(symbols), "n_days": len(cal.days), "n_articles": len(arts),
               "first_day": cal.days[0].isoformat(), "last_day": cal.days[-1].isoformat(), "lexicon_note": LEXICON_NOTE}
    db_path = db.write_demo(Path(__file__).parent / "var" / "demo.sqlite", data, feats_by, results, robust, db_meta)

    days = [d.isoformat() for d in cal.days]
    series = {}
    for s, g in feats.groupby("symbol"):
        g = g.sort_values("t")
        series[s] = {"S1": [_clean(v) for v in g["S1"]], "S5": [_clean(v) for v in g["S5"]],
                     "count": [int(v) for v in g["N1"]]}
    buckets = {s: {b: 0 for b in scorer.BUCKETS} for s in symbols}
    for s, b in zip(asg["symbol"], asg["bucket"]):
        buckets[s][b] += 1

    rng = np.random.default_rng(data["seed"])
    special = [a["id"] for a in arts if a["kind"] != "timestamp"]
    plain = [a["id"] for a in arts if a["kind"] == "timestamp"]
    pick = list(rng.choice(special, 120, replace=False)) + list(rng.choice(plain, 180, replace=False))
    sample = []
    for i in sorted(pick, key=lambda i: str(arts[i]["published"])):
        a = arts[i]
        sample.append({"headline": a["headline"], "symbols": a["symbols"], "kind": a["kind"],
                       "utc": a["published"], "local": timeutil.local_str(a["published"], "US") if a["precision"] != "date" else "",
                       "assigned": a["assigned"].isoformat() if a["assigned"] else None,
                       "reason": a["reason"] + (" (duplicate dropped)" if a["dropped_dup"] else ""),
                       "bucket": a["bucket"], "matched": a["matched"]})

    out = {
        "meta": {"banner": BANNER, "data_origin": "SYNTHETIC", "seed": data["seed"], "beta": data["beta"],
                 "market": "US", "n_symbols": len(symbols), "n_days": len(days), "n_articles": len(arts),
                 "n_assignments_kept": int(len(asg)), "first_day": days[0], "last_day": days[-1],
                 "lexicon_note": LEXICON_NOTE,
                 "probabilities": None},
        "symbols": symbols,
        "series_world": "planted",
        "series_dates": days[64:],
        "series": series,
        "buckets": buckets,
        "headlines": sample,
        "experiments": results,
        "robustness": robust,
        "cutoff_cases": timeutil.run_cutoff_cases(),
    }
    path = Path(__file__).parent / "web" / "data.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, allow_nan=False, separators=(",", ":")))
    for w, r in results.items():
        print(f"{w:8s} delta_ic={r['delta_ic']:+.4f} p={r['p_value']:.4f} "
              f"placebo_max={r['placebo_max']:+.4f} -> {r['verdict']}")
    print(f"robustness (10 seeds): null false-alarm rate {robust['null_false_alarm_rate']:.2f}, "
          f"planted detection rate {robust['planted_detection_rate']:.2f}")
    print(f"wrote {db_path}")
    print(f"wrote {path} ({path.stat().st_size // 1024} KB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
