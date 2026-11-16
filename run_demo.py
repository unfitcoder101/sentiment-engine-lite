"""Generate synthetic data, run both worlds and the 10-seed robustness check, write var/demo.sqlite and web/data.json.

All data is SYNTHETIC. web/data.json is the dashboard's static fallback: the same payloads the API serves.
"""
import json
import time
from pathlib import Path

from se import api, db, study, synthetic

ROOT = Path(__file__).parent
BANNER = api.BANNER
LEXICON_NOTE = ("Small hand-written demo word list. NOT Loughran-McDonald. Headlines are generated "
                "from the same words, so scoring is circular by construction.")


def main():
    t0 = time.time()
    data = synthetic.generate()
    cal, symbols, arts, asg = data["cal"], data["symbols"], data["articles"], data["assignments"]
    results = {w: study.run_study(data, w) for w in ("null", "planted")}
    robust = study.run_robustness()
    feats_by = {s: study.scorer_feats(asg, symbols, len(cal.days), s) for s in ("mock_model", "demo_lexicon")}
    meta = {"banner": BANNER, "data_origin": "SYNTHETIC", "seed": data["seed"], "beta": data["beta"], "market": "US",
            "n_symbols": len(symbols), "n_days": len(cal.days), "n_articles": len(arts),
            "n_assignments_kept": int(len(asg)), "first_day": cal.days[0].isoformat(),
            "last_day": cal.days[-1].isoformat(), "lexicon_note": LEXICON_NOTE}
    db_path = db.write_demo(ROOT / "var" / "demo.sqlite", data, feats_by, results, robust, meta)
    path = ROOT / "web" / "data.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(api.build_bundle(db_path), allow_nan=False, separators=(",", ":")))
    for w, r in results.items():
        c = r["comparisons"]["primary"][0]
        print(f"{w:8s} delta_ic={c['delta']:+.4f} p={c['p']:.4f} holm_p={c['p_holm']:.4f} -> {r['verdict']}")
    print(f"robustness (10 seeds): null false-alarm rate {robust['null_false_alarm_rate']:.2f}, "
          f"planted detection rate {robust['planted_detection_rate']:.2f}")
    print(f"wrote {db_path}")
    print(f"wrote {path} ({path.stat().st_size // 1024} KB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
