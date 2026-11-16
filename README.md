# Sentiment Engine Lite

**Educational research prototype. Data origin: SYNTHETIC. No evidence of predictive value. Not investment advice.**

A university prototype of a research pipeline for the question "does financial-news sentiment help rank
stocks?". It runs only on invented prices and invented headlines. It is a tested, explainable prototype with an
extensible research framework; it says nothing about real markets.

## Run

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py
.venv/bin/python -m se.cli serve
```

Open http://127.0.0.1:8000/ . The demo step takes about 25 seconds. `.venv/bin/python serve.py` also works
(static dashboard without the interactive tools). Full check: `sh scripts/check_all.sh`.

## What it shows

- **Timing:** each article is assigned to a trading day with a 15:30 New York cutoff; 15 worked cases.
- **Labels and features:** open-to-open returns scaled by volatility; four sentiment features.
- **A control experiment:** two synthetic worlds, one with an inserted link (beta 0.002), compared on frozen,
  hashed samples with placebos and a pre-registered success rule.
- **Storage and API:** SQLite, a read-only GET API, a ten-tab dashboard.
- **Gates:** which data and scorers are blocked, and why.

Results as produced (seed 7):

| World | Delta IC (C - B) | Holm-adjusted p | Verdict |
|---|---|---|---|
| null | +0.0111 | 0.3973 | SYNTHETIC SOFTWARE CHECK: no effect flagged |
| planted | +0.0603 | 0.0015 | SYNTHETIC SOFTWARE CHECK: placebo not excluded |

Ten pre-declared seeds: null-world false-alarm rate 0 of 10; planted-world detection rate 4 of 10.
See docs/METHODOLOGY.md section 10 for why the lagged placebo was not passed on seed 7.

## What it does NOT show

- Nothing about real companies, news or markets.
- The scorers are a demo word list (not Loughran-McDonald) and a MOCK MODEL (not FinBERT); both are circular on this data.
- FinBERT, finbert-tone and Loughran-McDonald adapters exist as code tested with fake files; no weights or dictionary are included.
- No pipeline over real or imported data exists. Confirmatory mode is refused by code.

## Documents

docs/PROFESSOR_WALKTHROUGH.md, DEMO_GUIDE.md, METHODOLOGY.md, WORKED_EXAMPLES.md, ARCHITECTURE.md, MODEL_CARD.md,
GATES.md, LIMITATIONS.md, CLAIMS_LEDGER.md, DECISIONS.md, GLOSSARY.md, SETUP.md, PREREGISTRATION_SYNTHETIC.md,
FINAL_REPORT.md. Screenshots: docs/screenshots/.

## Layout

`se/` modules, `se/scorers/` adapters, `configs/profile.yaml`, `gates.yaml`, `run_demo.py`, `serve.py`,
`web/index.html`, `tests/`, `scripts/check_all.sh`, `runs/prereg.json`.
