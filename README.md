# Sentiment Engine Lite

**Educational research prototype. Data origin: SYNTHETIC. No evidence of predictive value. Not investment advice.**

A small university prototype of a pipeline for the question "does financial-news sentiment help rank stocks?".
This version uses only invented prices and invented headlines. It checks that the software works; it says
nothing about real markets.

## Run

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py
.venv/bin/python serve.py
```

Then open http://127.0.0.1:8000/ . The demo step takes a few seconds and writes `web/data.json`.
Tests: `.venv/bin/python -m pytest -q` (30 tests).

## What it shows

- **Time handling:** each article is assigned to a trading day using a 15:30 New York cutoff, with weekends,
  holidays, date-only items and invalid timestamps handled explicitly (Cutoff cases tab).
- **De-duplication:** repeated headlines within 48 hours are dropped; multi-ticker articles are split 1/k.
- **Labels:** next-day open to the open 10 trading days later, scaled by ATR, top 30% per date.
- **Features:** S1, S5, news-volume surprise (NVS) and abnormal sentiment (ABN), ranked per date.
- **A software check:** two synthetic worlds share all random draws. In the "null" world news has no link to
  prices. In the "planted" world a known link (beta = 0.002 per day) is inserted on purpose. The experiment
  compares a price-only baseline with baseline + sentiment, with a shuffled-sentiment placebo and a block bootstrap.

Result with seed 7:

| World | Delta IC | p-value | Verdict |
|---|---|---|---|
| null | +0.0203 | 0.0920 | SYNTHETIC SOFTWARE CHECK: no effect flagged |
| planted | +0.0605 | 0.0005 | SYNTHETIC SOFTWARE CHECK: planted effect detected |

## What it does NOT show

- Nothing about real companies, real news or real markets. Every number comes from a random generator.
- The planted link was put there by the generator, so finding it is expected. It is a wiring test only.
- The word list is a hand-written demo list, **not Loughran-McDonald**, and the fake headlines are built from the
  same words, so scoring is circular by construction.
- One seed, one train/test split, unadjusted interval. Model scores are ranking scores, not probabilities.

## Planned (not built in this version)

FinBERT / finbert-tone / Loughran-McDonald adapters, SQLite layer and CSV importers, FastAPI service,
pre-registration hashing, ablations, power table, confirmatory-mode guard, full docs set.

Blocked items: FNSPID and Alpaca licences unresolved; FinBERT training dates unverified (exploratory only);
Loughran-McDonald version not frozen.

## Layout

`se/` core modules, `run_demo.py` pipeline, `serve.py` local server (127.0.0.1 only), `web/index.html` single-file
frontend (no external requests), `tests/`, `docs/METHOD.md`, `docs/screenshot-*.png` (headless Chrome, desktop and 360 px).
