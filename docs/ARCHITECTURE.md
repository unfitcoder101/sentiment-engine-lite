# Architecture

All data in this project is SYNTHETIC. This page explains how the pieces fit together.

## Layers

```
 dashboard (web/index.html)            one file, no external requests
        |  GET only
 API (se/api.py, FastAPI)              reads, never scores, never writes
        |  read-only connection
 storage (se/db.py, var/demo.sqlite)   articles, assignments, scores, features, runs
        ^
 run_demo.py                           writes the database and web/data.json
        |
 experiment (se/study.py, se/stats.py, se/metrics.py, se/verdicts.py)
        |
 features (se/features.py)   labels (se/labels.py)
        |
 scoring (se/scorer.py, se/scorers/)   demo lexicon, MOCK MODEL, gated adapters
        |
 timing and clean-up (se/timeutil.py, se/dedup.py)
        |
 data (se/synthetic.py; se/importers.py for bring-your-own CSV)
 rules: configs/profile.yaml, docs/PREREGISTRATION_SYNTHETIC.md, gates.yaml (se/config.py, se/gates.py)
```

## The life of one article

1. **Created** by the generator (or imported) with a first-published time, a headline and one or more tickers.
2. **Assigned** to a trading day by `timeutil.assign()`: at or before 15:30 New York means that day, otherwise the next trading day. Invalid timestamps are rejected with a reason.
3. **De-duplicated**: a repeat of the same normalised headline for the same stock within 48 hours is dropped. A k-ticker article gets weight 1/k per stock.
4. **Scored** by each scorer: the demo lexicon gives a bucket and a count-based x; the MOCK MODEL gives three output scores and a label.
5. **Aggregated** into S1, S5, NVS, ABN per stock and day, using only articles assigned up to that day, then ranked across stocks.
6. **Used** as model inputs in arms C, C_LM, C1, C2, C3 and D, and compared with price-only arm B on frozen samples.
7. **Stored** in SQLite and served read-only to the dashboard.

## Extension points

- New scorer: add a module under `se/scorers/`, add a gate entry in `gates.yaml`, add it to `SCORERS` in `se/api.py`.
- New market: add an entry to `MARKETS` in `se/timeutil.py` and supply a trading calendar.
- New feature: add it in `se/features.py` and to the design matrix in `se/study.py`; this changes the pre-registration, so re-freeze before running.
- Real data: only through `se/importers.py`, with a licence note; no pipeline over imported data exists yet.

## What is deliberately absent

No network code, no credentials, no model weights, no real prices or news.
