# Final report (2026-11-16) - all data SYNTHETIC

| Phase | Status | Evidence |
|---|---|---|
| 5 Experiment framework | COMPLETE | 20 tests; both worlds + 10 seeds run in 24 s; pre-registration frozen before the first run |
| 6 Gates and scorer adapters | COMPLETE (fake files only) | 7 tests; no weights or dictionary present, adapters never run on real inputs |
| 7 SQLite and importers | COMPLETE | 5 tests; importers store only, no pipeline over imported data |
| 8 API and dashboard | COMPLETE, one gap | 13 tests; cutoff-explainer button not clicked in a browser (its endpoint is tested) |
| 9 Docs and verification | COMPLETE | 24 tests; `sh scripts/check_all.sh` passes; verify 12/12 |

- Tests: 99 passed, 0 failed, 1 skipped (needs real local model weights, which this project does not hold).
- Null world verdict: `SYNTHETIC SOFTWARE CHECK: no effect flagged` (delta IC +0.0111, Holm p 0.3973).
- Planted world verdict: `SYNTHETIC SOFTWARE CHECK: placebo not excluded` (delta IC +0.0603, Holm p 0.0015; shuffle placebo C2 passed, lagged placebo C3 not passed).
- Ten pre-declared seeds: null false-alarm rate 0 of 10; planted detection rate 4 of 10 (3 not detected, 3 placebo not excluded; all five criteria met on 1).
- Reading: the instrument stays quiet with no link, but flags the inserted link less than half the time. With 30 tickers the price-only baseline is unstable, which is why the lagged placebo also beat it. Nothing was tuned after seeing this.
- Reproducibility: two rebuilds from a clean state gave identical frozen-sample hashes, verdicts and robustness table.
- Screenshots: every one of the 10 tabs at desktop and 360 px (20 files in docs/screenshots/), each one looked at; layout bugs found and fixed (Experiments tables, placebo chart size, timestamp wrapping). Static fallback mode checked by DOM dump on 4 tabs.
- Safety: work stayed in this folder plus a session scratch folder (Chrome profile, temp files); pip used its normal cache. No git remote, nothing pushed. Network used only for `pip install` and 127.0.0.1.
- Blocked items and unlocking action: FNSPID (written permission from authors + readable publisher terms); Alpaca/Benzinga (owner reads and records terms); finbert-tone (clear licence statement); FinBERT ProsusAI (documented training cutoff; exploratory only until then); Loughran-McDonald (dated pre-test version supplied locally); confirmatory study and real-data storage (all of the above resolved).
- Owner must still do by hand: review gates.yaml; push to a private repository if wanted; supply local model weights or an LM dictionary only under their terms; decide on a larger universe or more seeds before any real study.
- Deviations from the brief are listed in docs/DECISIONS.md (17 items).

Description: a tested, explainable prototype with an extensible research framework, run only on synthetic data. It provides no evidence about real markets.
