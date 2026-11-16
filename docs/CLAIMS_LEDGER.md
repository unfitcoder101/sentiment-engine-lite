# Claims ledger

## Implemented and tested (on hand cases or synthetic data)
- Trading-day assignment with cutoff, weekends, holidays, date-only and invalid timestamps (US and NSE).
- De-duplication and multi-ticker weights.
- Execution-based labels; S1, S5, NVS, ABN features; future-deletion invariance.
- Synthetic generator with a null and a planted world.
- Demo lexicon and MOCK MODEL scorers.
- Frozen samples with five abort checks, two purged folds, nine arms, bootstrap, Holm, two placebos, one verdict function.
- Pre-registration hashing, run modes, confirmatory guard.
- SQLite storage, CSV importers, read-only API, dashboard.

## Verified on synthetic data only
- The pipeline stays quiet where no link exists: 0 of 10 seeds flagged in the null world.
- The pipeline flags an inserted link (beta 0.002) on 4 of 10 seeds; on seed 7 it reports "placebo not excluded".
- Identical frozen-sample hashes and verdict texts across repeated runs.

## Implemented but never run on real inputs
- FinBERT, finbert-tone and Loughran-McDonald adapters (fake files only).
- Importers (tiny hand-made CSV files only).

## Proposed, not built
- Any pipeline over imported or licensed real data.
- A confirmatory study (blocked by the gates).
- Exchange-grade calendars, a richer cost model, more seeds or a larger universe.

## Cannot be claimed
- That news sentiment helps rank real stocks.
- That any number here estimates a real-world effect.
- That the scorers measure real tone, or that results would transfer to real data.
- That the design has adequate power: on this small universe it flagged the inserted link less than half the time.
