# Decisions

Recorded so that choices made while building are visible. All runs use SYNTHETIC data.

1. **Freeze before the first run.** `configs/profile.yaml` and the pre-registration were hashed (2026-11-16) before the Phase 5 framework produced any number. Nothing in them has changed since.
2. **Results kept as produced.** Seed 7 gives "placebo not excluded" in the planted world. Seeds, beta, thresholds and placebo rules were left alone.
3. **Two flag levels in the verdict.** "Flagged" = criteria 1 and 2. All five criteria = "criteria met". Written in the pre-registration before running.
4. **A fifth synthetic verdict text.** "effect flagged in null world (false alarm)" covers a case the original list left open. It has not occurred in any run.
5. **Noise scale of the mock model.** "N(0, 0.5)" was read as standard deviation 0.5.
6. **S2 minimum of 10 stocks per date.** P1 uses 20 as specified; S2 would lose most dates at 20, so 10 was declared in the config before running.
7. **No-coverage headlines.** Left out of the demo-lexicon features (they have no score); the mock model scores every headline.
8. **Rates.** False-alarm and detection rates count seeds that are flagged with both placebos passed.
9. **Run identifiers carry no timestamp** (`<mode>-seed<n>-<world>`), so repeated runs overwrite and can be compared byte for byte.
10. **`runs/prereg.json` is tracked in git**; the rest of `runs/` and `var/` are ignored. A clean rebuild keeps the freeze.
11. **Framework lives in `se/study.py`**, reusing `se/experiment.py` rather than rewriting it; the Stage 2 functions and tests remain.
12. **Headlines carry both scorers** as two separate objects; the API never merges lexicon buckets with model labels.
13. **The API rejects out-of-range limits** (400) instead of silently clamping them.
14. **Assign-day tool calendar** = synthetic 2020-2022 days plus the 2023 demo calendar; labelled illustrative.
15. **Importers only store.** No pipeline over imported data was built, so nothing real can be run by accident.
16. **Wording.** A fixed list of overclaiming words (see `se/verdicts.py`) is tested against verdicts, API strings, the dashboard and the docs.
17. **METHOD.md replaced** by METHODOLOGY.md.
