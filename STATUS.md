# STATUS

Done: Stages 1-4 (PROMPT.md) and Phase 5 of PROMPT2.md (experiment framework). 50 tests pass. Demo 23 s.
Phase 5 adds: MOCK MODEL scorer, arms A/B/C/C_LM/C1/C2/C3/D, frozen samples + 5 abort checks, 2 folds,
secondary metrics, Holm, placebo criteria, success rule, ablations, power, 10-seed robustness,
configs/profile.yaml + docs/PREREGISTRATION_SYNTHETIC.md frozen in runs/prereg.json, run modes, gates.yaml.

Results as produced (seed 7, beta 0.002, pre-registration frozen before the first run):
- null:    SYNTHETIC SOFTWARE CHECK: no effect flagged
- planted: SYNTHETIC SOFTWARE CHECK: placebo not excluded  (C vs B delta IC +0.0603, Holm p 0.0015; C3 placebo not passed)
- 10 seeds: null false-alarm rate 0.00; planted detection rate 0.40 (3 not detected, 3 placebo not excluded, 4 flagged)

Next: Phase 6 (scorer adapters), 7 (SQLite, importers), 8 (API, dashboard), 9 (docs, final checks).

Known issues:
- Baseline B has a negative test IC on seed 7 in both worlds, so arms that add any columns (C3, D) look
  better than B even in the null world. This is why the C3 placebo is not passed. Not tuned; documented.
- README results table and web dashboard still show the Stage 1-4 layout until Phases 8-9.
- runs/ is gitignored except runs/prereg.json.
