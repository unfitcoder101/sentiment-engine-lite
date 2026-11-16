# STATUS

Done: Stage 1 (core logic, 23 tests), Stage 2 (synthetic data, scorer, experiment, 7 tests),
Stage 3 (web/index.html, serve.py, screenshots in docs/), Stage 4 (README, docs/METHOD.md, local commit).

Results (seed 7, beta 0.002): null "no effect flagged" (p 0.0920); planted "planted effect detected" (p 0.0005).

Next: nothing required. Planned items are listed in README (not built).

Known issues:
- Null-world p-value 0.092 is close to the 0.05 threshold; only one seed was run.
- A fifth verdict text ("effect flagged in null world (false alarm)") was added for a case the spec left open.
- Articles with no lexicon coverage are excluded from sentiment features (a design choice, see METHOD.md).
- 360 px layout checked via a 360 px iframe in headless Chrome (Chrome will not open a window that narrow).
- Headlines/Cutoff tabs were not screenshot-checked, only Overview, Sentiment, Experiments.
- conftest.py (empty) at the root is an extra file so pytest finds the `se` package.
