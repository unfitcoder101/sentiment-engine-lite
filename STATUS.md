# STATUS

Done: Stages 1-4 (PROMPT.md) and Phases 5-9 (PROMPT2.md). All committed locally; no remote, nothing pushed.
Tests: 99 passed, 1 skipped (needs real model weights). `sh scripts/check_all.sh` passes; verify 12/12.

Results as produced (SYNTHETIC data, seed 7, beta 0.002, rules frozen before the first run):
- null:    SYNTHETIC SOFTWARE CHECK: no effect flagged
- planted: SYNTHETIC SOFTWARE CHECK: placebo not excluded
- 10 seeds: null false-alarm rate 0.00; planted detection rate 0.40

Run: `.venv/bin/python run_demo.py` then `.venv/bin/python -m se.cli serve` (http://127.0.0.1:8000/).
Read first: docs/FINAL_REPORT.md, docs/PROFESSOR_WALKTHROUGH.md, docs/DECISIONS.md.

Next: nothing required. Owner actions are listed in docs/FINAL_REPORT.md and docs/GATES.md.

Known issues:
- Planted link flagged on only 4 of 10 seeds; baseline B is unstable in a 30-ticker universe (see METHODOLOGY.md s10).
- Cutoff-explainer button not clicked in a browser (endpoint tested, form screenshot-checked).
- Adapters and importers never run on real inputs. No pipeline over imported data.
- Do not re-run `prereg-freeze` unless starting a new pre-registration.
