# STATUS

Done: Stages 1-4 (PROMPT.md); PROMPT2.md Phases 5, 6, 7, 8. Each phase committed locally.
- 5: experiment framework, pre-registration (frozen before first run), gates.yaml, run modes.
- 6: LM lexicon / local HF classifier / FinBERT / finbert-tone adapters (fake-tested only, no weights).
- 7: SQLite layer (var/demo.sqlite, gitignored), CSV importers.
- 8: read-only FastAPI (`python -m se.cli serve`), 10-tab dashboard with static fallback (web/data.json),
     20 screenshots in docs/screenshots/ (every tab, desktop and 360 px, all looked at).

Results as produced (seed 7, beta 0.002):
- null:    SYNTHETIC SOFTWARE CHECK: no effect flagged
- planted: SYNTHETIC SOFTWARE CHECK: placebo not excluded  (C vs B delta IC +0.0603, Holm p 0.0015; C3 not passed)
- 10 seeds: null false-alarm rate 0.00; planted detection rate 0.40

Next: Phase 9 (docs set, 14 US cutoff cases + NSE cases in tests, scripts/check_all.sh, verify, final report).

Known issues:
- Baseline B has negative test IC on seed 7, so placebo C3 (and D) beat B even in the null world. Not tuned.
- Cutoff-explainer button was not clicked in a browser (its API endpoint is tested; the form is screenshot-checked).
- README still describes Stages 1-4 until Phase 9.
