# Pre-registration: synthetic software check

Written before the Phase 5 framework was run. All data is SYNTHETIC. This document and
`configs/profile.yaml` are hashed into `runs/prereg.json`; a `synthetic_validation` run aborts if either changes.

## Purpose
Check that the software can see an effect that the data generator inserted, and stays quiet where none
was inserted. Nothing here can say anything about real markets.

## Hypotheses (about the software, on invented data)
- H1 (planted world, beta = 0.002): adding sentiment features (arm C) raises mean daily IC over the baseline (arm B).
- H0 check (null world): the same procedure does not flag an effect.

## Data
30 fake tickers, business days 2020-2022, main seed 7. Robustness seeds 1 to 10, declared here before any run.
Scorers: primary = MOCK MODEL (not FinBERT), a deterministic fake classifier whose logits are
1.2*P + e1, 1.2*N + e2, 0.8 + e3 with e ~ Normal(mean 0, sd 0.5) taken from a hash of the headline;
lexicon baseline = the hand-written demo lexicon (not Loughran-McDonald).

## Samples (frozen once per run, before any placebo)
- PANEL: rows after the 64-day warm-up with a valid baseline vector and valid R_exec, Z_exec, L_exec.
  Membership depends only on prices and labels.
- P1: PANEL rows in the test years 2021 and 2022.
- S2: P1 rows whose original (unshuffled, unlagged) has_news = 1 for the primary scorer.
Key lists are written to `runs/<run_id>/samples/` with SHA-256 hashes. Five checks abort the run:
training keys equal the frozen fold keys; evaluated keys equal frozen P1; S2 keys equal frozen S2;
daily-IC date lists equal across arms; row counts equal across arms.

## Arms
A_lowvol (score = -ATR%), A_highvol (+ATR%), B (atr_pct, ret_5, ret_20, vol_10),
C (B + S1, S5, NVS, ABN ranks + has_news, primary scorer), C_LM (same from the demo lexicon),
C1 (B + NVS + has_news), C2 (C with the sentiment block shuffled across stocks within each date, seeds 1000-1019),
C3 (C with sentiment taken from 20 trading days earlier; missing = rank 0.5, has_news 0), D (sentiment only).

## Model and folds
StandardScaler + LogisticRegression(C = 1, class_weight balanced), target L_exec, refit inside each fold.
Expanding window, test years 2021 and 2022. A training row needs t + 11 < first test bar - 5.

## Metrics
Primary: mean daily Spearman IC of score vs Z_exec over P1 dates with at least 20 stocks.
Secondary (P1, and S2 with at least 10 stocks per date): IC vs R_exec, mean daily ROC-AUC vs L_exec,
precision@K (K = top 10% of the day), net-of-cost top-K excess return
(mean R_exec of top-K - median R_exec of the day - 0.004), and IC by year.

## Statistics
Paired stationary bootstrap on the daily IC difference: mean block 20, 2000 resamples, seed 12345.
One-sided p = (1 + #{centred resampled mean >= observed}) / (B + 1). Intervals are unadjusted.
Block lengths 10 and 40 are reported as sensitivity only. Holm-Bonferroni over exactly three primary
comparisons: C vs B, C vs C1, C vs C_LM. Everything else is exploratory.

## Placebo criteria
- C2 PASSED if delta_IC(C - B) >= max of the 20 shuffled deltas AND the mean shuffled delta < 0.005.
- C3 PASSED if delta_IC(C3 - B) <= 0.5 * delta_IC(C - B) AND the unadjusted one-sided p of C3 vs B > 0.10.

## Success rule (P1)
1. Holm-adjusted p of C vs B <= 0.05.
2. delta_IC(C - B) >= 0.01.
3. C beats C1 and C_LM: Holm-adjusted p <= 0.05 and positive deltas.
4. Both placebos PASSED.
5. Net-of-cost top-K of C not lower than B.

## Verdict texts (one function)
"Flagged" means criteria 1 and 2 hold. Not flagged: "not detected" (planted world) or "no effect flagged"
(null world). Flagged but a placebo not passed: "placebo not excluded". Flagged with placebos passed:
planted world gives "criteria met" if all five criteria hold, otherwise "planted effect detected";
null world gives "effect flagged in null world (false alarm)". All start with "SYNTHETIC SOFTWARE CHECK:".
Expected in advance: criterion 3 will often fail, because the demo lexicon recovers the hidden tone by
construction and the mock model is a noisier copy of it. That is not a defect and will not be tuned away.

## Robustness
Both worlds for seeds 1-10. Reported: null-world false-alarm rate (share of seeds flagged with placebos
passed) and planted-world detection rate (same definition). Seeds are not changed after seeing results.
The reason for this section: an earlier single-seed run gave a null-world p of 0.092.

## Exploratory (labelled as such)
Ablations dropping S1, S5, NVS, ABN in turn; article weights 1 versus 1/k; power approximation
MDE ~ bootstrap SE * (z(1 - 0.05/3) + 0.8416).
