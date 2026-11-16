# Methodology

**All data is synthetic. Nothing here is evidence about markets.** This replaces the earlier METHOD.md.

## 1. Timing and cutoff
A signal for day t may only use news that existed by the cutoff on day t. Each article is converted from UTC to
market local time using the system time-zone database. If the local day is a trading day and the time is at or
before the cutoff (15:30 New York, 15:00 Kolkata), it belongs to that day; one microsecond later it belongs to
the next trading day. Weekend, holiday and date-only items go to the next trading day. Missing, zone-less and
after-as-of timestamps are rejected. Example: 19:30:00Z on 14 March 2023 is 15:30 in New York, so 14 March;
19:30:01Z goes to 15 March.

## 2. Executable label
A signal formed at 15:30 cannot trade at that day's close, so the label starts at the next open:

- R_exec(t) = Open(t+11) / Open(t+1) - 1
- Z_exec(t) = R_exec(t) / max(ATR%(t), 0.001), where ATR% is the 14-day average true range divided by the close
- L_exec(t) = 1 if Z_exec is in the top 30% of stocks that date (at least 20 stocks needed)

No equivalence with any close-to-close label is claimed.
Worked number: Open(t+1) = 101, Open(t+11) = 108.9, ATR% = 0.02 gives R_exec = 0.078218 and Z_exec = 3.910891.

## 3. Features
S1 (tone today), S5 (tone over five days), NVS (news volume versus the stock's own recent rate), ABN (tone versus
the stock's own recent tone), has_news. Each is ranked across stocks per date; missing values get 0.5. The first
64 trading days are a warm-up. A test confirms that deleting later articles never changes features at day t.

## 4. Frozen samples
PANEL = rows with valid price features and labels (depends only on prices). P1 = PANEL rows in 2021 and 2022.
S2 = P1 rows with news. The lists are written and hashed once, before any placebo. Five checks stop the run if any
arm trains or is scored on different rows.

## 5. Folds, purge and embargo
Expanding window: train on everything before 2021 and test on 2021; then train before 2022 and test on 2022.
A training row needs t + 11 < first test bar - 5, so its label window ends at least five bars before testing starts.
Example: first test bar 100 means the last usable training day is 83.

## 6. Statistics
The primary number is the mean daily Spearman rank correlation (IC) between model score and Z_exec. Arms are
compared by the daily difference in IC, with a paired stationary bootstrap (mean block 20 days, 2000 resamples,
fixed seed). The one-sided p-value is (1 + count of centred resampled means at or above the observed mean) / 2001.
Intervals are unadjusted.

## 7. Holm correction
Three planned comparisons (C vs B, C vs C1, C vs C_LM) are adjusted together. Example: raw p of 0.01, 0.04, 0.03
become 0.03, 0.06, 0.06. Everything else is labelled exploratory.

## 8. Placebos
C2 shuffles the sentiment block across stocks within each date, 20 times; the real result must be at least as
large as every shuffle and the shuffles must average under 0.005. C3 uses sentiment from 20 trading days earlier;
its gain over B must be at most half of C's and not stand out on its own (p above 0.10).

## 9. Success rule and verdict
Five fixed criteria (see docs/PREREGISTRATION_SYNTHETIC.md). One function turns them into one fixed sentence.

## 10. Results as produced (seed 7, beta 0.002)
- Null world: "SYNTHETIC SOFTWARE CHECK: no effect flagged". Delta IC (C - B) +0.0111, Holm p 0.3973.
- Planted world: "SYNTHETIC SOFTWARE CHECK: placebo not excluded". Delta IC +0.0603, Holm p 0.0015, C2 passed,
  C3 not passed (C3 gained +0.0329 over B against a limit of +0.0302).
- Ten pre-declared seeds: null-world false-alarm rate 0 of 10; planted-world detection rate 4 of 10
  (3 not detected, 3 placebo not excluded; all five criteria met on 1 of 10).

Why C3 did not pass on seed 7: the price-only baseline B had a negative test IC (-0.022), so adding almost any
columns pulled the model back towards zero and looked like a gain. The same thing happens in the null world,
where no link exists at all. This is a weakness of comparing against an unstable baseline in a small universe.
It was not tuned away.

## 11. Why a null or mixed result is acceptable
The aim is an honest instrument. A design that reports "not detected" when the data is too thin is doing its job.
Seeds, beta, thresholds and criteria were frozen before the first run and have not been changed since.
