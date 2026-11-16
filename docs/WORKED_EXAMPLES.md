# Worked examples

Every number here is reproduced by a test (tests/test_stage1.py, test_phase5.py, test_phase9.py) and shown live in the
dashboard's Method tools tab. The inputs are hand-made or SYNTHETIC.

## Trading-day assignment, US market (cutoff 15:30 New York, inclusive)
Calendar: weekdays of 2023 minus 01-02, 01-16, 02-20, 04-07, 05-29, 06-19, 07-04, 09-04, 11-23, 12-25.

| # | Published (UTC) | New York time | Assigned day | Reason |
|---|---|---|---|---|
| 1 | 2023-03-14 19:30Z | 15:30:00 EDT | 2023-03-14 | same_day (exactly at cutoff) |
| 2 | 2023-03-14 19:30:01Z | 15:30:01 EDT | 2023-03-15 | after_cutoff |
| 3 | 2023-01-10 20:30Z | 15:30 EST | 2023-01-10 | same_day (winter offset) |
| 4 | 2023-03-18 15:00Z | Saturday 11:00 EDT | 2023-03-20 | weekend |
| 5 | 2023-03-17 21:00Z | Friday 17:00 EDT | 2023-03-20 | after_cutoff |
| 6 | date-only 2023-03-14 | none | 2023-03-15 | date_only |
| 7 | date-only 2023-03-17 | none | 2023-03-20 | date_only |
| 8 | 2023-11-22 21:00Z | 16:00 EST | 2023-11-24 | after_cutoff, Thanksgiving closed |
| 9 | 2023-11-23 15:00Z | 10:00 EST | 2023-11-24 | holiday |
| 10 | 2023-01-10 19:30Z | 14:30 EST | 2023-01-10 | same_day |
| 11 | 2023-11-03 19:30Z | 15:30 EDT | 2023-11-03 | same_day (last Friday before clocks change) |
| 12 | 2023-11-03 20:00Z | 16:00 EDT | 2023-11-06 | after_cutoff |
| 13 | 2023-11-06 20:30Z | 15:30 EST | 2023-11-06 | same_day (first Monday after clocks change) |
| 14 | 2023-11-06 19:30Z | 14:30 EST | 2023-11-06 | same_day |

A timestamp with no zone (for example `2023-03-14T19:30:00`) is rejected as invalid.

NSE (cutoff 15:00 Kolkata): 09:30:00Z is 15:00 IST, same day; 09:30:01Z goes to the next trading day;
18:45Z on 13 March is 00:15 IST on 14 March, so it belongs to 14 March.

## De-duplication
Copies of one headline for one stock at 0h, 47h, 49h, 96h: keep 0h; drop 47h (within 48h of 0h); keep 49h;
drop 96h (47h after the last kept copy). Kept: 0h and 49h.

## True range and ATR (window 3)
Bars (High, Low, Close): t0 101, 99, 100; t1 103, 101, 102; t2 102, 100, 101; t3 104, 102, 103.
TR = 3, 2, 3. ATR = 8 / 3 = 2.666667. ATR% = 2.666667 / 103 = 0.025890.

## Executable return
Open(t+1) = 101, Open(t+11) = 108.9: R_exec = 108.9 / 101 - 1 = 0.078218. With ATR% = 0.02, Z_exec = 3.910891.
With ATR% = 0.0004 the floor 0.001 is used instead.

## Top-30% label
20 stocks with distinct Z_exec: percentile ranks 0.05 to 1.00; ranks 0.70 and above are 14/20 to 20/20, which is
7 of 20 labels. A date with 19 stocks gets no labels. Ties 1, 2, 2, 3 get average ranks 1, 2.5, 2.5, 4.

## Purge and embargo
First test bar 100, horizon 10, embargo 5: need t + 11 < 95, so the last training day is 83.

## Holm correction
Raw p = 0.01, 0.04, 0.03. Sorted: 0.01, 0.03, 0.04. Multiply by 3, 2, 1: 0.03, 0.06, 0.04. Take running maximum:
0.03, 0.06, 0.06. In the original order: 0.03, 0.06, 0.06.

## Rank metrics
Spearman of (1, 2, 3, 4) against (1, 3, 2, 4) is 0.8. AUC of scores (0.1, 0.4, 0.35, 0.8) with labels (0, 0, 1, 1) is 0.75.
