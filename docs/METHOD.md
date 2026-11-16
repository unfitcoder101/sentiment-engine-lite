# Method (plain language)

**All data is synthetic. Nothing here is evidence about markets.**

## 1. From publication time to trading day
Each article has a UTC timestamp. It is converted to the market's local time with the system time-zone
database (no hard-coded offsets). If the local day is a trading day and the time is at or before the cutoff
(15:30 New York; 15:00 Kolkata for NSE), the article belongs to that day. Otherwise it belongs to the next
trading day. Date-only items always go to the next trading day. Missing, zone-less and after-as-of
timestamps are rejected.

## 2. De-duplication
Headlines are normalised (NFKC, lowercase, no punctuation, no ticker). A repeat for the
same stock within 48 hours of the last kept copy is dropped. An article naming k stocks counts 1/k for each.

## 3. Label
For signal day t: R_exec = Open(t+11) / Open(t+1) - 1. Z_exec = R_exec divided by ATR%(t) (14-day average true range over close, floor 0.001). L_exec = 1
if a stock is in the top 30% of Z_exec on that date (needs at least 20 stocks). This is not claimed to be
equivalent to any close-to-close label.

## 4. Sentiment features
Using only articles assigned to days up to t: S1 (weighted mean score on day t), S5 (days t-4..t),
NVS (log news count in the last 5 days versus the prior 60-day rate), ABN (S5 minus the prior 60-day mean,
needs 10 baseline articles), has_news. The first 64 days are excluded. Each feature is turned into a
per-date rank; missing values get 0.5.

## 5. Scorer
Score = (positive words - negative words) / tokens, from a hand-written demo list of 20 + 20 words.
It is **not Loughran-McDonald**. The fake headlines are generated from the same list, so the scorer is
circular by construction. Headlines with no listed word are "no lexicon coverage": they are not called
neutral and are left out of the sentiment features.

## 6. Synthetic worlds
30 fake tickers, business days 2020-2022, daily volatility 1-3%, about 0.5-3 articles per stock per week,
each with a hidden tone u. Both worlds share every random draw. In the null world tone and prices are
unrelated. In the planted world the mean tone of day t adds beta = 0.002 per day to returns on days
t+1..t+5, with half of the first day's effect in the overnight gap. Beta was fixed before running. An
earlier, larger build used 0.0005, which was too small to detect.

## 7. Experiment
Train 2020-2021, test 2022, with training rows required to finish 5 days before the first test bar.
Model: standardised logistic regression on L_exec. Arm B uses price features (atr_pct, ret_5, ret_20,
vol_10). Arm C adds the sentiment block. Placebo C2 shuffles the sentiment block across stocks within
each date, 20 times. Metric: daily Spearman rank correlation (IC) between model score and Z_exec; the
quantity of interest is the mean daily difference C - B. A paired stationary bootstrap (mean block 20,
2000 resamples, fixed seed) gives a one-sided p-value and an unadjusted 95% interval. Flag rule, fixed in advance: p < 0.05 and the observed difference at least as large as all 20 placebos.

## 8. Reading the result
"Planted effect detected" means the software found something the generator inserted. "No effect flagged"
means it stayed quiet where nothing was inserted. With seed 7 the null world's p-value is 0.092: not
flagged, but one seed and one split are a thin check. If the null world were ever flagged, the code reports
"effect flagged in null world (false alarm)".
