# Glossary

All examples in this project use SYNTHETIC data.

- **Arm**: one version of the model, defined by which inputs it gets (B = prices only, C = prices plus sentiment, and so on).
- **ATR / ATR%**: average true range over 14 days, a volatility measure; ATR% divides it by the close.
- **Block bootstrap**: resampling runs of consecutive days so that day-to-day dependence is respected.
- **Bucket**: a lexicon's category for a headline (positive, negative, neutral (zero-net), no lexicon coverage).
- **Contamination**: the risk that a scorer was trained on text from the test period.
- **Cutoff**: the local time after which news counts for the next trading day (15:30 New York).
- **Delta IC**: the difference in mean daily IC between two arms.
- **Embargo / purge**: the gap that keeps training labels from overlapping the test period.
- **Fold**: one train/test split (test year 2021, then 2022).
- **Frozen sample**: a fixed, hashed list of rows that every arm must use.
- **Gate**: a recorded permission or block on a data source, scorer or study type.
- **Holm correction**: a way to adjust several p-values so that testing three things does not inflate false alarms.
- **IC**: daily Spearman rank correlation between model score and the later scaled return.
- **Label (model)**: a classifier's chosen class. Kept separate from lexicon buckets.
- **L_exec, R_exec, Z_exec**: the top-30% flag, the open-to-open return and its ATR-scaled version.
- **Null world / planted world**: synthetic data without and with an inserted link from tone to returns.
- **P1 / S2 / PANEL**: test rows / test rows with news / all usable rows.
- **Placebo**: a deliberately broken version of the inputs (shuffled or lagged) that should not help.
- **Pre-registration**: writing the rules down and hashing them before running.
- **S1, S5, NVS, ABN**: tone today, tone over five days, news-volume surprise, abnormal tone.
- **Verdict**: one fixed sentence produced by one function from the fixed success rule.
