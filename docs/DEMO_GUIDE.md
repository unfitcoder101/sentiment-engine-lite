# Demo guide

Everything shown is SYNTHETIC. Say so first; the banner says it on every screen.

## Commands
```
.venv/bin/python run_demo.py
.venv/bin/python -m se.cli serve
```
Expected output of the first command (about 25 seconds):
```
null     delta_ic=+0.0111 p=0.1324 holm_p=0.3973 -> SYNTHETIC SOFTWARE CHECK: no effect flagged
planted  delta_ic=+0.0603 p=0.0005 holm_p=0.0015 -> SYNTHETIC SOFTWARE CHECK: placebo not excluded
robustness (10 seeds): null false-alarm rate 0.00, planted detection rate 0.40
```
Then open http://127.0.0.1:8000/ .

## What to click and what to say
1. **Overview.** "This is a pipeline check on invented data. Two worlds: one with no link, one with a link I inserted."
2. **Cutoff cases.** "News after 15:30 New York cannot be used the same day. These 15 rows are computed by the real code, including the two clock-change weeks."
3. **Method tools.** Type `2023-03-14T19:30:01Z` and press Assign day: one second late goes to the next day. Scroll to the worked label numbers.
4. **Headlines.** Pick a ticker. "Each fake headline has a UTC time, a New York time, an assigned day and a reason. Lexicon bucket and model label are separate columns."
5. **Sentiment.** Switch scorer. "S5 is the five-day average tone. Gaps mean no news, not zero."
6. **Feature explanations.** Pick a day. "These are the actual articles behind this S5 value."
7. **Model comparison.** "The mock model agrees with the lexicon because it is built from it. Real models are listed as not loaded, with the reason."
8. **Experiments.** Start with the two summary cards. "The null world is quiet. In the planted world the main comparison is clearly positive, but one placebo was not passed, so the stored verdict is 'placebo not excluded'. I did not change the rules to get a nicer sentence." Then the ten-seed table: 0 of 10 false alarms, 4 of 10 detections.
9. **Gates.** "This is why there is no real data: each blocked item and what would unlock it."
10. **Methodology and limitations.** Close on what cannot be claimed.

## Likely questions
- *Why not real data?* Licences are unresolved (Gates tab).
- *Why did the placebo not pass?* The baseline was unstable on that seed; see docs/METHODOLOGY.md section 10.
- *Is FinBERT used?* No. Only an adapter exists, tested with fake files.
