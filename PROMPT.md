# SENTIMENT ENGINE LITE: LOW-TOKEN WORKING BUILD FOR CLAUDE CODE

Goal: a small, genuinely runnable, honest prototype I can demo to my professor. A bigger earlier build was deleted, so build the SMALL version below, in ~/dev/sentiment-engine-lite (new, empty folder). It is a separate university project about whether financial-news sentiment helps rank stocks. ALL data is SYNTHETIC. No evidence of predictive value exists. Never claim otherwise.

## TOKEN RULES (I have very few tokens)
- Minimal narration. No plans, no recaps. One short line per stage.
- Write each file in one go. Do not re-read files you just wrote. Do not print large outputs.
- Run pytest once per stage, only the new tests. If a test fails, find the real bug; never edit expected numbers.
- After EACH stage, write STATUS.md (done / next / known issues, max 15 lines) so work can resume if I run out.
- If you run low on budget, finish the current stage cleanly and stop. A working smaller version beats a broken larger one.

## SAFETY
- Work only in ~/dev/sentiment-engine-lite. Never read or touch ~/dev/ai-stock-analyzer-clean or any other folder.
- No network except `pip install`. No credentials, no model weights, no datasets. Bind the server to 127.0.0.1.
- Check `df -h .` first: stop if less than 2 GB free. Keep the venv small.
- `git init` and commit LOCALLY at the end only. Never push. Never `rm -rf` anything outside this folder.
- Never write the words "predicts", "edge", "profit", "validated", "significant" about results. Never call any score a probability.

## SETUP
python3 -m venv .venv ; .venv/bin/pip install numpy pandas scikit-learn pytest ; pin versions into requirements.txt. Python standard library only for everything else (http.server, zoneinfo, sqlite not needed). Add .gitignore (.venv, __pycache__, .pytest_cache).

## LAYOUT
```
README.md  STATUS.md  requirements.txt  .gitignore
se/ __init__.py timeutil.py dedup.py labels.py features.py synthetic.py scorer.py stats.py experiment.py
run_demo.py        # generate data, run everything, write web/data.json (target < 2 minutes)
serve.py           # python http.server on 127.0.0.1:8000 serving web/
web/index.html     # ONE file, inline CSS+JS, no external requests, no CDN
tests/test_*.py
docs/METHOD.md     # max 600 words, plain language
```

## STAGE 1: CORE LOGIC + TESTS
timeutil.py: markets US (America/New_York, close 16:00, cutoff 15:30) and NSE (Asia/Kolkata, close 15:30, cutoff 15:00). TradingCalendar(list of dates): is_trading_day, next_trading_day (strictly after). assign(published_utc, market, calendar, precision): convert UTC to local with zoneinfo (never hard-code offsets). If the local date is a trading day AND local time <= cutoff (exactly at cutoff included, one microsecond later NOT) -> same day. Otherwise next trading day. precision "date" -> always the next trading day. Return (trading_date, reason) with reasons same_day, after_cutoff, weekend, holiday, date_only. Invalid (return None + reason): missing, naive timestamp with no declared zone, after as_of.
Test calendar: Mon-Fri 2023 minus 2023-01-02, 01-16, 02-20, 04-07, 05-29, 06-19, 07-04, 09-04, 11-23, 12-25. Required US test cases:
- 2023-03-14 19:30:00Z -> 2023-03-14 (15:30 EDT, included)
- 2023-03-14 19:30:01Z -> 2023-03-15
- 2023-01-10 20:30:00Z -> 2023-01-10 (15:30 EST, winter offset)
- 2023-03-18 15:00:00Z (Saturday) -> 2023-03-20
- 2023-03-17 21:00:00Z (Friday after cutoff) -> 2023-03-20
- date-only 2023-03-14 -> 2023-03-15 ; date-only 2023-03-17 -> 2023-03-20
- 2023-11-22 21:00:00Z -> 2023-11-24 (Thanksgiving 11-23 closed)
- 2023-11-23 15:00:00Z -> 2023-11-24
- naive timestamp -> invalid.
dedup.py: normalise headline (NFKC, lowercase, punctuation to spaces, collapse spaces, drop the ticker token). Per (symbol, normalised headline): keep the first; drop a later copy if within 48 hours (<=) of the MOST RECENT KEPT copy, else keep it. Test: copies at 0h, 47h, 49h, 96h -> keep 0h and 49h. Multi-ticker article with k symbols -> weight 1/k each.
labels.py (adjusted OHLC per symbol on the market calendar, H=10, ATR window 14, ATR_FLOOR=0.001, MIN_GROUP=20, TOP_SHARE=0.30):
- TR(t) = max(High-Low, |High-Close(t-1)|, |Low-Close(t-1)|); ATR = simple mean of the last 14 TR including t; ATR_pct = ATR/Close(t).
- R_exec(t) = Open(t+11)/Open(t+1) - 1 ; Z_exec(t) = R_exec / max(ATR_pct(t), 0.001) (ATR at signal date t, never t+1).
- L_exec: within each (market, date) group, percentile rank (average ties) of Z_exec; label 1 if pct >= 0.70 - 1e-12; NaN if the group has fewer than 20 stocks or value is NaN.
- Rows missing any needed bar or with non-positive prices get NaN labels.
- Do NOT claim equivalence with any close-to-close label.
Tests (exact numbers): bars (H,L,C): t0 101,99,100; t1 103,101,102; t2 102,100,101; t3 104,102,103 with window 3 -> TR 3,2,3, ATR 2.666667, ATR_pct 0.025890. Open(t+1)=101, Open(t+11)=108.9, ATR_pct 0.02 -> R_exec 0.078218, Z_exec 3.910891. ATR_pct 0.0004 uses floor 0.001. Group of 20 distinct values -> exactly 7 labels; group of 19 -> all NaN; ties [1,2,2,3] -> average ranks [1,2.5,2.5,4].
features.py: for stock s, trading-day index t, articles with score x and weight w assigned to days; M = sum(w*x)/sum(w); N = count of assignments.
- S1 = M over day t ; S5 = M over days t-4..t (missing if none).
- NVS = ln(1+N5) - ln(1+Bn), N5 = N over t-4..t, Bn = 5*N_base/60, N_base = N over t-64..t-5.
- ABN = S5 - pooled article-weighted mean over days t-64..t-5; missing if S5 missing or N_base < 10.
- has_news = 1 if N5 >= 1. First 64 trading days excluded for everyone.
- Per date, per feature: fractional rank (rank-0.5)/n among non-missing stocks (average ties); missing -> 0.5; if fewer than 5 non-missing stocks that date -> all 0.5.
Test: future-deletion invariance (features at day t unchanged if you delete every article first published after day t's cutoff). Plus one hand-computed toy for each feature.

## STAGE 2: SYNTHETIC DATA + SCORER + EXPERIMENT
synthetic.py (seed fixed, e.g. 7): 30 fake tickers SYN01.., business days 2020-01-01 to 2022-12-31 minus a listed holiday set, daily vol 1-3%, Open(t+1) = Close(t)*(1+overnight gap), High/Low consistent. News: 0.5-3 articles/stock/week, hidden tone u~N(0,1), headlines built from my small demo word list (about 20 positive and 20 negative generic finance words YOU write; say clearly it is NOT Loughran-McDonald and is circular by construction). Include after-hours, weekend, date-only, multi-ticker, duplicate and a few invalid timestamps. Two worlds sharing all random draws: "null" (no link) and "planted" (mean u of articles assigned to day t adds beta=0.002 per day drift on days t+1..t+5; half of day t+1's effect goes into the overnight gap). beta = 0.002 is fixed BEFORE running; note in docs/METHOD.md that an earlier larger build used 0.0005, which was too small to detect.
scorer.py: demo lexicon x = (P-N)/max(1,tokens). Buckets: positive (P>N), negative (N>P), "neutral (zero-net)" (P=N>0), "no lexicon coverage" (P+N=0, never called neutral). Probabilities null.
experiment.py: baseline features atr_pct, ret_5, ret_20, vol_10 (close-based, through day t only). Arms: B baseline; C = B + S1,S5,NVS,ABN ranks + has_news; C2 placebo = C with the sentiment block shuffled across stocks within each date (train and test), 20 shuffles fixed seeds. Train 2020-2021, test 2022; training rows need t+11 < first test bar - 5 (purge+embargo). StandardScaler + LogisticRegression(C=1, class_weight balanced), target L_exec. Primary metric: daily Spearman IC of score vs Z_exec (dates with >= 20 stocks), averaged. delta series d_t = IC_C(t) - IC_B(t). stats.py: paired stationary bootstrap, mean block length 20, 2000 resamples, fixed seed, one-sided p = (1+#{centred resampled mean >= observed})/(B+1); unadjusted 95% interval. Placebo C2 passes if observed delta >= max of the 20 shuffled deltas. Verdict text only: "SYNTHETIC SOFTWARE CHECK: planted effect detected" / "not detected" / "placebo not excluded" / "no effect flagged". Tests for stats: bootstrap reproducible from seed, Holm not needed. Run both worlds. Add the cutoff cases (the test table above) computed by the real assign() into the output.
run_demo.py writes web/data.json: meta (synthetic banner, seed, beta), symbols, per-symbol daily series (S1, S5, count) for the planted world, 300 sample headlines with UTC time, local time, assigned day, reason, bucket, matched words, experiment results for both worlds, the cutoff cases.

## STAGE 3: FRONTEND (web/index.html, one file)
Permanent top banner: "Educational research prototype. Data origin: SYNTHETIC. No evidence of predictive value. Not investment advice." System fonts, CSS variables with a dark-mode variant, works at 360 px width, keyboard-focusable, every chart has a table. Tabs:
1 Overview (what/why in plain words, what is built vs planned, blocked items with reasons: FNSPID and Alpaca licences unresolved, FinBERT training dates unverified so exploratory only, Loughran-McDonald not frozen).
2 Sentiment (symbol picker; inline SVG: S5 line, news-count bars; bucket counts with "no lexicon coverage" as its own bar; data table).
3 Headlines (filter by bucket; columns UTC, New York time, assigned day, reason, matched words).
4 Experiments (null and planted side by side: delta IC, p-value, unadjusted interval, 20 placebo values with the observed marked, verdict, a standing note "software check on invented data, not evidence about markets").
5 Cutoff cases (the table from the real code).
serve.py serves web/ at http://127.0.0.1:8000/ . Verify: `node --check` on the extracted script if node exists; if Playwright/Chromium exists take desktop and 360 px screenshots into docs/; otherwise say plainly the layout was not checked in a browser.

## STAGE 4: DOCS AND CLOSE
README.md: exact run commands (`bash`-free: `.venv/bin/python run_demo.py` then `.venv/bin/python serve.py`), what it shows, what it does NOT show. docs/METHOD.md (<= 600 words). `git add -A && git commit` locally (no push). Final output, max 12 lines: tests passed/failed, demo runtime, both world verdicts exactly as produced, whether the frontend was browser-checked, what is NOT built (full list below), the two commands to run.

## EXPLICITLY NOT BUILT IN THIS VERSION (list in README as "planned")
FinBERT / finbert-tone / Loughran-McDonald adapters, SQLite layer and CSV importers, FastAPI service, pre-registration hashing, ablations, power table, confirmatory-mode guard, full docs set.
