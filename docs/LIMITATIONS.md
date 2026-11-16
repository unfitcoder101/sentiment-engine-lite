# Limitations

- **SYNTHETIC data only.** Prices and headlines are invented. No statement about real markets is possible.
- **Circular scoring.** Headlines are generated from the demo word list that the scorers read.
- **Small universe.** 30 tickers and about 490 test days. The price-only baseline is unstable between seeds
  (negative test IC on seed 7), which makes "gain over baseline" a fragile yardstick. The lagged placebo (C3)
  exposed this.
- **Coarse robustness.** Ten seeds give rates in steps of 10%. The planted link was flagged on 4 of 10.
- **Planted strength chosen by the author.** beta = 0.002 was fixed before running, but it is still a choice.
- **Simple costs.** A flat 0.004 per trade; no liquidity, borrowing or capacity.
- **Illustrative calendars.** Weekdays minus a listed holiday set, not exchange calendars; no half days.
- **Adapters untested on real inputs.** FinBERT, finbert-tone and Loughran-McDonald code was run only on fake files.
- **Importers store, nothing more.** No pipeline runs over imported data; importing does not clear a licence.
- **One market tested end to end.** NSE timing has unit tests only.
- **Browser checks are screenshots.** Every tab was looked at in desktop and 360 px; the cutoff-explainer button
  was not clicked in a browser (its API endpoint is tested).
