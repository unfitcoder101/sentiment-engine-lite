# Ten-minute walkthrough

**Minute 0-1: the question and the honest answer.** "My project asks whether news sentiment helps rank stocks.
I cannot answer that yet: the news datasets I wanted have unresolved licences. So I built and tested the
instrument first, on SYNTHETIC data, and I can show that it behaves sensibly."

**Minute 1-3: timing is the hard part.** Open Cutoff cases. "If a headline arrives at 15:30:01 New York time, a
signal for that day must not use it. Here are 15 cases computed by the real code: the boundary second, weekends,
Thanksgiving, both clock-change weeks, date-only items, and a timestamp without a zone being rejected."

**Minute 3-4: the label.** Method tools. "I measure the return from the next open to the open ten days later,
scaled by volatility: 101 to 108.9 is 7.82%, divided by 2% volatility gives 3.91. I do not claim this equals a
close-to-close label."

**Minute 4-5: features and scorers.** Feature explanations, then Model comparison. "Four features summarise tone
and volume. The scorers here are a demo word list and a mock model. FinBERT is not loaded; only an adapter exists."

**Minute 5-8: the control experiment.** Experiments. "Two fake worlds share all random draws. In one I inserted a
small link from tone to returns. The rules were written and hashed before the run. Every arm is scored on
identical, hashed rows; 192 checks enforce that.
Result on the main seed: the null world is quiet. In the planted world the sentiment arm beats the baseline by
0.060 IC, Holm-adjusted p 0.0015, and beats 20 shuffled placebos. But the lagged placebo also beat the baseline,
so the stored verdict is 'placebo not excluded'. Across ten pre-declared seeds: no false alarms, and the inserted
link flagged four times."

**Minute 8-9: what I learned.** "With 30 stocks the price-only baseline is unstable, so 'better than baseline' is
a fragile yardstick. A real study would need a larger universe. I kept the result as produced rather than adjust
thresholds after seeing it."

**Minute 9-10: what is blocked and what is next.** Gates. "Each blocked item has an owner and an unlocking step.
Until a data licence and a clean scorer exist, the code refuses to run in confirmatory mode."

One-line description: a tested, explainable prototype with an extensible research framework, run only on synthetic data.
