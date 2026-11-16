# Scorer cards

All experiments in this project use SYNTHETIC headlines. No scorer output is a probability of any price move.

## MOCK MODEL (not FinBERT) - `mock_model`
- Purpose: stand in for a classifier so the pipeline can be exercised end to end.
- How: logits 1.2*P + e1, 1.2*N + e2, 0.8 + e3 (P, N = demo-lexicon counts; e from a hash of the headline, sd 0.5), softmax, x = positive - negative.
- Licence: not applicable. Contamination: not applicable. Allowed use: synthetic demo only.
- Failure modes: it knows nothing about language; it is a noisy copy of the demo lexicon.

## Demo lexicon - `demo_lexicon`
- Purpose: a transparent baseline. 20 positive and 20 negative hand-written words. It is NOT Loughran-McDonald.
- Licence: not applicable. Allowed use: synthetic demo only.
- Failure modes: circular on synthetic headlines (they are built from the same words); no negation, no context.
- Buckets: positive, negative, "neutral (zero-net)", "no lexicon coverage" (never called neutral).

## FinBERT (ProsusAI) - `finbert_prosus` (adapter only, not loaded)
- Licence: CC BY-NC-SA 4.0, non-commercial research (owner's audit, 2026-11-14).
- Contamination: unresolved. Training-data dates are unverified, so overlap with a test period cannot be ruled out.
- Allowed use: exploratory only, never confirmatory. Every output carries "EXPLORATORY ONLY, training-data dates unverified".
- State here: no weights included or downloaded; the adapter loads a local directory only and was tested with fake files.

## finbert-tone - `finbert_tone` (adapter only, blocked)
- Licence: unresolved. Allowed use: blocked. The adapter raises an error before touching any file.
- Its labels are LABEL_0..2; the documented map (neutral, positive, negative) is used only if the owner unblocks it.

## Loughran-McDonald lexicon - `lm_lexicon` (adapter only, not loaded)
- Licence: free for academic use; the owner must supply the CSV locally. Not bundled, downloaded or copied.
- Contamination: the dictionary version is not frozen before 2020, so exploratory only.
- Failure modes: word counts ignore context; headline-length texts often have no coverage.
