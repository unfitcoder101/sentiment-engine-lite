"""Demo lexicon scorer.

The word lists below are a small hand-written DEMO list. They are NOT Loughran-McDonald.
The synthetic headlines are generated from these same words, so the scorer is circular
by construction: it only shows that the software pipeline is wired together.
"""
from .dedup import normalise

POS = ["gains", "growth", "strong", "beat", "upgrade", "record", "surge", "rally", "robust", "expands",
       "improves", "outperform", "boost", "rebound", "upbeat", "wins", "solid", "raises", "optimistic", "recovery"]
NEG = ["loss", "decline", "weak", "miss", "downgrade", "lawsuit", "slump", "selloff", "cuts", "falls",
       "warning", "probe", "default", "layoffs", "recall", "shortfall", "plunge", "delays", "fraud", "bankruptcy"]
_POS, _NEG = set(POS), set(NEG)
BUCKETS = ["positive", "negative", "neutral (zero-net)", "no lexicon coverage"]


def score(headline, symbols=()):
    tokens = normalise(headline, symbols).split()
    pos = [t for t in tokens if t in _POS]
    neg = [t for t in tokens if t in _NEG]
    p, n = len(pos), len(neg)
    if p + n == 0:
        bucket = "no lexicon coverage"
    elif p > n:
        bucket = "positive"
    elif n > p:
        bucket = "negative"
    else:
        bucket = "neutral (zero-net)"
    return {"x": (p - n) / max(1, len(tokens)), "P": p, "N": n, "tokens": len(tokens),
            "bucket": bucket, "matched": pos + neg, "probabilities": None}
