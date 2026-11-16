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


# ---------- MOCK MODEL (not FinBERT) ----------
import hashlib
import math

MOCK_ID = "mock_model"
MOCK_LABEL = "MOCK MODEL (not FinBERT)"
MOCK_LABELS = ["positive", "negative", "neutral"]
MOCK_NOISE_SD = 0.5
MOCK_NOTE = "Softmax output scores of a deterministic fake model. Not probabilities of any price move."


def _hash_normals(text, n=3):
    """n standard normal draws fixed by the SHA-256 of the text (Box-Muller on hash bytes)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(n):
        u1 = (int.from_bytes(h[8 * i:8 * i + 4], "big") + 1) / 2 ** 32
        u2 = int.from_bytes(h[8 * i + 4:8 * i + 8], "big") / 2 ** 32
        out.append(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
    return out


def mock_model_score(headline, symbols=()):
    base, e = score(headline, symbols), _hash_normals(headline)
    logits = [1.2 * base["P"] + MOCK_NOISE_SD * e[0], 1.2 * base["N"] + MOCK_NOISE_SD * e[1], 0.8 + MOCK_NOISE_SD * e[2]]
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    p = [v / sum(ex) for v in ex]
    label = MOCK_LABELS[p.index(max(p))]
    return {"scorer_id": MOCK_ID, "scorer_label": MOCK_LABEL, "x": p[0] - p[1], "label": label, "bucket": label,
            "output_scores": {"positive": p[0], "negative": p[1], "neutral": p[2]}, "note": MOCK_NOTE}
