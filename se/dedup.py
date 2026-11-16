"""Headline normalisation, 48-hour de-duplication and multi-ticker weights."""
import re
import unicodedata
from datetime import timedelta

WINDOW = timedelta(hours=48)


def normalise(headline, symbols=()):
    s = unicodedata.normalize("NFKC", headline).lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    drop = {x.lower() for x in symbols}
    return " ".join(t for t in s.split() if t not in drop)


def expand_and_dedup(articles):
    """articles: dicts with 'symbols' (list), 'headline', 'ts' (aware datetime).

    Returns (kept, dropped): one row per (article, symbol) with 'symbol' and 'weight' = 1/k.
    A copy is dropped if it is within 48 h (<=) of the most recent KEPT copy for that symbol.
    """
    kept, dropped, last = [], [], {}
    for a in sorted(articles, key=lambda a: a["ts"]):
        syms = a["symbols"]
        norm = normalise(a["headline"], syms)
        for s in syms:
            row = dict(a, symbol=s, weight=1.0 / len(syms))
            prev = last.get((s, norm))
            if prev is not None and a["ts"] - prev <= WINDOW:
                dropped.append(row)
            else:
                last[(s, norm)] = a["ts"]
                kept.append(row)
    return kept, dropped
