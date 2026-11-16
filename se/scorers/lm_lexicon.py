"""Loughran-McDonald lexicon adapter.

The dictionary is NEVER bundled, downloaded or copied by this project. The owner supplies a local CSV
under the dictionary's own terms. Status in gates.yaml: exploratory only (version not frozen pre-2020).
"""
import csv
import hashlib
import re
from pathlib import Path

SCORER_ID = "lm_lexicon"
FLAG = "EXPLORATORY ONLY, dictionary version not frozen before the test period"
_TOKEN = re.compile(r"[a-z]+")


class LexiconMissingError(RuntimeError):
    pass


def tokenise(text):
    """Lowercase, letters only, whole tokens."""
    return _TOKEN.findall(text.lower())


class LMLexicon:
    def __init__(self, path):
        if not path or not Path(path).is_file():
            raise LexiconMissingError(
                f"Loughran-McDonald CSV not found at {path!r}. This project does not include or download it. "
                "Obtain the master dictionary yourself under its terms, keep it outside the repository, and pass "
                "its local path. Required columns: Word, Positive, Negative.")
        raw = Path(path).read_bytes()
        self.version = "sha256:" + hashlib.sha256(raw).hexdigest()[:16]
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
        cols = {c.strip().lower(): c for c in (rows[0].keys() if rows else [])}
        missing = [c for c in ("word", "positive", "negative") if c not in cols]
        if missing:
            raise ValueError(f"lexicon CSV is missing columns: {missing} (need Word, Positive, Negative)")
        self.positive, self.negative = set(), set()
        for r in rows:
            word = r[cols["word"]].strip().lower()
            # positive number = member (year added); negative number = removed; 0 = not a member
            if self._num(r[cols["positive"]]) > 0:
                self.positive.add(word)
            if self._num(r[cols["negative"]]) > 0:
                self.negative.add(word)

    @staticmethod
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def score(self, headline):
        tokens = tokenise(headline)
        pos = [t for t in tokens if t in self.positive]
        neg = [t for t in tokens if t in self.negative]
        p, n = len(pos), len(neg)
        if p + n == 0:
            bucket = "no lexicon coverage"
        elif p > n:
            bucket = "positive"
        elif n > p:
            bucket = "negative"
        else:
            bucket = "neutral (zero-net)"
        return {"scorer_id": SCORER_ID, "version": self.version, "x": (p - n) / max(1, len(tokens)),
                "lm_pos_count": p, "lm_neg_count": n, "tokens": len(tokens), "bucket": bucket,
                "matched": pos + neg, "probabilities": None, "flag": FLAG}
