"""FinBERT (ProsusAI) adapter: EXPLORATORY ONLY. Local directory only; never downloaded."""
from .. import gates as gates_mod
from .hf_classifier import LocalHFClassifier

GATE = "scorer:finbert_prosus"
FLAG = "EXPLORATORY ONLY, training-data dates unverified"


class FinBERTScorer(LocalHFClassifier):
    scorer_id = "finbert_prosus"
    flag = FLAG       # carried by every output

    def __init__(self, model_dir, gates=None):
        g = gates_mod.gate(gates or gates_mod.load(), GATE) or {}
        if g.get("allowed_use") != "exploratory":
            raise gates_mod.ScorerBlockedError(f"{GATE}: allowed_use is {g.get('allowed_use')!r}; exploratory use only")
        super().__init__(model_dir)
