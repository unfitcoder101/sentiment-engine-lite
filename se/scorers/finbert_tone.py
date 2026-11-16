"""finbert-tone adapter: BLOCKED (licence unresolved) unless the owner changes gates.yaml."""
from .. import gates as gates_mod
from .hf_classifier import LocalHFClassifier

GATE = "scorer:finbert_tone"
FLAG = "EXPLORATORY ONLY, training-data dates unverified"
# Its config uses LABEL_0..2; this map comes from the model-card comment and is used only if unblocked.
OVERRIDE = {"map": {"LABEL_0": "neutral", "LABEL_1": "positive", "LABEL_2": "negative"},
            "source": "finbert-tone model card comment: LABEL_0 neutral, LABEL_1 positive, LABEL_2 negative"}


class FinBERTToneScorer(LocalHFClassifier):
    scorer_id = "finbert_tone"
    flag = FLAG

    def __init__(self, model_dir, gates=None):
        g = gates_mod.gate(gates or gates_mod.load(), GATE) or {}
        if g.get("allowed_use", "blocked") == "blocked" or g.get("licence") == "unresolved":
            # raised before any file is touched
            raise gates_mod.ScorerBlockedError(
                f"{GATE} is blocked: licence {g.get('licence')!r}. Unlock: {g.get('unlock', 'owner decision')}")
        super().__init__(model_dir, override=OVERRIDE)
