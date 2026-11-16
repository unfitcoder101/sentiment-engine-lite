"""Local-directory sequence classifier adapter. Nothing is ever downloaded.

torch and transformers are optional and imported lazily; they are NOT installed by this project.
"""
import json
import math
import os
from pathlib import Path

ROLES = ("positive", "negative", "neutral")
_ALIASES = {"positive": "positive", "pos": "positive", "negative": "negative", "neg": "negative",
            "neutral": "neutral", "neu": "neutral"}


class AmbiguousLabelMap(ValueError):
    pass


class MissingDependencyError(RuntimeError):
    pass


def resolve_label_map(id2label, override=None):
    """Return {role: class index}. Names must clearly be positive/negative/neutral, or a documented override is needed."""
    if not id2label:
        raise AmbiguousLabelMap("config.json has no id2label")
    names = {int(k): str(v) for k, v in id2label.items()}
    if override is not None:
        if not str(override.get("source", "")).strip():
            raise AmbiguousLabelMap("a label-map override must document its source")
        roles = {names_key: role for names_key, role in override["map"].items()}
        out = {role: idx for idx, name in names.items() for key, role in roles.items() if key == name}
    else:
        out = {}
        for idx, name in names.items():
            role = _ALIASES.get(name.strip().lower())
            if role is None or role in out:
                raise AmbiguousLabelMap(
                    f"label names {sorted(names.values())} are not clearly positive/negative/neutral; "
                    "supply a documented override or do not use this model")
            out[role] = idx
    if set(out) != set(ROLES) or len(set(out.values())) != 3:
        raise AmbiguousLabelMap(f"label map does not cover exactly positive/negative/neutral: {out}")
    return out


def softmax(logits):
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    return [v / sum(ex) for v in ex]


class LocalHFClassifier:
    scorer_id = "hf_local"
    flag = None

    def __init__(self, model_dir, override=None):
        self.model_dir = Path(model_dir)
        cfg_path = self.model_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"no local model at {str(self.model_dir)!r} (config.json not found). This project loads a LOCAL "
                "directory only and never downloads weights or tokenizers.")
        self.config = json.loads(cfg_path.read_text())
        self.label_map = resolve_label_map(self.config.get("id2label"), override)
        self._pipeline = None

    def _load(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise MissingDependencyError(
                "torch and transformers are optional and not installed. Install them yourself only if you hold "
                "local model weights under their licence; this project never downloads either.") from exc
        tok = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
        model.eval()
        self._pipeline = (torch, tok, model)

    def format_output(self, class_scores):
        """Map per-class output scores (softmax) to the common record. Pure function, testable without torch."""
        p = {role: float(class_scores[idx]) for role, idx in self.label_map.items()}
        label = max(p, key=p.get)
        out = {"scorer_id": self.scorer_id, "x": p["positive"] - p["negative"], "label": label, "output_scores": p,
               "note": "Classifier output scores. Not probabilities of any price move."}
        if self.flag:
            out["flag"] = self.flag
        return out

    def score(self, headline):
        if self._pipeline is None:
            self._load()
        torch, tok, model = self._pipeline
        with torch.no_grad():
            logits = model(**tok(headline, return_tensors="pt", truncation=True)).logits[0].tolist()
        return self.format_output(softmax(logits))
