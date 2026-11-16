"""Frozen run settings (configs/profile.yaml) and pre-registration hashing."""
import hashlib
import json
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_REL = "configs/profile.yaml"
PREREG_DOC_REL = "docs/PREREGISTRATION_SYNTHETIC.md"
PREREG_REL = "runs/prereg.json"


class ConfigError(RuntimeError):
    pass


class PreregMismatchError(RuntimeError):
    pass


def load(root=ROOT):
    return yaml.safe_load((Path(root) / CONFIG_REL).read_text())


def check_against_code(cfg):
    """The yaml is the record; the code constants must agree with it."""
    from . import experiment, features, labels, synthetic
    pairs = [
        ("labels.horizon", cfg["labels"]["horizon"], labels.H),
        ("labels.atr_window", cfg["labels"]["atr_window"], labels.ATR_WINDOW),
        ("labels.atr_floor", cfg["labels"]["atr_floor"], labels.ATR_FLOOR),
        ("labels.min_group", cfg["labels"]["min_group"], labels.MIN_GROUP),
        ("labels.top_share", cfg["labels"]["top_share"], labels.TOP_SHARE),
        ("features.warmup", cfg["features"]["warmup"], features.WARMUP),
        ("features.base_window", cfg["features"]["base_window"], features.BASE),
        ("features.min_base", cfg["features"]["min_base"], features.MIN_BASE),
        ("features.min_rank", cfg["features"]["min_rank"], features.MIN_RANK),
        ("metrics.min_stocks_p1", cfg["metrics"]["min_stocks_p1"], experiment.MIN_IC_N),
        ("data.seed", cfg["data"]["seed"], synthetic.SEED),
        ("data.beta", cfg["data"]["beta"], synthetic.BETA),
        ("data.n_symbols", cfg["data"]["n_symbols"], synthetic.N_SYMBOLS),
    ]
    bad = [f"{name}: yaml {a!r} != code {b!r}" for name, a, b in pairs if a != b]
    if bad:
        raise ConfigError("config and code disagree: " + "; ".join(bad))


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_hashes(root=ROOT):
    root = Path(root)
    return {"config_sha256": _sha_file(root / CONFIG_REL), "prereg_doc_sha256": _sha_file(root / PREREG_DOC_REL)}


def prereg_freeze(root=ROOT):
    root = Path(root)
    rec = dict(current_hashes(root), config=CONFIG_REL, prereg_doc=PREREG_DOC_REL, frozen_on=date.today().isoformat())
    path = root / PREREG_REL
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def prereg_status(root=ROOT):
    """{'match': bool, 'reason': str, hashes...}; never raises."""
    path = Path(root) / PREREG_REL
    if not path.exists():
        return {"match": False, "reason": "runs/prereg.json missing (run: python -m se.cli prereg-freeze)"}
    frozen, now = json.loads(path.read_text()), current_hashes(root)
    diff = [k for k in now if frozen.get(k) != now[k]]
    return {"match": not diff, "reason": "hashes match" if not diff else "changed since freeze: " + ", ".join(diff),
            "frozen_on": frozen.get("frozen_on"), **now}
