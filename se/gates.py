"""Loader for gates.yaml and the confirmatory-mode guard."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class ConfirmatoryGateError(RuntimeError):
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("confirmatory run refused: " + "; ".join(self.reasons))


class ScorerBlockedError(RuntimeError):
    pass


def load(path=None):
    return yaml.safe_load(Path(path or ROOT / "gates.yaml").read_text())


def gate(gates, key):
    return (gates.get("gates") or {}).get(key)


def confirmatory_reasons(gates, data_key, scorer_key, prereg_match, frozen_samples_exist):
    """Every reason a confirmatory run must be refused (empty list = permitted)."""
    reasons = []
    d, s, st = gate(gates, data_key), gate(gates, scorer_key), gate(gates, "study:confirmatory")
    if d is None:
        reasons.append(f"{data_key}: no gate entry")
    elif d.get("licence") != "verified_permitted":
        reasons.append(f"{data_key}: data licence is {d.get('licence')!r}, needs 'verified_permitted'")
    if s is None:
        reasons.append(f"{scorer_key}: no gate entry")
    elif s.get("contamination") != "verified_clear":
        reasons.append(f"{scorer_key}: contamination is {s.get('contamination')!r}, needs 'verified_clear'")
    if st is None or st.get("status") != "allowed":
        reasons.append(f"study:confirmatory: status is {(st or {}).get('status')!r}, needs 'allowed'")
    if not prereg_match:
        reasons.append("pre-registration: no matching frozen prereg")
    if not frozen_samples_exist:
        reasons.append("frozen samples: sample files not present")
    return reasons


def check_confirmatory(gates, data_key, scorer_key, prereg_match, frozen_samples_exist):
    reasons = confirmatory_reasons(gates, data_key, scorer_key, prereg_match, frozen_samples_exist)
    if reasons:
        raise ConfirmatoryGateError(reasons)
    return True


def table(gates):
    rows = [("gate", "status", "licence", "contamination", "allowed use", "unlock")]
    for key, g in gates["gates"].items():
        rows.append((key, g.get("status", ""), g.get("licence", "-"), g.get("contamination", "-"),
                     g.get("allowed_use", "-"), g.get("unlock", "")))
    widths = [max(len(str(r[i])) for r in rows) for i in range(5)]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(r[:5], widths)) + "  " + str(r[5]) for r in rows]
    a = gates.get("audit", {})
    return f"Source: {a.get('source')} dated {a.get('date')}\n" + "\n".join(lines)
