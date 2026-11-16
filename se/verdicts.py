"""The single place where verdict texts are generated."""
import re

FORBIDDEN = ("predicts", "edge", "profit", "validated", "significant")
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN) + r")\b", re.IGNORECASE)
MODES = ("synthetic_validation", "exploratory", "confirmatory")
PREFIX = "SYNTHETIC SOFTWARE CHECK: "


def find_forbidden(text):
    """Forbidden whole words found in text (case-insensitive)."""
    return sorted({m.group(1).lower() for m in _FORBIDDEN_RE.finditer(text)})


def verdict_text(mode, world, detected, placebos_passed, criteria_met):
    if mode == "exploratory":
        return "EXPLORATORY: criteria " + ("met" if criteria_met else "not met")
    if mode != "synthetic_validation":
        raise ValueError(f"no verdict text exists for mode {mode!r}")
    if not detected:
        tail = "not detected" if world == "planted" else "no effect flagged"
    elif not placebos_passed:
        tail = "placebo not excluded"
    elif world != "planted":
        tail = "effect flagged in null world (false alarm)"
    else:
        tail = "criteria met" if criteria_met else "planted effect detected"
    return PREFIX + tail
