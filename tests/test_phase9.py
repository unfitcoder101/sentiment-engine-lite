import re
from datetime import date
from pathlib import Path

import pytest

from se import cli, labels, stats, timeutil
from se.timeutil import assign, demo_2023_calendar
from se.verdicts import find_forbidden

ROOT = Path(__file__).resolve().parent.parent
CAL = demo_2023_calendar()
DOCS = ["ARCHITECTURE", "METHODOLOGY", "MODEL_CARD", "LIMITATIONS", "GATES", "SETUP", "DEMO_GUIDE", "WORKED_EXAMPLES",
        "CLAIMS_LEDGER", "GLOSSARY", "PROFESSOR_WALKTHROUGH", "DECISIONS", "PREREGISTRATION_SYNTHETIC"]

# The 14 specified US cases (input, precision, expected day).
US_CASES = [
    ("2023-03-14T19:30:00Z", "timestamp", "2023-03-14"), ("2023-03-14T19:30:01Z", "timestamp", "2023-03-15"),
    ("2023-01-10T20:30:00Z", "timestamp", "2023-01-10"), ("2023-03-18T15:00:00Z", "timestamp", "2023-03-20"),
    ("2023-03-17T21:00:00Z", "timestamp", "2023-03-20"), ("2023-03-14", "date", "2023-03-15"),
    ("2023-03-17", "date", "2023-03-20"), ("2023-11-22T21:00:00Z", "timestamp", "2023-11-24"),
    ("2023-11-23T15:00:00Z", "timestamp", "2023-11-24"), ("2023-01-10T19:30:00Z", "timestamp", "2023-01-10"),
    ("2023-11-03T19:30:00Z", "timestamp", "2023-11-03"), ("2023-11-03T20:00:00Z", "timestamp", "2023-11-06"),
    ("2023-11-06T20:30:00Z", "timestamp", "2023-11-06"), ("2023-11-06T19:30:00Z", "timestamp", "2023-11-06"),
]
# NSE hand cases: Asia/Kolkata is UTC+5:30 all year, cutoff 15:00 local.
NSE_CASES = [
    ("2023-03-14T09:30:00Z", "timestamp", "2023-03-14", "same_day"),       # 15:00 IST exactly: included
    ("2023-03-14T09:30:01Z", "timestamp", "2023-03-15", "after_cutoff"),
    ("2023-03-17T10:00:00Z", "timestamp", "2023-03-20", "after_cutoff"),   # Friday 15:30 IST
    ("2023-03-18T05:00:00Z", "timestamp", "2023-03-20", "weekend"),        # Saturday 10:30 IST
    ("2023-03-13T18:45:00Z", "timestamp", "2023-03-14", "same_day"),       # 00:15 IST on the 14th: local date differs from UTC date
    ("2023-03-14", "date", "2023-03-15", "date_only"),
]


@pytest.mark.parametrize("value,precision,expected", US_CASES)
def test_all_14_us_cases(value, precision, expected):
    assert assign(value, "US", CAL, precision)[0] == date.fromisoformat(expected)


def test_us_case_table_is_complete():
    assert len(US_CASES) == 14
    table = {(c[0], c[1]): c[2] for c in timeutil.CUTOFF_CASES}
    assert all(table[(v, p)] == e for v, p, e in US_CASES) and len(timeutil.CUTOFF_CASES) == 15
    assert all(r["match"] for r in timeutil.run_cutoff_cases())


@pytest.mark.parametrize("value,precision,expected,reason", NSE_CASES)
def test_nse_hand_cases(value, precision, expected, reason):
    assert assign(value, "NSE", CAL, precision) == (date.fromisoformat(expected), reason)


def test_docs_exist_and_use_careful_language():
    assert not (ROOT / "docs" / "METHOD.md").exists()
    for name in DOCS + ["FINAL_REPORT"]:
        assert (ROOT / "docs" / f"{name}.md").is_file(), name
    for path in [ROOT / "README.md", ROOT / "STATUS.md"] + sorted((ROOT / "docs").glob("*.md")):
        text = path.read_text()
        assert find_forbidden(text) == [], path.name
        assert "SYNTHETIC" in text or "synthetic" in text, path.name
    assert "No equivalence with any close-to-close label is claimed" in (ROOT / "docs" / "METHODOLOGY.md").read_text()


def test_worked_examples_numbers_are_reproduced():
    doc = (ROOT / "docs" / "WORKED_EXAMPLES.md").read_text()
    h, l, c = [101, 103, 102, 104], [99, 101, 100, 102], [100, 102, 101, 103]
    values = [f"{labels.atr(h, l, c, 3)[3]:.6f}", f"{labels.atr_pct(h, l, c, 3)[3]:.6f}",
              f"{108.9 / 101 - 1:.6f}", f"{labels.z_exec(108.9 / 101 - 1, 0.02):.6f}"]
    assert values == ["2.666667", "0.025890", "0.078218", "3.910891"]
    for v in values + ["0.03, 0.06, 0.06", "0h, 47h, 49h, 96h"]:
        assert v in doc, v
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    for value, _, expected in US_CASES:
        assert re.search(re.escape(value.replace("T", " ").replace(":00Z", "Z")) + r".*" + expected[5:], doc), value


def test_cli_prereg_status(capsys):
    assert cli.main(["prereg-status"]) == 0 and "MATCH" in capsys.readouterr().out
