"""python -m se.cli verify: schema version, row counts, frozen-sample hashes, banner flags."""
import hashlib
import json
import re

from . import api, config, db

ROOT = config.ROOT


def main():
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    st = config.prereg_status()
    check("pre-registration hashes match runs/prereg.json", st["match"], st["reason"])
    db_path = ROOT / "var" / "demo.sqlite"
    if not db_path.is_file():
        check("database var/demo.sqlite exists", False, "run: .venv/bin/python run_demo.py")
        return 1
    conn = db.connect_ro(db_path)
    check("schema version", db.schema_version(conn) == db.SCHEMA_VERSION, str(db.schema_version(conn)))
    counts = db.row_counts(conn)
    check("row counts are non-zero", all(counts[t] > 0 for t in db.TABLES), json.dumps(counts))
    origins = {r[0] for t in ("symbols", "prices", "articles", "runs") for r in conn.execute(f"SELECT DISTINCT data_origin FROM {t}")}
    check("every stored row is flagged SYNTHETIC", origins == {"SYNTHETIC"}, str(sorted(origins)))
    check("banner stored in database", db.get_meta(conn).get("banner") == api.BANNER)
    for run_id, raw in conn.execute("SELECT run_id, json FROM run_results WHERE key = 'result'").fetchall():
        res = json.loads(raw)
        bad = []
        for name, digest in res["samples"]["hashes"].items():
            f = ROOT / "runs" / run_id / "samples" / f"{name}.txt"
            if not f.is_file() or hashlib.sha256(f.read_text().rstrip("\n").encode("utf-8")).hexdigest() != digest:
                bad.append(name)
        check(f"frozen sample files match stored hashes ({run_id})", not bad, ", ".join(bad))
        check(f"abort checks all passed ({run_id})", res["assertions"]["failed"] == 0, f"{res['assertions']['checked']} checks")
    conn.close()
    bundle = json.loads((ROOT / "web" / "data.json").read_text())
    check("every static payload carries banner and SYNTHETIC flag",
          all(v.get("banner") == api.BANNER and v.get("data_origin") == "SYNTHETIC" for v in bundle.values()), f"{len(bundle)} payloads")
    html = (ROOT / "web" / "index.html").read_text()
    check("dashboard has the permanent banner", api.BANNER in html)
    check("dashboard makes no external requests", not re.search(r"https?://", html))
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1
