"""Bring-your-own-data CSV importers. They never contact a network.

IMPORTING DOES NOT CLEAR A LICENCE. The importer only records the licence note you supply; whether you
may use the data is your responsibility (see gates.yaml, data:user_import).
"""
import csv
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import db

ORIGIN = "user_import"
LICENCE_WARNING = "Importing does NOT clear a licence; the note is recorded as supplied."


class ImporterError(ValueError):
    pass


def _rows(csv_path):
    text = Path(csv_path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
    return cols, list(reader)


def _need(cols, names, what):
    missing = [n for n in names if n not in cols]
    if missing:
        raise ImporterError(f"{what} CSV is missing columns: {missing}")


def import_prices(db_path, csv_path, source_name, market="US"):
    """Columns: symbol, date, open, high, low, close. Rejected rows are counted by reason."""
    if not (source_name or "").strip():
        raise ImporterError("--source-name is required")
    cols, rows = _rows(csv_path)
    _need(cols, ["symbol", "date", "open", "high", "low", "close"], "prices")
    rejected, good = Counter(), []
    for r in rows:
        sym = (r[cols["symbol"]] or "").strip().upper()
        if not sym:
            rejected["missing_symbol"] += 1
            continue
        try:
            d = date.fromisoformat((r[cols["date"]] or "").strip())
        except ValueError:
            rejected["bad_date"] += 1
            continue
        try:
            o, h, l, c = (float(r[cols[k]]) for k in ("open", "high", "low", "close"))
        except (TypeError, ValueError):
            rejected["non_numeric_price"] += 1
            continue
        if min(o, h, l, c) <= 0:
            rejected["non_positive_price"] += 1
        elif h < max(o, c) or l > min(o, c):
            rejected["inconsistent_high_low"] += 1
        else:
            good.append((sym, d.isoformat(), "user", o, h, l, c, ORIGIN))
    conn = db.connect(db_path)
    with conn:
        conn.executemany("INSERT OR IGNORE INTO symbols VALUES (?, ?, ?)", [(g[0], market, ORIGIN) for g in good])
        before = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)", good)
        inserted = conn.total_changes - before
    conn.close()
    rejected["duplicate_row"] += len(good) - inserted
    return {"imported": inserted, "rejected": {k: v for k, v in rejected.items() if v}, "data_origin": ORIGIN,
            "source_name": source_name}


def import_news(db_path, csv_path, source_name, licence_note, source_tz=None, market="US"):
    """Columns: headline, symbol (several separated by ';'), published; optional precision ('date').

    Only the first-published time is accepted: any column whose name contains 'updated' rejects the file.
    Naive timestamps are rejected unless a source time zone is declared.
    """
    if not (source_name or "").strip():
        raise ImporterError("--source-name is required")
    if not (licence_note or "").strip():
        raise ImporterError("--licence-note is required (it is recorded, it does not clear a licence)")
    tz = None
    if source_tz:
        try:
            tz = ZoneInfo(source_tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ImporterError(f"unknown source time zone {source_tz!r}") from exc
    cols, rows = _rows(csv_path)
    updated = [c for c in cols if "updated" in c]
    if updated:
        raise ImporterError(f"columns containing 'updated' are not accepted: {updated}. "
                            "Supply only the first-published time.")
    _need(cols, ["headline", "symbol", "published"], "news")
    rejected, good = Counter(), []
    for r in rows:
        headline = (r[cols["headline"]] or "").strip()
        syms = [s.strip().upper() for s in (r[cols["symbol"]] or "").split(";") if s.strip()]
        raw = (r[cols["published"]] or "").strip()
        precision = (r[cols["precision"]] or "").strip().lower() if "precision" in cols else "timestamp"
        if not headline:
            rejected["missing_headline"] += 1
        elif not syms:
            rejected["missing_symbol"] += 1
        elif not raw:
            rejected["missing_timestamp"] += 1
        elif precision == "date":
            try:
                good.append((headline, syms, date.fromisoformat(raw[:10]).isoformat(), "date"))
            except ValueError:
                rejected["unparseable_timestamp"] += 1
        else:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                rejected["unparseable_timestamp"] += 1
                continue
            if dt.tzinfo is None:
                if tz is None:
                    rejected["naive_timestamp_no_source_tz"] += 1
                    continue
                dt = dt.replace(tzinfo=tz)
            good.append((headline, syms, dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "timestamp"))
    conn = db.connect(db_path)
    with conn:
        for headline, syms, published, precision in good:
            conn.executemany("INSERT OR IGNORE INTO symbols VALUES (?, ?, ?)", [(s, market, ORIGIN) for s in syms])
            cur = conn.execute(
                "INSERT INTO articles (headline, first_published, precision, kind, source_name, data_origin, licence_note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (headline, published, precision, "user", source_name, ORIGIN, licence_note))
            conn.executemany("INSERT OR IGNORE INTO article_symbols VALUES (?, ?, ?)",
                             [(cur.lastrowid, s, 1.0 / len(set(syms))) for s in set(syms)])
    conn.close()
    return {"imported": len(good), "rejected": dict(rejected), "data_origin": ORIGIN, "source_name": source_name,
            "licence_note": licence_note, "warning": LICENCE_WARNING}


def cli(args):
    try:
        if args.cmd == "import-prices":
            rep = import_prices(args.db, args.csv, args.source_name, args.market)
        else:
            rep = import_news(args.db, args.csv, args.source_name, args.licence_note, args.source_tz, args.market)
    except ImporterError as exc:
        print(f"import refused: {exc}")
        return 2
    print(f"imported {rep['imported']} rows into {args.db} (origin {rep['data_origin']})")
    for reason, n in sorted(rep["rejected"].items()):
        print(f"  rejected {n}: {reason}")
    print(LICENCE_WARNING)
    return 0
