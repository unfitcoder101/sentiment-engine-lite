"""SQLite storage (stdlib sqlite3). WAL, foreign keys on, versioned schema, parameterised queries only."""
import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path

from .dedup import normalise

SCHEMA_VERSION = 1
TABLES = ["meta", "symbols", "prices", "articles", "article_symbols", "assignments", "scores", "features",
          "runs", "run_results"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_on TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS symbols (symbol TEXT PRIMARY KEY, market TEXT NOT NULL, data_origin TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prices (
  symbol TEXT NOT NULL REFERENCES symbols(symbol), date TEXT NOT NULL, world TEXT NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, data_origin TEXT NOT NULL,
  PRIMARY KEY (symbol, date, world));
CREATE TABLE IF NOT EXISTS articles (
  article_id INTEGER PRIMARY KEY, headline TEXT NOT NULL,
  first_published TEXT,            -- ONLY the first-published time is stored; no update times
  precision TEXT NOT NULL, kind TEXT, source_name TEXT NOT NULL, data_origin TEXT NOT NULL, licence_note TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS article_symbols (
  article_id INTEGER NOT NULL REFERENCES articles(article_id), symbol TEXT NOT NULL REFERENCES symbols(symbol),
  weight REAL NOT NULL, PRIMARY KEY (article_id, symbol));
CREATE TABLE IF NOT EXISTS assignments (
  article_id INTEGER NOT NULL REFERENCES articles(article_id), symbol TEXT NOT NULL REFERENCES symbols(symbol),
  trading_date TEXT, reason TEXT NOT NULL, kept INTEGER NOT NULL, dedup_group TEXT NOT NULL,
  PRIMARY KEY (article_id, symbol));
CREATE TABLE IF NOT EXISTS scores (
  article_id INTEGER NOT NULL REFERENCES articles(article_id), scorer_id TEXT NOT NULL, version TEXT NOT NULL,
  bucket TEXT NOT NULL, p_pos REAL, p_neg REAL, p_neu REAL, x REAL NOT NULL,
  lm_pos_count INTEGER, lm_neg_count INTEGER, matched_terms TEXT, PRIMARY KEY (article_id, scorer_id));
CREATE TABLE IF NOT EXISTS features (
  symbol TEXT NOT NULL REFERENCES symbols(symbol), date TEXT NOT NULL, scorer_id TEXT NOT NULL,
  s1 REAL, s5 REAL, nvs REAL, abn REAL, has_news INTEGER NOT NULL, n1 INTEGER NOT NULL, n5 INTEGER NOT NULL,
  PRIMARY KEY (symbol, date, scorer_id));
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, mode TEXT NOT NULL, world TEXT NOT NULL, seed INTEGER, verdict TEXT NOT NULL,
  prereg_match INTEGER NOT NULL, data_origin TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run_results (
  run_id TEXT NOT NULL REFERENCES runs(run_id), key TEXT NOT NULL, json TEXT NOT NULL, PRIMARY KEY (run_id, key));
CREATE INDEX IF NOT EXISTS idx_assign_symbol_date ON assignments(symbol, trading_date);
"""


def connect(path):
    """Read-write connection; creates the schema if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR IGNORE INTO schema_version (version, applied_on) VALUES (?, ?)",
                 (SCHEMA_VERSION, date.today().isoformat()))
    conn.commit()
    return conn


def connect_ro(path):
    """Read-only connection (URI mode=ro). Any write raises sqlite3.OperationalError."""
    conn = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def schema_version(conn):
    return conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]


def row_counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}   # names from a fixed list


def get_meta(conn):
    return {r["key"]: json.loads(r["value"]) for r in conn.execute("SELECT key, value FROM meta")}


def dedup_group(symbol, headline, symbols):
    return hashlib.sha256(f"{symbol}|{normalise(headline, symbols)}".encode("utf-8")).hexdigest()[:12]


def _nan(v):
    return None if v is None or v != v else float(v)


def write_demo(path, data, feats, results, robust, meta):
    """Write the whole synthetic demo. `feats` maps scorer_id -> feature DataFrame."""
    path = Path(path)
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    conn = connect(path)
    cal, origin = data["cal"], "SYNTHETIC"
    kept = set(zip(data["assignments"]["id"], data["assignments"]["symbol"]))
    with conn:
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                         [(k, json.dumps(v)) for k, v in dict(meta, robustness=robust).items()])
        conn.executemany("INSERT INTO symbols VALUES (?, ?, ?)", [(s, "US", origin) for s in data["symbols"]])
        for world, bars in data["bars"].items():
            conn.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             [(r.symbol, r.date.isoformat(), world, r.open, r.high, r.low, r.close, origin)
                              for r in bars.itertuples(index=False)])
        arts, links, assigns, scores = [], [], [], []
        for a in data["articles"]:
            arts.append((a["id"], a["headline"], a["published"], a["precision"], a["kind"], "synthetic generator",
                         origin, "not applicable (generated data)"))
            day = a["assigned"].isoformat() if a["assigned"] else None
            for s in a["symbols"]:
                links.append((a["id"], s, 1.0 / len(a["symbols"])))
                assigns.append((a["id"], s, day, a["reason"], int((a["id"], s) in kept),
                                dedup_group(s, a["headline"], a["symbols"])))
            scores.append((a["id"], "demo_lexicon", "demo-1", a["bucket"], None, None, None, a["x"], None, None,
                           json.dumps(a["matched"])))
            p = a["scores_mock"]
            scores.append((a["id"], "mock_model", "mock-1", a["label_mock"], p["positive"], p["negative"], p["neutral"],
                           a["x_mock"], None, None, None))
        conn.executemany("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", arts)
        conn.executemany("INSERT INTO article_symbols VALUES (?, ?, ?)", links)
        conn.executemany("INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)", assigns)
        conn.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", scores)
        for scorer_id, f in feats.items():
            conn.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             [(r.symbol, cal.days[int(r.t)].isoformat(), scorer_id, _nan(r.S1), _nan(r.S5), _nan(r.NVS),
                               _nan(r.ABN), int(r.has_news), int(r.N1), int(r.N5)) for r in f.itertuples(index=False)])
        for r in results.values():
            conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (r["run_id"], r["mode"], r["world"], r["seed"], r["verdict"], int(r["prereg"]["match"]), origin))
            conn.execute("INSERT INTO run_results VALUES (?, ?, ?)", (r["run_id"], "result", json.dumps(r)))
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return path
