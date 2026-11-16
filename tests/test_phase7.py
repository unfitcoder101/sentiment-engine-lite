import json
import socket
import sqlite3

import pytest

from se import cli, db, importers, study, synthetic


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    data = synthetic.generate()
    n = len(data["cal"].days)
    feats = {s: study.scorer_feats(data["assignments"], data["symbols"], n, s) for s in ("mock_model", "demo_lexicon")}
    results = {"null": study.run_study(data, "null", write=False, light=True)}
    path = db.write_demo(tmp_path_factory.mktemp("db") / "demo.sqlite", data, feats, results, {"rows": []},
                         {"data_origin": "SYNTHETIC", "seed": data["seed"]})
    return path, data, feats, results


def test_round_trip(demo):
    path, data, feats, results = demo
    conn = db.connect_ro(path)
    counts = db.row_counts(conn)
    assert db.schema_version(conn) == db.SCHEMA_VERSION == 1
    assert counts["symbols"] == 30 and counts["prices"] == 2 * 30 * len(data["cal"].days)
    assert counts["articles"] == len(data["articles"]) and counts["scores"] == 2 * len(data["articles"])
    assert counts["article_symbols"] == counts["assignments"] == sum(len(a["symbols"]) for a in data["articles"])
    assert counts["features"] == sum(len(f) for f in feats.values()) and counts["runs"] == 1
    kept = conn.execute("SELECT COUNT(*) FROM assignments WHERE kept = 1").fetchone()[0]
    assert kept == len(data["assignments"])
    a = next(x for x in data["articles"] if x["kind"] == "multi-ticker" and x["assigned"])
    row = conn.execute("SELECT * FROM articles WHERE article_id = ?", (a["id"],)).fetchone()
    assert (row["headline"], row["first_published"], row["data_origin"]) == (a["headline"], a["published"], "SYNTHETIC")
    w = [r["weight"] for r in conn.execute("SELECT weight FROM article_symbols WHERE article_id = ?", (a["id"],))]
    assert sum(w) == pytest.approx(1.0) and len(w) == len(a["symbols"])
    s = {r["scorer_id"]: r for r in conn.execute("SELECT * FROM scores WHERE article_id = ?", (a["id"],))}
    assert s["demo_lexicon"]["p_pos"] is None and s["demo_lexicon"]["x"] == pytest.approx(a["x"])
    assert s["mock_model"]["p_pos"] + s["mock_model"]["p_neg"] + s["mock_model"]["p_neu"] == pytest.approx(1.0)
    f = feats["mock_model"].iloc[1234]
    got = conn.execute("SELECT s5, n5 FROM features WHERE symbol = ? AND date = ? AND scorer_id = 'mock_model'",
                       (f.symbol, data["cal"].days[int(f.t)].isoformat())).fetchone()
    assert got["n5"] == int(f.N5) and (got["s5"] is None) == (f.S5 != f.S5)
    stored = json.loads(conn.execute("SELECT json FROM run_results").fetchone()[0])
    assert stored["verdict"] == results["null"]["verdict"] and db.get_meta(conn)["data_origin"] == "SYNTHETIC"
    invalid = conn.execute("SELECT COUNT(*) FROM assignments WHERE trading_date IS NULL AND kept = 0").fetchone()[0]
    assert invalid >= sum(1 for x in data["articles"] if x["assigned"] is None)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(articles)")]
    assert "first_published" in cols and not any("updated" in c for c in cols)
    conn.close()


def test_read_only_connection_refuses_writes(demo):
    conn = db.connect_ro(demo[0])
    for sql in ("INSERT INTO meta (key, value) VALUES ('x', '1')", "DELETE FROM articles", "DROP TABLE prices",
                "UPDATE symbols SET market = 'X'"):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute(sql)
    conn.close()


def test_wal_and_foreign_keys(tmp_path):
    conn = db.connect(tmp_path / "x.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO prices VALUES ('NOPE', '2023-01-03', 'user', 1, 1, 1, 1, 'user_import')")
    conn.close()


NEWS = """headline,symbol,published,precision
Aware timestamp,AAA,2023-03-14T19:30:00Z,
Naive timestamp,AAA,2023-03-14 15:30:00,
,AAA,2023-03-14T19:30:00Z,
No symbol,,2023-03-14T19:30:00Z,
No time,AAA,,
Garbage time,AAA,next tuesday,
Two tickers,AAA;BBB,2023-03-15T10:00:00+00:00,
Date only,BBB,2023-03-16,date
"""


def test_news_importer_rejections(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", no_network)
    csv_path, dbp = tmp_path / "news.csv", tmp_path / "user.sqlite"
    csv_path.write_text(NEWS)
    for kwargs in (dict(source_name="", licence_note="ok"), dict(source_name="me", licence_note=" "),
                   dict(source_name="me", licence_note=None)):
        with pytest.raises(importers.ImporterError, match="required"):
            importers.import_news(dbp, csv_path, **kwargs)
    rep = importers.import_news(dbp, csv_path, "my export", "personal notes, no redistribution")
    assert rep["imported"] == 3 and rep["data_origin"] == "user_import" and "NOT clear" in rep["warning"]
    assert rep["rejected"] == {"naive_timestamp_no_source_tz": 1, "missing_headline": 1, "missing_symbol": 1,
                               "missing_timestamp": 1, "unparseable_timestamp": 1}
    conn = db.connect_ro(dbp)
    rows = conn.execute("SELECT * FROM articles ORDER BY article_id").fetchall()
    assert {r["data_origin"] for r in rows} == {"user_import"} and rows[0]["licence_note"].startswith("personal")
    assert [r["first_published"] for r in rows] == ["2023-03-14T19:30:00Z", "2023-03-15T10:00:00Z", "2023-03-16"]
    assert conn.execute("SELECT COUNT(*) FROM article_symbols").fetchone()[0] == 4
    conn.close()
    rep2 = importers.import_news(tmp_path / "tz.sqlite", csv_path, "my export", "note", source_tz="America/New_York")
    assert rep2["imported"] == 4 and "naive_timestamp_no_source_tz" not in rep2["rejected"]
    conn = db.connect_ro(tmp_path / "tz.sqlite")
    assert conn.execute("SELECT first_published FROM articles WHERE headline = 'Naive timestamp'").fetchone()[0] == "2023-03-14T19:30:00Z"
    conn.close()
    bad = tmp_path / "upd.csv"
    bad.write_text("headline,symbol,published,last_updated_at\nx,AAA,2023-03-14T19:30:00Z,2023-03-15T00:00:00Z\n")
    with pytest.raises(importers.ImporterError, match="updated"):
        importers.import_news(dbp, bad, "me", "note")
    with pytest.raises(importers.ImporterError, match="time zone"):
        importers.import_news(dbp, csv_path, "me", "note", source_tz="Mars/Olympus")


def test_price_importer_and_cli(tmp_path, capsys):
    p = tmp_path / "prices.csv"
    p.write_text("symbol,date,open,high,low,close\nAAA,2023-01-03,10,11,9,10.5\nAAA,2023-01-03,10,11,9,10.5\n"
                 "AAA,2023-13-01,10,11,9,10\nAAA,2023-01-04,10,11,9,0\nAAA,2023-01-05,10,9,8,9.5\nAAA,2023-01-06,a,b,c,d\n")
    rep = importers.import_prices(tmp_path / "u.sqlite", p, "broker export")
    assert rep["imported"] == 1 and rep["rejected"] == {"duplicate_row": 1, "bad_date": 1, "non_positive_price": 1,
                                                        "inconsistent_high_low": 1, "non_numeric_price": 1}
    news = tmp_path / "n.csv"
    news.write_text("headline,symbol,published\nHello,AAA,2023-03-14T19:30:00Z\n")
    assert cli.main(["import-news", str(news), "--db", str(tmp_path / "u.sqlite")]) == 2
    assert "refused" in capsys.readouterr().out
    assert cli.main(["import-news", str(news), "--db", str(tmp_path / "u.sqlite"), "--source-name", "me",
                     "--licence-note", "mine"]) == 0
    assert "does NOT clear a licence" in capsys.readouterr().out
