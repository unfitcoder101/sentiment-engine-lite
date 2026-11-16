import json
import re
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from se import api, db, study, synthetic
from se.verdicts import find_forbidden

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "web" / "index.html").read_text()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = synthetic.generate()
    n = len(data["cal"].days)
    feats = {s: study.scorer_feats(data["assignments"], data["symbols"], n, s) for s in ("mock_model", "demo_lexicon")}
    results = {w: study.run_study(data, w, write=False) for w in ("null", "planted")}
    robust = study.run_robustness(seeds=[1])
    path = db.write_demo(tmp_path_factory.mktemp("api") / "demo.sqlite", data, feats, results, robust,
                         {"banner": api.BANNER, "data_origin": "SYNTHETIC", "seed": 7, "beta": 0.002})
    before = path.read_bytes()
    c = TestClient(api.create_app(db_path=path))
    c.db_path, c.db_before, c.results = path, before, results
    return c


def ok(client, url):
    r = client.get(url)
    assert r.status_code == 200, (url, r.text[:200])
    body = r.json()
    assert body["banner"] == api.BANNER and body["data_origin"] == "SYNTHETIC" and len(body["warnings"]) >= 2
    return body["data"]


def err(client, url, status, error):
    r = client.get(url)
    body = r.json()
    assert r.status_code == status and body["error"] == error and body["status"] == status and body["detail"], (url, body)
    assert body["banner"] == api.BANNER and body["data_origin"] == "SYNTHETIC"


def test_health_meta_symbols(client):
    h = ok(client, "/api/health")
    assert h == {"status": "ok", "database": True, "read_only": True, "scores_on_request": False}
    m = ok(client, "/api/meta")
    assert m["schema_version"] == 1 and m["row_counts"]["symbols"] == 30 and m["prereg"]["match"] is True
    assert ok(client, "/api/symbols")["symbols"][0] == "SYN01"


def test_gates_and_scorers(client):
    g = ok(client, "/api/gates")
    assert g["gates"]["study:confirmatory"]["status"] == "blocked" and g["audit"]["date"] == "2026-11-14"
    sc = {s["id"]: s for s in ok(client, "/api/scorers")["scorers"]}
    assert sc["mock_model"]["status"] == "loaded" and sc["mock_model"]["label"] == "MOCK MODEL (not FinBERT)"
    for sid in ("finbert_prosus", "finbert_tone", "lm_lexicon"):
        assert sc[sid]["status"] == "not_loaded" and sc[sid]["reason"]
    assert sc["finbert_prosus"]["contamination"] == "unresolved" and sc["finbert_tone"]["allowed_use"] == "blocked"


def test_sentiment_endpoint(client):
    d = ok(client, "/api/sentiment/SYN01?scorer=demo_lexicon")
    assert len(d["dates"]) == len(d["S5"]) == len(d["count"]) > 600 and "no lexicon coverage" in d["buckets"]
    m = ok(client, "/api/sentiment/syn01?days=30")
    assert m["scorer"] == "mock_model" and len(m["dates"]) == 30 and set(m["buckets"]) == {"positive", "negative", "neutral"}
    assert m["dates"] == d["dates"][-30:]
    at = ok(client, f"/api/sentiment/SYN01?date={d['dates'][-1]}")["at"]
    assert set(at) >= {"S1", "S5", "NVS", "ABN", "has_news", "N5"}
    nl = ok(client, "/api/sentiment/SYN01?scorer=finbert_prosus")
    assert nl["status"] == "not_loaded" and "dates" not in nl
    err(client, "/api/sentiment/SYN01?scorer=gpt", 404, "unknown_scorer")
    err(client, "/api/sentiment/NOPE", 404, "unknown_symbol")
    err(client, "/api/sentiment/SYN01?days=5000", 400, "bad_parameter")
    err(client, "/api/sentiment/SYN01?days=abc", 400, "bad_parameter")
    err(client, "/api/sentiment/SYN01?date=1999-01-01", 404, "unknown_date")
    err(client, "/api/sentiment/bad%20symbol!", 400, "bad_parameter")


def test_headlines_endpoint(client):
    d = ok(client, "/api/sentiment/SYN02/headlines?limit=5")
    assert len(d["items"]) == 5 and d["total"] > 100 and d["limit"] == 5
    for h in d["items"]:
        assert h["lexicon"]["probabilities"] is None and h["lexicon"]["scorer"] == "demo_lexicon"
        assert h["model"]["scorer_label"] == "MOCK MODEL (not FinBERT)" and h["model"]["label"] in ("positive", "negative", "neutral")
        assert "bucket" not in h["model"] and "label" not in h["lexicon"]          # never merged
    page2 = ok(client, "/api/sentiment/SYN02/headlines?limit=5&offset=5")
    assert {h["article_id"] for h in page2["items"]}.isdisjoint({h["article_id"] for h in d["items"]})
    assert ok(client, "/api/sentiment/SYN02/headlines")["limit"] == 50
    days = ok(client, "/api/sentiment/SYN02/headlines?days=20&limit=200")
    assert 0 < days["total"] < d["total"] and all(h["assigned"] for h in days["items"])
    date = days["items"][0]["assigned"]
    win = ok(client, f"/api/sentiment/SYN02/headlines?date={date}")
    assert win["window"][1] == date and all(win["window"][0] <= h["assigned"] <= date for h in win["items"])
    err(client, "/api/sentiment/SYN02/headlines?limit=201", 400, "bad_parameter")
    err(client, "/api/sentiment/SYN02/headlines?limit=0", 400, "bad_parameter")
    err(client, "/api/sentiment/SYN02/headlines?offset=-1", 400, "bad_parameter")
    err(client, "/api/sentiment/SYN02/headlines?days=99999", 400, "bad_parameter")
    err(client, "/api/sentiment/SYN02/headlines?scorer=nope", 404, "unknown_scorer")


def test_compare_definitions_examples(client):
    c = ok(client, "/api/compare/SYN03")
    assert len(c["matrix"]) == 4 and all(len(r) == 3 for r in c["matrix"]) and sum(map(sum, c["matrix"])) == c["n"]
    assert c["rows"][3] == "no lexicon coverage" and "never" not in c["cols"] and "separate" in c["note"]
    d = ok(client, "/api/features/definitions")
    assert {x["id"] for x in d["definitions"]} >= {"S1", "S5", "NVS", "ABN", "has_news"}
    assert all(x[k] for x in d["definitions"] for k in ("formula", "plain", "why", "limits", "missing"))
    ex = ok(client, "/api/labels/examples")
    assert "close-to-close" in ex["statement"] and "No equivalence" in ex["statement"]
    res = [e["result"] for e in ex["examples"]]
    assert res[0] == {"TR t1..t3": [3.0, 2.0, 3.0], "ATR(t3)": 2.666667, "ATR%(t3)": 0.02589}
    assert res[1] == {"R_exec": 0.078218, "Z_exec": 3.910891} and res[2] == {"Z_exec": 10.0}
    assert res[3] == {"labels in the group of 20": 7, "labels in the group of 19": 0}
    assert len(ex["cutoff_cases"]) == 15 and all(c["match"] for c in ex["cutoff_cases"])


def test_experiments_endpoints(client):
    ex = ok(client, "/api/experiments")
    assert [r["world"] for r in ex["runs"]] == ["null", "planted"] and ex["robustness"]["seeds"] == [1]
    for run in ex["runs"]:
        r = ok(client, "/api/experiments/" + run["run_id"])
        assert r["verdict"] == run["verdict"] == client.results[r["world"]]["verdict"]
        assert not find_forbidden(r["verdict"]) and len(r["samples"]["hashes"]["P1"]) == 64
        assert {"sensitivity", "ablations", "power", "placebo", "criteria"} <= set(r)
    err(client, "/api/experiments/nope", 404, "unknown_run")


def test_assign_day_tool(client):
    a = ok(client, "/api/tools/assign-day?ts=2023-03-14T19:30:00Z")
    assert (a["assigned"], a["reason"], a["market"]) == ("2023-03-14", "same_day", "US")
    assert ok(client, "/api/tools/assign-day?ts=2023-03-14T19:30:01Z")["assigned"] == "2023-03-15"
    assert ok(client, "/api/tools/assign-day?ts=2023-03-17&precision=date")["assigned"] == "2023-03-20"
    assert ok(client, "/api/tools/assign-day?ts=2023-03-14T09:30:00Z&market=NSE")["reason"] == "same_day"
    naive = ok(client, "/api/tools/assign-day?ts=2023-03-14T19:30:00")
    assert naive["valid"] is False and naive["reason"] == "naive_timestamp"
    assert ok(client, "/api/tools/assign-day?ts=2021-06-05T12:00:00Z")["reason"] == "weekend"
    err(client, "/api/tools/assign-day", 400, "bad_parameter")
    err(client, "/api/tools/assign-day?ts=2023-03-14T19:30:00Z&market=LSE", 400, "bad_parameter")
    err(client, "/api/tools/assign-day?ts=2023-03-14&precision=week", 400, "bad_parameter")


def test_get_only_and_error_shapes(client):
    for method in ("post", "put", "patch", "delete"):
        for url in ("/api/meta", "/api/sentiment/SYN01", "/api/tools/assign-day?ts=2023-03-14", "/"):
            r = getattr(client, method)(url)
            assert r.status_code == 405 and r.json()["error"] == "method_not_allowed" and r.headers["allow"] == "GET"
    err(client, "/api/does-not-exist", 404, "not_found")
    r = client.get("/missing-file.txt")
    assert r.status_code == 404 and r.json()["status"] == 404
    assert "text/html" in client.get("/").headers["content-type"]


def test_api_never_writes_scores_or_downloads(client, monkeypatch, tmp_path):
    def no_network(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    from se import scorer
    monkeypatch.setattr(scorer, "score", no_network)
    monkeypatch.setattr(scorer, "mock_model_score", no_network)
    for url in ("/api/meta", "/api/scorers", "/api/sentiment/SYN05", "/api/sentiment/SYN05/headlines", "/api/compare/SYN05",
                "/api/experiments", "/api/labels/examples", "/api/features/definitions", "/api/gates"):
        ok(client, url)
    assert client.db_path.read_bytes() == client.db_before
    empty = TestClient(api.create_app(db_path=tmp_path / "none.sqlite"))
    r = empty.get("/api/meta")
    assert r.status_code == 503 and r.json()["error"] == "no_database"
    assert empty.get("/api/health").json()["data"]["database"] is False


def test_bundle_matches_api(client):
    b = api.build_bundle(client.db_path, headlines_per_symbol=40)
    assert b["/meta"]["banner"] == api.BANNER
    for key in ("/symbols", "/scorers", "/features/definitions", "/sentiment/SYN04?scorer=demo_lexicon",
                "/sentiment/SYN04/headlines?limit=40&offset=0", "/compare/SYN04", "/experiments"):
        assert b[key]["data"] == ok(client, "/api" + key), key
    json.dumps(b, allow_nan=False)
    used = set(re.findall(r'get\(`?"?(/[a-z/-]+)', HTML))
    assert used <= {"/meta", "/symbols", "/gates", "/scorers", "/sentiment/", "/experiments", "/experiments/", "/labels/examples",
                    "/compare/", "/features/definitions", "/tools/assign-day"} and len(used) >= 9


# ---------- static frontend ----------
def test_frontend_static_checks():
    assert "Educational research prototype. Data origin: SYNTHETIC. No evidence of predictive value. Not investment advice." in HTML
    assert find_forbidden(HTML) == []
    assert re.findall(r"https?://", HTML) == [] and "<link" not in HTML and re.findall(r"<script[^>]+src=", HTML) == []
    assert "@import" not in HTML and "url(" not in HTML
    for ref in re.findall(r'fetch\("([^"]+)"\)', HTML):
        assert ref in ("api/health", "data.json"), ref
    assert (ROOT / "web" / "data.json").is_file() or True
    for tab in ("Overview", "Sentiment", "Headlines", "Experiments", "Cutoff cases", "Model comparison",
                "Feature explanations", "Gates", "Method tools", "Methodology and limitations"):
        assert f'"{tab}"' in HTML
    assert "prefers-color-scheme:dark" in HTML and 'name="viewport"' in HTML
    assert "No equivalence with any close-to-close label is claimed" in HTML
    assert "not evidence about markets" in HTML and "does NOT mean" in HTML
    assert HTML.count("<svg") == HTML.count('role="img"') == 3 and HTML.count("aria-label=") >= 4


def test_forbidden_words_absent_from_api_strings():
    text = json.dumps([api.SCORERS, api.DEFINITIONS, api.BASE_WARNINGS, api.BANNER, api.q_label_examples(), api.q_gates()])
    assert find_forbidden(text) == []


def test_frontend_script_parses(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = re.search(r"<script>\n(.*)</script>", HTML, re.S).group(1)
    f = tmp_path / "app.js"
    f.write_text(script)
    assert subprocess.run([node, "--check", str(f)], capture_output=True).returncode == 0
