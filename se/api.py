"""Read-only JSON API (GET only) over var/demo.sqlite. It never scores text and never downloads anything."""
import json
import re
from pathlib import Path

import numpy as np

from . import config, db, gates, labels, timeutil
from .scorer import BUCKETS, MOCK_LABEL, MOCK_LABELS

ROOT = config.ROOT
DB_PATH = ROOT / "var" / "demo.sqlite"
BANNER = ("Educational research prototype. Data origin: SYNTHETIC. "
          "No evidence of predictive value. Not investment advice.")
BASE_WARNINGS = ["Data origin: SYNTHETIC. Nothing here describes a real company or market.",
                 "Scores are lexicon counts or classifier output scores, not probabilities of any price move."]
MAX_DAYS, MAX_LIMIT, DEFAULT_LIMIT = 1000, 200, 50
NO_EQUIVALENCE = "No equivalence with any close-to-close label is claimed."
_SYMBOL = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")

SCORERS = [
    {"id": "mock_model", "label": MOCK_LABEL, "kind": "model", "status": "loaded", "outputs": "labels " + "/".join(MOCK_LABELS),
     "reason": "Deterministic fake classifier built from the demo lexicon counts plus hash noise. Demo only."},
    {"id": "demo_lexicon", "label": "Demo lexicon (not Loughran-McDonald)", "kind": "lexicon", "status": "loaded",
     "outputs": "buckets " + " / ".join(BUCKETS),
     "reason": "Hand-written 20 + 20 word list. Synthetic headlines are built from it, so it is circular by construction."},
    {"id": "finbert_prosus", "label": "FinBERT (ProsusAI)", "kind": "model", "status": "not_loaded", "outputs": "labels",
     "reason": "No model weights are included or downloaded. Adapter exists; exploratory use only."},
    {"id": "finbert_tone", "label": "finbert-tone", "kind": "model", "status": "not_loaded", "outputs": "labels",
     "reason": "Blocked: licence unresolved. The adapter refuses to load."},
    {"id": "lm_lexicon", "label": "Loughran-McDonald lexicon", "kind": "lexicon", "status": "not_loaded", "outputs": "buckets",
     "reason": "No dictionary file supplied. The owner must provide a local CSV under its terms."},
]
SCORER_IDS = {s["id"]: s for s in SCORERS}

DEFINITIONS = [
    {"id": "S1", "name": "Same-day sentiment", "formula": "S1(s,t) = sum(w*x) / sum(w) over articles assigned to stock s on trading day t",
     "plain": "The average tone of today's news about the stock.", "why": "The freshest signal available at the cutoff.",
     "limits": "Often based on a single headline, so it is noisy.", "missing": "Missing when no scored article is assigned to day t; rank becomes 0.5."},
    {"id": "S5", "name": "Five-day sentiment", "formula": "S5(s,t) = sum(w*x) / sum(w) over trading days t-4..t",
     "plain": "The average tone of the last five trading days of news.", "why": "Smooths single-headline noise.",
     "limits": "Slower to react; mixes old and new items equally.", "missing": "Missing when there is no scored article in the five days; rank becomes 0.5."},
    {"id": "NVS", "name": "News-volume surprise", "formula": "NVS = ln(1 + N5) - ln(1 + 5*N_base/60), N5 = articles in t-4..t, N_base = articles in t-64..t-5",
     "plain": "Is the stock getting more coverage than usual, regardless of tone?", "why": "Separates 'more news' from 'better news' (arm C1 uses only this).",
     "limits": "Says nothing about direction.", "missing": "Never missing after the 64-day warm-up (zero articles gives a defined value)."},
    {"id": "ABN", "name": "Abnormal sentiment", "formula": "ABN = S5 - (sum(w*x) / sum(w) over t-64..t-5)",
     "plain": "Is recent tone better or worse than this stock's own usual tone?", "why": "Removes a stock's habitual tone.",
     "limits": "Needs history; unstable with few baseline articles.", "missing": "Missing if S5 is missing or fewer than 10 baseline articles; rank becomes 0.5."},
    {"id": "has_news", "name": "Has recent news", "formula": "has_news = 1 if N5 >= 1 else 0",
     "plain": "Whether there was any scored news in the last five trading days.", "why": "Lets the model treat 'no news' differently from 'neutral news'. Also defines sample S2.",
     "limits": "A crude on/off flag.", "missing": "Never missing after the warm-up."},
    {"id": "ranks", "name": "Per-date ranks", "formula": "rank_frac = (average-tie rank - 0.5) / n among stocks with a value that date",
     "plain": "Each feature is compared across stocks on the same day, on a 0 to 1 scale.", "why": "Makes days comparable and limits outliers.",
     "limits": "Discards the size of differences.", "missing": "Missing values get 0.5; if fewer than 5 stocks have a value that date, all get 0.5."},
    {"id": "labels", "name": "Execution-based label", "formula": "R_exec(t) = Open(t+11)/Open(t+1) - 1; Z_exec = R_exec / max(ATR%(t), 0.001); L_exec = 1 if Z_exec is in the top 30% that date",
     "plain": "The return you could actually have traded after seeing day t's news: enter at the next open, exit ten trading days later.", "why": "A signal formed at the cutoff cannot trade at that day's close.",
     "limits": NO_EQUIVALENCE, "missing": "Missing if any needed bar is missing or non-positive, or fewer than 20 stocks that date."},
]


class ApiError(Exception):
    def __init__(self, status, error, detail):
        self.status, self.error, self.detail = status, error, detail


def envelope(data, extra_warnings=()):
    return {"data_origin": "SYNTHETIC", "banner": BANNER, "warnings": BASE_WARNINGS + list(extra_warnings), "data": data}


def error_body(status, error, detail):
    return {"error": error, "detail": detail, "status": status, "data_origin": "SYNTHETIC", "banner": BANNER,
            "warnings": BASE_WARNINGS}


def _r(v):
    return None if v is None else round(float(v), 4)


def _int_param(name, value, default, lo, hi):
    if value is None or value == "":
        return default
    try:
        n = int(value)
    except ValueError:
        raise ApiError(400, "bad_parameter", f"{name} must be an integer")
    if not lo <= n <= hi:
        raise ApiError(400, "bad_parameter", f"{name} must be between {lo} and {hi}")
    return n


def _symbol(conn, symbol):
    if not _SYMBOL.match(symbol or ""):
        raise ApiError(400, "bad_parameter", "symbol has an invalid format")
    row = conn.execute("SELECT symbol FROM symbols WHERE symbol = ?", (symbol.upper(),)).fetchone()
    if row is None:
        raise ApiError(404, "unknown_symbol", f"symbol {symbol!r} is not in the database")
    return row["symbol"]


def _scorer(scorer, default="mock_model"):
    sid = scorer or default
    if sid not in SCORER_IDS:
        raise ApiError(404, "unknown_scorer", f"scorer {sid!r} is unknown; known: {sorted(SCORER_IDS)}")
    return SCORER_IDS[sid]


def _trading_days(conn):
    return [r[0] for r in conn.execute("SELECT DISTINCT date FROM prices WHERE world = 'planted' ORDER BY date")]


# ---------- payload functions (also used to build the static fallback bundle) ----------
def q_meta(conn):
    meta = {k: v for k, v in db.get_meta(conn).items() if k != "robustness"}
    return dict(meta, schema_version=db.schema_version(conn), row_counts=db.row_counts(conn),
                prereg=config.prereg_status(), primary_scorer="mock_model", read_only=True)


def q_gates():
    return gates.load()


def q_scorers():
    g = gates.load()["gates"]
    out = []
    for s in SCORERS:
        e = g.get("scorer:" + s["id"], {})
        out.append(dict(s, licence=e.get("licence_note") or e.get("licence", "-"), contamination=e.get("contamination", "-"),
                        allowed_use=e.get("allowed_use", "-"), gate_status=e.get("status", "-"), unlock=e.get("unlock", "-")))
    return {"scorers": out, "note": "Lexicon buckets and model labels are different things and are never merged. "
                                    "On synthetic data the demo lexicon recovers the hidden tone by construction."}


def q_symbols(conn):
    return {"symbols": [r[0] for r in conn.execute("SELECT symbol FROM symbols ORDER BY symbol")]}


def q_sentiment(conn, symbol, scorer=None, days=None, date=None):
    sym, sc = _symbol(conn, symbol), _scorer(scorer)
    if sc["status"] != "loaded":
        return {"symbol": sym, "scorer": sc["id"], "scorer_label": sc["label"], "status": "not_loaded", "reason": sc["reason"]}
    n = _int_param("days", days, MAX_DAYS, 1, MAX_DAYS)
    rows = conn.execute("SELECT * FROM features WHERE symbol = ? AND scorer_id = ? ORDER BY date DESC LIMIT ?",
                        (sym, sc["id"], n)).fetchall()[::-1]
    counts = dict(conn.execute(
        "SELECT sc.bucket, COUNT(*) FROM assignments a JOIN scores sc ON sc.article_id = a.article_id AND sc.scorer_id = ? "
        "WHERE a.symbol = ? AND a.kept = 1 GROUP BY sc.bucket", (sc["id"], sym)).fetchall())
    order = BUCKETS if sc["kind"] == "lexicon" else MOCK_LABELS
    out = {"symbol": sym, "scorer": sc["id"], "scorer_label": sc["label"], "status": "loaded", "kind": sc["kind"],
           "dates": [r["date"] for r in rows], "S1": [_r(r["s1"]) for r in rows], "S5": [_r(r["s5"]) for r in rows],
           "count": [r["n1"] for r in rows], "bucket_kind": "lexicon buckets" if sc["kind"] == "lexicon" else "model labels",
           "buckets": {b: int(counts.get(b, 0)) for b in order}}
    if date:
        r = conn.execute("SELECT * FROM features WHERE symbol = ? AND scorer_id = ? AND date = ?", (sym, sc["id"], date)).fetchone()
        if r is None:
            raise ApiError(404, "unknown_date", f"no feature row for {sym} on {date!r}")
        out["at"] = {"date": date, "S1": _r(r["s1"]), "S5": _r(r["s5"]), "NVS": _r(r["nvs"]), "ABN": _r(r["abn"]),
                     "has_news": r["has_news"], "N1": r["n1"], "N5": r["n5"]}
    return out


def q_headlines(conn, symbol, scorer=None, days=None, limit=None, offset=None, date=None):
    sym, sc = _symbol(conn, symbol), _scorer(scorer)
    lim = _int_param("limit", limit, DEFAULT_LIMIT, 1, MAX_LIMIT)
    off = _int_param("offset", offset, 0, 0, 10 ** 6)
    where, params, window = "s.symbol = ?", [sym], None
    tdays = _trading_days(conn)
    if date:
        if date not in tdays:
            raise ApiError(404, "unknown_date", f"{date!r} is not a trading day in the database")
        i = tdays.index(date)
        window = [tdays[max(0, i - 4)], date]
        where += " AND s.trading_date BETWEEN ? AND ?"
        params += window
    elif days not in (None, ""):
        n = _int_param("days", days, MAX_DAYS, 1, MAX_DAYS)
        where += " AND s.trading_date >= ?"
        params.append(tdays[-min(n, len(tdays))])
    total = conn.execute(f"SELECT COUNT(*) FROM assignments s WHERE {where}", params).fetchone()[0]
    rows = conn.execute(
        "SELECT a.article_id, a.headline, a.first_published, a.precision, a.kind, s.trading_date, s.reason, s.kept, "
        "l.weight, lx.bucket AS lex_bucket, lx.x AS lex_x, lx.matched_terms, mm.bucket AS model_label, mm.x AS model_x, "
        "mm.p_pos, mm.p_neg, mm.p_neu FROM assignments s JOIN articles a ON a.article_id = s.article_id "
        "JOIN article_symbols l ON l.article_id = s.article_id AND l.symbol = s.symbol "
        "LEFT JOIN scores lx ON lx.article_id = a.article_id AND lx.scorer_id = 'demo_lexicon' "
        "LEFT JOIN scores mm ON mm.article_id = a.article_id AND mm.scorer_id = 'mock_model' "
        f"WHERE {where} ORDER BY s.trading_date IS NULL, s.trading_date DESC, a.article_id LIMIT ? OFFSET ?",
        params + [lim, off]).fetchall()
    items = [{
        "article_id": r["article_id"], "headline": r["headline"], "kind": r["kind"], "utc": r["first_published"],
        "local": timeutil.local_str(r["first_published"], "US") if r["precision"] != "date" else "",
        "assigned": r["trading_date"], "reason": r["reason"], "kept": bool(r["kept"]), "weight": _r(r["weight"]),
        "lexicon": {"scorer": "demo_lexicon", "bucket": r["lex_bucket"], "x": _r(r["lex_x"]),
                    "matched": json.loads(r["matched_terms"] or "[]"), "probabilities": None},
        "model": {"scorer": "mock_model", "scorer_label": MOCK_LABEL, "label": r["model_label"], "x": _r(r["model_x"]),
                  "output_scores": {"positive": _r(r["p_pos"]), "negative": _r(r["p_neg"]), "neutral": _r(r["p_neu"])}},
    } for r in rows]
    return {"symbol": sym, "scorer": sc["id"], "total": total, "limit": lim, "offset": off, "window": window, "items": items}


def q_compare(conn, symbol):
    sym = _symbol(conn, symbol)
    rows = conn.execute(
        "SELECT lx.bucket, mm.bucket, lx.x, mm.x FROM assignments s "
        "JOIN scores lx ON lx.article_id = s.article_id AND lx.scorer_id = 'demo_lexicon' "
        "JOIN scores mm ON mm.article_id = s.article_id AND mm.scorer_id = 'mock_model' "
        "WHERE s.symbol = ? AND s.kept = 1", (sym,)).fetchall()
    matrix = [[sum(1 for r in rows if r[0] == b and r[1] == l) for l in MOCK_LABELS] for b in BUCKETS]
    covered = [r for r in rows if r[0] in ("positive", "negative")]
    agree = sum(1 for r in covered if (r[2] > 0) == (r[3] > 0))
    return {"symbol": sym, "n": len(rows), "rows": BUCKETS, "row_scorer": "demo_lexicon (buckets)",
            "cols": MOCK_LABELS, "col_scorer": MOCK_LABEL + " (labels)", "matrix": matrix,
            "sign_agreement": {"n": len(covered), "agree": agree, "share": _r(agree / len(covered)) if covered else None,
                               "definition": "among headlines the lexicon marks positive or negative: same sign of x"},
            "note": "Agreement between two scorers on the same synthetic headlines. Lexicon buckets and model labels are "
                    "kept separate. High agreement is expected: the mock model is built from the lexicon counts plus noise."}


def q_definitions():
    return {"definitions": DEFINITIONS, "statement": NO_EQUIVALENCE}


def q_experiments(conn):
    runs = [dict(r) for r in conn.execute("SELECT run_id, mode, world, seed, verdict, prereg_match FROM runs ORDER BY world")]
    return {"runs": runs, "robustness": db.get_meta(conn).get("robustness"),
            "note": "Software check on invented data, not evidence about markets."}


def q_experiment(conn, run_id):
    r = conn.execute("SELECT json FROM run_results WHERE run_id = ? AND key = 'result'", (run_id,)).fetchone()
    if r is None:
        raise ApiError(404, "unknown_run", f"run {run_id!r} is not in the database")
    return json.loads(r["json"])


def q_label_examples():
    """Worked examples, computed now by the real functions."""
    h, l, c = [101, 103, 102, 104], [99, 101, 100, 102], [100, 102, 101, 103]
    tr = labels.true_range(h, l, c)
    o = np.full(12, 100.0)
    o[1], o[11] = 101.0, 108.9
    r = float(labels.r_exec(o)[0])
    return {"statement": NO_EQUIVALENCE, "examples": [
        {"title": "True range and ATR (window 3)", "inputs": "Bars (High, Low, Close): t0 101,99,100; t1 103,101,102; t2 102,100,101; t3 104,102,103",
         "steps": "TR = max(High-Low, |High-prev Close|, |Low-prev Close|); ATR = mean of the last 3 TR; ATR% = ATR / Close",
         "result": {"TR t1..t3": [float(v) for v in tr[1:]], "ATR(t3)": round(float(labels.atr(h, l, c, 3)[3]), 6),
                    "ATR%(t3)": round(float(labels.atr_pct(h, l, c, 3)[3]), 6)}},
        {"title": "Executable return and scaled return", "inputs": "Open(t+1) = 101, Open(t+11) = 108.9, ATR%(t) = 0.02",
         "steps": "R_exec = Open(t+11)/Open(t+1) - 1; Z_exec = R_exec / max(ATR%, 0.001)",
         "result": {"R_exec": round(r, 6), "Z_exec": round(float(labels.z_exec(r, 0.02)), 6)}},
        {"title": "ATR floor", "inputs": "R_exec = 0.01, ATR%(t) = 0.0004", "steps": "ATR% below the floor 0.001 is replaced by 0.001",
         "result": {"Z_exec": round(float(labels.z_exec(0.01, 0.0004)), 6)}},
        {"title": "Top-30% label", "inputs": "A date with 20 stocks with distinct Z_exec; a date with 19 stocks",
         "steps": "L_exec = 1 if percentile rank >= 0.70; a group smaller than 20 gets no labels",
         "result": {"labels in the group of 20": int(np.nansum(labels.l_exec_group(np.arange(20.0)))),
                    "labels in the group of 19": int(np.sum(~np.isnan(labels.l_exec_group(np.arange(19.0)))))}},
        {"title": "Average ranks for ties", "inputs": "Values 1, 2, 2, 3", "steps": "Tied values share the mean of their ranks",
         "result": {"ranks": [float(v) for v in labels.average_ranks([1, 2, 2, 3])]}},
    ], "cutoff_cases": timeutil.run_cutoff_cases()}


def q_assign(conn, ts, market=None, precision=None):
    market, precision = market or "US", precision or "timestamp"
    if market not in timeutil.MARKETS:
        raise ApiError(400, "bad_parameter", f"market must be one of {sorted(timeutil.MARKETS)}")
    if precision not in ("timestamp", "date"):
        raise ApiError(400, "bad_parameter", "precision must be 'timestamp' or 'date'")
    if not ts:
        raise ApiError(400, "bad_parameter", "ts is required, e.g. 2023-03-14T19:30:00Z")
    from datetime import date as _date
    days = [_date.fromisoformat(d) for d in _trading_days(conn)] + timeutil.demo_2023_calendar().days
    day, reason = timeutil.assign(ts, market, timeutil.TradingCalendar(days), precision)
    m = timeutil.MARKETS[market]
    return {"input": ts, "market": market, "precision": precision, "time_zone": str(m["tz"]),
            "cutoff_local": m["cutoff"].strftime("%H:%M"), "local": timeutil.local_str(ts, market) if precision != "date" else "",
            "assigned": day.isoformat() if day else None, "reason": reason, "valid": day is not None,
            "calendar": "synthetic 2020-2022 business days plus the 2023 demo calendar (weekdays minus ten listed holidays); "
                        "illustrative, not an exchange calendar"}


def build_bundle(db_path=DB_PATH, headlines_per_symbol=40):
    """Static fallback for the dashboard: the same payloads the API serves, keyed by request path."""
    conn = db.connect_ro(db_path)
    b = {"/meta": q_meta(conn), "/gates": q_gates(), "/scorers": q_scorers(), "/symbols": q_symbols(conn),
         "/features/definitions": q_definitions(), "/experiments": q_experiments(conn), "/labels/examples": q_label_examples()}
    for run in b["/experiments"]["runs"]:
        b["/experiments/" + run["run_id"]] = q_experiment(conn, run["run_id"])
    for sym in b["/symbols"]["symbols"]:
        for sc in ("mock_model", "demo_lexicon"):
            b[f"/sentiment/{sym}?scorer={sc}"] = q_sentiment(conn, sym, sc)
        b[f"/sentiment/{sym}/headlines?limit={headlines_per_symbol}&offset=0"] = q_headlines(conn, sym, None, None, headlines_per_symbol, 0)
        b[f"/compare/{sym}"] = q_compare(conn, sym)
    conn.close()
    return {k: envelope(v) for k, v in b.items()}


# ---------- FastAPI app ----------
def create_app(db_path=DB_PATH, web_dir=None):
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    app = FastAPI(title="Sentiment Engine Lite (synthetic, read-only)", docs_url=None, redoc_url=None, openapi_url=None)
    web_dir = Path(web_dir or ROOT / "web")

    def fail(status, error, detail, headers=None):
        return JSONResponse(error_body(status, error, detail), status_code=status, headers=headers)

    @app.middleware("http")
    async def get_only(request: Request, call_next):
        if request.method not in ("GET", "HEAD"):
            return fail(405, "method_not_allowed", "this API is read-only: GET only", {"Allow": "GET"})
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def on_api_error(request, exc):
        return fail(exc.status, exc.error, exc.detail)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request, exc):
        return fail(exc.status_code, "not_found" if exc.status_code == 404 else "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request, exc):
        return fail(400, "bad_parameter", "invalid request parameters")

    def run(fn, *args, needs_db=True):
        warnings = [] if config.prereg_status()["match"] else ["Pre-registration hashes do not match the frozen record."]
        if not needs_db:
            return envelope(fn(*args), warnings)
        if not Path(db_path).is_file():
            raise ApiError(503, "no_database", "var/demo.sqlite not found; run: .venv/bin/python run_demo.py")
        conn = db.connect_ro(db_path)
        try:
            return envelope(fn(conn, *args), warnings)
        finally:
            conn.close()

    @app.get("/api/health")
    def health():
        return envelope({"status": "ok", "database": Path(db_path).is_file(), "read_only": True, "scores_on_request": False})

    @app.get("/api/meta")
    def meta():
        return run(q_meta)

    @app.get("/api/gates")
    def gates_():
        return run(q_gates, needs_db=False)

    @app.get("/api/scorers")
    def scorers():
        return run(q_scorers, needs_db=False)

    @app.get("/api/symbols")
    def symbols():
        return run(q_symbols)

    @app.get("/api/sentiment/{symbol}")
    def sentiment(symbol: str, scorer: str = None, days: str = None, date: str = None):
        return run(q_sentiment, symbol, scorer, days, date)

    @app.get("/api/sentiment/{symbol}/headlines")
    def headlines(symbol: str, scorer: str = None, days: str = None, limit: str = None, offset: str = None, date: str = None):
        return run(q_headlines, symbol, scorer, days, limit, offset, date)

    @app.get("/api/compare/{symbol}")
    def compare(symbol: str):
        return run(q_compare, symbol)

    @app.get("/api/features/definitions")
    def definitions():
        return run(q_definitions, needs_db=False)

    @app.get("/api/experiments")
    def experiments():
        return run(q_experiments)

    @app.get("/api/experiments/{run_id}")
    def experiment(run_id: str):
        return run(q_experiment, run_id)

    @app.get("/api/labels/examples")
    def label_examples():
        return run(q_label_examples, needs_db=False)

    @app.get("/api/tools/assign-day")
    def assign_day(ts: str = None, market: str = None, precision: str = None):
        return run(q_assign, ts, market, precision)

    @app.get("/api/{rest:path}")
    def unknown(rest: str):
        raise ApiError(404, "not_found", f"no such endpoint: /api/{rest}")

    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


def serve(host="127.0.0.1", port=8000):
    import uvicorn
    print(f"Serving dashboard and API at http://{host}:{port}/  (Ctrl+C to stop)")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
