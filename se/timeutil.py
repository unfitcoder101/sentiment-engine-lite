"""Publication time -> trading day. UTC offsets come from zoneinfo only (never hard-coded)."""
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

MARKETS = {
    "US": {"tz": ZoneInfo("America/New_York"), "close": time(16, 0), "cutoff": time(15, 30)},
    "NSE": {"tz": ZoneInfo("Asia/Kolkata"), "close": time(15, 30), "cutoff": time(15, 0)},
}


class TradingCalendar:
    def __init__(self, dates):
        self.days = sorted(set(dates))
        self.index = {d: i for i, d in enumerate(self.days)}

    def is_trading_day(self, d):
        return d in self.index

    def next_trading_day(self, d):
        """First trading day strictly after d (None if past the calendar end)."""
        i = bisect_right(self.days, d)
        return self.days[i] if i < len(self.days) else None


def weekday_calendar(start, end, holidays=()):
    hol = {date.fromisoformat(h) if isinstance(h, str) else h for h in holidays}
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return TradingCalendar(out)


def assign(published_utc, market, calendar, precision="timestamp", as_of=None):
    """Return (trading_date, reason). Invalid input returns (None, reason)."""
    m = MARKETS[market]
    if published_utc is None or published_utc == "":
        return None, "missing"
    if precision == "date":
        d = published_utc
        if isinstance(d, str):
            try:
                d = date.fromisoformat(d[:10])
            except ValueError:
                return None, "unparseable"
        elif isinstance(d, datetime):
            d = d.date()
        if as_of is not None and d > as_of.astimezone(m["tz"]).date():
            return None, "after_as_of"
        nxt = calendar.next_trading_day(d)
        return (nxt, "date_only") if nxt else (None, "outside_calendar")
    dt = published_utc
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return None, "unparseable"
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None, "naive_timestamp"
    if as_of is not None and dt > as_of:
        return None, "after_as_of"
    local = dt.astimezone(m["tz"])
    d = local.date()
    if calendar.is_trading_day(d):
        if local.time() <= m["cutoff"]:
            return d, "same_day"
        reason = "after_cutoff"
    else:
        reason = "weekend" if d.weekday() >= 5 else "holiday"
    nxt = calendar.next_trading_day(d)
    return (nxt, reason) if nxt else (None, "outside_calendar")


def cutoff_utc(trading_date, market):
    m = MARKETS[market]
    return datetime.combine(trading_date, m["cutoff"], tzinfo=m["tz"]).astimezone(timezone.utc)


def local_str(published_utc, market):
    """Local wall-clock string for display ('' if not a valid aware timestamp)."""
    try:
        dt = datetime.fromisoformat(str(published_utc).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        return ""
    return dt.astimezone(MARKETS[market]["tz"]).strftime("%Y-%m-%d %H:%M:%S %Z")


DEMO_2023_HOLIDAYS = ["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
                      "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25"]

# (input, precision, expected trading date or None, note)
CUTOFF_CASES = [
    ("2023-03-14T19:30:00Z", "timestamp", "2023-03-14", "15:30 EDT, exactly at cutoff: included"),
    ("2023-03-14T19:30:01Z", "timestamp", "2023-03-15", "one second after cutoff"),
    ("2023-01-10T20:30:00Z", "timestamp", "2023-01-10", "15:30 EST, winter offset"),
    ("2023-03-18T15:00:00Z", "timestamp", "2023-03-20", "Saturday"),
    ("2023-03-17T21:00:00Z", "timestamp", "2023-03-20", "Friday after cutoff"),
    ("2023-03-14", "date", "2023-03-15", "date-only: always next trading day"),
    ("2023-03-17", "date", "2023-03-20", "date-only on a Friday"),
    ("2023-11-22T21:00:00Z", "timestamp", "2023-11-24", "Thanksgiving 11-23 closed"),
    ("2023-11-23T15:00:00Z", "timestamp", "2023-11-24", "published on a holiday"),
    ("2023-03-14T19:30:00", "timestamp", None, "naive timestamp: invalid"),
]


def demo_2023_calendar():
    return weekday_calendar(date(2023, 1, 1), date(2023, 12, 31), DEMO_2023_HOLIDAYS)


def run_cutoff_cases():
    cal, rows = demo_2023_calendar(), []
    for value, precision, expected, note in CUTOFF_CASES:
        got, reason = assign(value, "US", cal, precision)
        got_s = got.isoformat() if got else None
        rows.append({"input": value, "precision": precision, "local": local_str(value, "US") if precision != "date" else "",
                     "assigned": got_s, "reason": reason, "expected": expected,
                     "match": got_s == expected, "note": note})
    return rows
