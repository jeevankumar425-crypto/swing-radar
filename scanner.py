"""Swing Radar daily scanner.

NSE daily file -> data checks -> indicators -> top swing picks -> Telegram.
Research aid only, not investment advice.
"""
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

DATA = "data"
HIST, IDEAS, STATE = f"{DATA}/history.csv", f"{DATA}/ideas.json", f"{DATA}/state.json"
NAMES_FILE = f"{DATA}/names.json"
PRICES, EVENTS, STATUS = f"{DATA}/prices.json", f"{DATA}/events.json", f"{DATA}/status.json"
NAMES = {}
IST = ZoneInfo("Asia/Kolkata")
HEAD = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}
URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{:%Y%m%d}_F_0000.csv.zip"
COLS = {  # NSE file column -> our name
    "TckrSymb": "symbol", "SctySrs": "series", "OpnPric": "open", "HghPric": "high",
    "LwPric": "low", "ClsPric": "close", "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume", "TtlTrfVal": "value",
}
KEEP_TOP = 300          # keep the 300 most traded stocks each day
ETFS = {"GOLDBEES", "SILVERBEES", "NIFTYBEES", "BANKBEES", "ITBEES", "JUNIORBEES", "PSUBNKBEES"}
ETF_MIN_VALUE = 5e7    # ETFs trade less, so a lower daily-value floor
FEATURES = 8            # bump to force one fresh scan after the output changes
SCHEMA = 2              # bump this to rebuild stored history when what we keep changes
BACKFILL_DAYS = 400     # more than a year of history for the 200-day average and long-term list
MIN_VALUE = 2e8         # skip stocks trading under Rs 20 crore a day
MAX_ERRORS = 5          # stop if downloads keep failing


class DataProblem(Exception):
    pass


def tg(text):
    """Print, and send to Telegram when the secrets are set."""
    print(text)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(state):
    json.dump(state, open(STATE, "w"), indent=1)


# ---------- 1. download and check ----------
def normalize(df, d):
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise DataProblem(f"NSE file format changed, missing columns: {missing}")
    if len(df) < 500:
        raise DataProblem(f"{d}: file looks incomplete ({len(df)} rows)")
    df = df.rename(columns=COLS)[list(COLS.values())]
    df = df[df.series == "EQ"].drop(columns="series")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().query("close > 0 and high >= low")
    df = pd.concat([df.nlargest(KEEP_TOP, "value"), df[df.symbol.isin(ETFS)]]).drop_duplicates("symbol")
    df.insert(0, "date", pd.Timestamp(d))
    return df


def capture_names(raw):
    if "FinInstrmNm" in raw.columns:
        NAMES.update(dict(zip(raw["TckrSymb"].astype(str), raw["FinInstrmNm"].astype(str))))


def load_names(latest):
    names = json.load(open(NAMES_FILE)) if os.path.exists(NAMES_FILE) else {}
    names.update(NAMES)
    if not names:  # first run with news: read the latest file once, just for company names
        r = requests.get(URL.format(latest.date()), headers=HEAD, timeout=30)
        if r.ok:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                capture_names(pd.read_csv(z.open(z.namelist()[0])))
            names.update(NAMES)
    json.dump(names, open(NAMES_FILE, "w"))
    return names


def fetch_day(d):
    """Return that day's data, or None if NSE has no file (holiday or not out yet)."""
    r = requests.get(URL.format(d), headers=HEAD, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]))
    capture_names(raw)
    return normalize(raw, d)


def update_history(today, state):
    os.makedirs(DATA, exist_ok=True)
    if state.get("schema") != SCHEMA:  # stored data is out of date: rebuild it
        state.update(schema=SCHEMA, holidays=[], last_date=None)
        hist = pd.DataFrame()
    else:
        hist = pd.read_csv(HIST, parse_dates=["date"]) if os.path.exists(HIST) else pd.DataFrame()
    have = set(hist["date"].dt.date) if len(hist) else set()
    holidays = set(state.get("holidays", []))
    days = [today - timedelta(days=i) for i in range(BACKFILL_DAYS, -1, -1)]
    todo = [d for d in days if d.weekday() < 5 and d not in have and d.isoformat() not in holidays]
    frames, errors = [], []
    for d in todo:
        try:
            df = fetch_day(d)
        except Exception as e:  # blocked, timeout, changed format
            errors.append(f"{d}: {e}")
            if len(errors) > MAX_ERRORS:
                raise DataProblem("Too many failed downloads. Latest: " + errors[-1])
            continue
        if df is None:
            if (today - d).days > 3:  # settled holiday, do not retry
                holidays.add(d.isoformat())
        else:
            frames.append(df)
        time.sleep(0.4)
    state["holidays"] = sorted(holidays)
    if frames:
        hist = pd.concat([hist] + frames, ignore_index=True)
    if hist.empty:
        raise DataProblem("No price data available. " + (errors[-1] if errors else ""))
    hist = hist.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
    cutoff = pd.Timestamp(today - timedelta(days=BACKFILL_DAYS + 10))
    hist = hist[hist.date >= cutoff]
    hist.to_csv(HIST, index=False)
    return hist


# ---------- 2. indicators ----------
def indicators(g):
    g = g.sort_values("date").reset_index(drop=True)
    c, h, l, v = g.close, g.high, g.low, g.volume
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    g["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    g["sma50"], g["sma200"] = c.rolling(50).mean(), c.rolling(200).mean()
    g["atr"] = tr.rolling(14).mean()
    g["vratio"] = v / v.rolling(20).mean()
    g["high20"], g["low10"] = h.rolling(20).max(), l.rolling(10).min()
    return g


# ---------- 3. setups and score ----------
def pick(x, etf=False):
    if etf:
        if x.value < ETF_MIN_VALUE:
            return None
        if x.close > x.sma50 > x.sma200:
            pattern = "Uptrend, above both averages"
        elif x.close > x.sma200:
            pattern = "Pullback inside a long-term uptrend"
        else:
            pattern = "Weak trend, below 200-day average"
    else:
        if x.value < MIN_VALUE or not x.close > x.sma200 or not 40 < x.rsi < 78:
            return None
        near_high = x.close >= 0.98 * x.high20
        pullback = abs(x.close / x.sma50 - 1) <= 0.03 and x.close >= 0.99 * x.sma50
        if near_high and x.vratio >= 1.3:
            pattern = "Breakout with volume"
        elif near_high:
            pattern = "Near 20-day high"
        elif pullback:
            pattern = "Pullback to 50-day average"
        else:
            return None
    stop = min(max(x.close - 2 * x.atr, x.low10), x.close - 0.5 * x.atr)
    target = x.close + 3 * x.atr
    risk = x.close - stop
    rr = 3 * x.atr / risk
    trend = 40 * (x.close > x.sma50) + 30 * (x.sma50 > x.sma200) + 30 * (x.close > x.sma200)
    momentum = 50 * max(0, 1 - abs(x.rsi - 62) / 22) + 50 * min(1, max(0, (x.vratio - 0.8) / 1.2))
    reward = min(100, max(0, (rr - 1) / 2 * 100)) * (0.5 if risk / x.close > 0.08 else 1)
    fourth = 50  # neutral until news (stocks) and fund quality (ETFs) are connected
    parts = [round(float(p)) for p in (trend, momentum, reward, fourth)]
    why = (f"{pattern}, RSI {x.rsi:.0f}." if etf else
           f"Above 50 and 200-day averages, RSI {x.rsi:.0f}, volume {x.vratio:.1f}x its 20-day average.")
    return {
        "symbol": x.symbol, "close": round(float(x.close), 2), "stop": round(float(stop), 2),
        "target": round(float(target), 2), "score": round(sum(parts) / 4), "parts": parts,
        "pattern": pattern, "timeframe": "1-3 months" if etf else "2-4 weeks", "why": why,
    }


def long_pick(g):
    """Price-based long-term candidate. Fundamentals (profits, debt) are NOT checked."""
    x = g.iloc[-1]
    if len(g) < 212 or x.value < MIN_VALUE:
        return None
    c = g.close
    ret6 = c.iloc[-1] / c.iloc[-127] - 1       # about 6 months
    ret9 = c.iloc[-1] / c.iloc[-191] - 1       # about 9 months
    vol = float(c.pct_change().tail(126).std() * np.sqrt(252))
    tail = c.tail(190)
    mdd = float((tail / tail.cummax() - 1).min())  # worst fall from a peak
    rising = x.sma200 > g.sma200.iloc[-11]
    if not (x.close > x.sma200 and x.sma50 > x.sma200 and rising
            and ret6 > 0 and ret9 > 0.10 and mdd > -0.30 and vol < 0.45):
        return None
    clip = lambda v: max(0.0, min(1.0, v))
    trend = 40 + 30 + 30  # all three trend tests passed above
    returns = 50 * clip(ret6 / 0.40) + 50 * clip(ret9 / 0.60)
    stability = 50 * clip(1 - vol / 0.50) + 50 * clip(1 + mdd / 0.35)
    parts = [round(float(p)) for p in (trend, returns, stability, 50)]  # 4th: fundamentals not checked
    stop = max(x.sma200 * 0.97, x.close * 0.82)
    if stop >= x.close * 0.95:
        stop = x.close * 0.90
    target = x.close * (1 + min(0.30, max(0.10, ret9)))
    return {
        "symbol": x.symbol, "close": round(float(x.close), 2), "stop": round(float(stop), 2),
        "target": round(float(target), 2), "score": round(sum(parts) / 4), "parts": parts,
        "pattern": "Steady uptrend, shallow pullbacks", "timeframe": "1-3 years",
        "why": f"Up {ret6:.0%} in 6 months, above its 200-day average, worst fall {abs(mdd):.0%}, "
               f"volatility {vol:.0%}. Fundamentals not checked.",
    }


def scan(hist):
    latest = hist.date.max()
    swing, etf, long_ = [], [], []
    for sym, g in hist.groupby("symbol"):
        if len(g) < 205 or g.date.max() != latest:
            continue
        g = indicators(g)
        # a price jump versus the previous close means a split or bonus: skip, prices are not adjusted
        if (g.prev_close / g.close.shift(1) - 1).abs().tail(210).gt(0.02).any():
            continue
        x = g.iloc[-1]
        if x[["rsi", "sma50", "sma200", "atr", "vratio", "high20", "low10"]].isna().any():
            continue
        is_etf = sym in ETFS
        p = pick(x, is_etf)
        if p:
            (etf if is_etf else swing).append(p)
        if not is_etf:
            lp = long_pick(g)
            if lp:
                long_.append(lp)
    rank = lambda rows: sorted(rows, key=lambda r: -r["score"])
    return {"swing": rank(swing)[:10], "etf": rank(etf)[:5], "long": rank(long_)[:10]}


# ---------- 3b. news headlines ----------
NEWS_URL = "https://news.google.com/rss/search?q={}&hl=en-IN&gl=IN&ceid=IN:en"


def words(*w):
    return re.compile(r"\b(" + "|".join(re.escape(x) for x in w) + r")\b")


GOOD = words("upgrade", "upgrades", "order win", "bags order", "wins order", "profit jumps", "profit rises",
             "record", "beats", "surges", "approval", "dividend", "bonus", "buyback", "acquires", "rallies")
BAD = words("downgrade", "downgrades", "plunges", "falls", "slumps", "loss", "misses", "cuts", "weak",
            "lawsuit", "ban", "slides", "drops")
SERIOUS = words("fraud", "probe", "raid", "penalty", "default", "scam", "arrest", "insolvency", "resigns",
                "show-cause", "sebi order", "sebi bars", "sebi notice")


def headlines(name, sym):
    """Up to 8 headlines from the last 7 days (title, link, source). Titles and links only."""
    short = re.sub(r"\b(LIMITED|LTD\.?)\b", "", name, flags=re.I).strip().title() or sym
    cutoff = datetime.now(IST) - timedelta(days=7)
    for query in (f'"{short}" when:7d', f"{sym} share price when:7d"):
        r = requests.get(NEWS_URL.format(quote_plus(query)), headers=HEAD, timeout=30)
        r.raise_for_status()
        out, seen = [], set()
        for it in ET.fromstring(r.content).findall("./channel/item")[:20]:
            title = (it.findtext("title") or "").strip()
            try:
                when = parsedate_to_datetime(it.findtext("pubDate"))
            except Exception:
                continue
            if title and title not in seen and when >= cutoff:
                seen.add(title)
                out.append({"title": title, "link": it.findtext("link") or "", "source": it.findtext("source") or ""})
        if out:
            return out[:8]
        time.sleep(0.5)
    return []


def add_news(res, names):
    """Replace the neutral news score for stock picks with a simple headline check."""
    found = 0
    for p in res["swing"]:
        try:
            hl = headlines(names.get(p["symbol"], p["symbol"]), p["symbol"])
        except Exception:
            hl = []  # this stock's news failed: leave its news score neutral
        time.sleep(0.5)
        low = [h["title"].lower() for h in hl]
        pos, neg = sum(bool(GOOD.search(t)) for t in low), sum(bool(BAD.search(t)) for t in low)
        flags = [h["title"] for h, t in zip(hl, low) if SERIOUS.search(t)]
        p["parts"][3] = max(10, min(90, 50 + 12 * (pos - neg) - 15 * len(flags)))
        p["score"] = round(sum(p["parts"]) / 4)
        p["headlines"], p["flags"] = hl[:3], flags[:2]
        found += bool(hl)
    res["swing"].sort(key=lambda r: -r["score"])
    return res, [f"News: headlines found for {found} of {len(res['swing'])} picks (simple keyword check, not a full reading)."]


# ---------- 3c. second source: BSE ----------
BSE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{:%Y%m%d}_F_0000.CSV"
TOL_WARN = 0.01     # a gap over 1% keeps the pick but without the [checked] tag
TOL = 0.02          # a gap over 2% on any day removes the pick (NSE and BSE closes differ a little)
CHECK_DAYS = 5      # compare the last 5 sessions


class CheckUnavailable(Exception):
    """The second source could not be used. Picks go out unverified."""


def bse_closes(days):
    """{(symbol, date): close} from BSE's daily files. Days with no file are skipped."""
    out, fetched = {}, 0
    head = {**HEAD, "Referer": "https://www.bseindia.com/"}
    for d in days:
        r = requests.get(BSE_URL.format(d), headers=head, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        if "TckrSymb" not in df.columns or "ClsPric" not in df.columns:
            raise CheckUnavailable("BSE file format changed")
        fetched += 1
        syms = df["TckrSymb"].astype(str).str.strip()
        for sym, c in zip(syms, pd.to_numeric(df["ClsPric"], errors="coerce")):
            if c > 0:
                out[(sym, pd.Timestamp(d))] = float(c)
        time.sleep(0.4)
    if not fetched:
        raise CheckUnavailable("no BSE files could be downloaded")
    return out


def crosscheck(res, hist, today):
    """Compare recent closes with BSE. Drops picks where the two exchanges disagree."""
    picks = res["swing"] + res["etf"] + res["long"]
    days = [pd.Timestamp(d).date() for d in sorted(hist.date.unique())[-CHECK_DAYS:]]
    theirs = bse_closes(days)
    mine = hist.set_index(["symbol", "date"]).close
    # wide disagreement across the whole file means one of the files is bad
    latest, both, wide = pd.Timestamp(days[-1]), 0, 0
    for sym in hist.loc[hist.date == latest, "symbol"]:
        t = theirs.get((sym, latest))
        if t:
            both += 1
            wide += abs(t / float(mine[(sym, latest)]) - 1) > 0.02
    if both >= 50 and wide / both > 0.10:
        raise DataProblem(f"NSE and BSE closes differ by over 2% for {wide} of {both} stocks. Picks paused.")
    bad, ok, unk, why, loose = [], 0, 0, {}, {}
    for p in picks:
        diffs = []
        for d in days:
            key = (p["symbol"], pd.Timestamp(d))
            if key in mine.index and key in theirs:
                n, b = float(mine[key]), theirs[key]
                diffs.append((abs(b / n - 1), d, n, b))
        if len(diffs) < 3:
            p["verified"] = None
            unk += 1
        elif max(diffs)[0] > TOL:
            p["verified"] = False
            if p["symbol"] not in why:
                gap, d, n, b = max(diffs)
                why[p["symbol"]] = f"{p['symbol']} on {d:%d %b}: NSE {n:.2f} vs BSE {b:.2f}, {gap:.1%} apart"
            bad.append(p["symbol"])
        elif max(diffs)[0] > TOL_WARN:
            p["verified"] = None
            gap, d, n, b = max(diffs)
            loose.setdefault(p["symbol"], f"{p['symbol']} on {d:%d %b}: NSE {n:.2f} vs BSE {b:.2f}, {gap:.1%} apart")
            unk += 1
        else:
            p["verified"] = True
            ok += 1
    if len(bad) > max(2, len(picks) // 3):
        raise DataProblem(f"NSE and BSE disagree on {len(bad)} of {len(picks)} picks. Picks paused.")
    for k in ("swing", "etf", "long"):
        res[k] = [p for p in res[k] if p["verified"] is not False]
    note = f"Checked against BSE: {ok} match, {unk} not tagged, {len(bad)} removed."
    return res, ([note] + [f"Removed: {w}" for w in why.values()]
                 + [f"Kept without tag: {w}" for w in loose.values()])


# ---------- 3d. data for the dashboard's Home tab ----------
def now_iso():
    return datetime.now(IST).isoformat(timespec="minutes")


def market_snapshot(hist):
    """Last close and previous close for every stored symbol, plus a simple market summary."""
    latest = hist.date.max()
    last = hist[hist.date == latest].set_index("symbol")
    px = {sym: [round(float(r.close), 2), round(float(r.prev_close), 2)] for sym, r in last.iterrows()}
    stocks = last[~last.index.isin(ETFS)]
    stocks = stocks[stocks.prev_close > 0]
    chg = stocks.close / stocks.prev_close - 1
    sma = hist[~hist.symbol.isin(ETFS)].groupby("symbol").close.apply(
        lambda c: c.tail(50).mean() if len(c) >= 50 else np.nan).reindex(stocks.index)
    valid = int(sma.notna().sum())
    mk = {"stocks": int(len(stocks)), "adv": int((chg > 0).sum()), "dec": int((chg < 0).sum()),
          "above50": round(100 * int((stocks.close > sma).sum()) / valid) if valid else None}
    if "NIFTYBEES" in last.index:
        c, p = float(last.loc["NIFTYBEES", "close"]), float(last.loc["NIFTYBEES", "prev_close"])
        if p > 0:
            mk["nifty"] = {"close": round(c, 2), "chg": round((c / p - 1) * 100, 2)}
    return {"asof": str(latest.date()), "p": px, "market": mk}


def build_events(res, notes, prev, latest):
    t, d, ev = now_iso(), str(latest.date()), []
    old = {p["symbol"] for p in (prev or {}).get("swing", [])}
    for p in res["swing"]:
        if p["symbol"] not in old:
            ev.append({"t": t, "d": d, "k": "New pick",
                       "m": f"{p['symbol']} entered your swing picks (score {p['score']}, {p['pattern']})."})
        for fl in p.get("flags", [])[:1]:
            ev.append({"t": t, "d": d, "k": "Warning", "m": f"{p['symbol']}: {fl[:140]}"})
    for n in notes:
        ev.append({"t": t, "d": d, "k": "News" if n.startswith("News") else "Data check", "m": n})
    return ev


def save_events(new):
    old = json.load(open(EVENTS)) if os.path.exists(EVENTS) else []
    seen = {(e.get("d"), e["m"]) for e in old}
    fresh = [e for e in new if (e.get("d"), e["m"]) not in seen]
    json.dump((fresh + old)[:40], open(EVENTS, "w"), indent=1)


def save_status(ok, asof=None, notes=None, error=None):
    json.dump({"ok": ok, "time": now_iso(), "asof": asof, "notes": notes or [], "error": error},
              open(STATUS, "w"), indent=1)


def touch_status():
    old = json.load(open(STATUS)) if os.path.exists(STATUS) else {"asof": None, "notes": []}
    old.update(ok=True, time=now_iso(), error=None)
    json.dump(old, open(STATUS, "w"), indent=1)


def record_failure(e):
    try:
        os.makedirs(DATA, exist_ok=True)
        old = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
        save_status(False, old.get("asof"), old.get("notes"), str(e)[:200])
        save_events([{"t": now_iso(), "d": now_iso(), "k": "Scan failed", "m": str(e)[:200]}])
    except Exception:
        pass


# ---------- 4. run ----------
def main():
    today = datetime.now(IST).date()
    state = load_state()
    hist = update_history(today, state)
    latest = hist.date.max()
    if (today - latest.date()).days > 4:
        raise DataProblem(f"Latest NSE data is from {latest.date()}, older than expected. Picks paused.")
    json.dump(market_snapshot(hist), open(PRICES, "w"))
    if state.get("last_date") == str(latest.date()) and state.get("features") == FEATURES:
        print("No new trading session since the last run.")
        touch_status()
        save_state(state)
        return
    res = scan(hist)
    try:
        res, check_notes = crosscheck(res, hist, today)
    except DataProblem:
        raise
    except CheckUnavailable as e:  # the second source failing should not stop the scan
        check_notes = [f"BSE check unavailable ({e}). Picks are unverified."]
    except Exception as e:
        check_notes = [f"BSE check failed ({type(e).__name__}). Picks are unverified."]
    try:
        res, notes = add_news(res, load_names(latest))
    except Exception as e:  # news is a bonus: never stop the scan for it
        notes = [f"News check failed ({type(e).__name__}); news scores are neutral."]
    notes = check_notes + notes
    prev = json.load(open(IDEAS)) if os.path.exists(IDEAS) else {}
    json.dump({"asof": str(latest.date()), **res, "notes": notes}, open(IDEAS, "w"), indent=1)
    save_events(build_events(res, notes, prev, latest))
    save_status(True, str(latest.date()), notes)
    lines = [f"Swing Radar picks for {latest.date():%d %b %Y} (end-of-day data)"]
    for i, p in enumerate(res["swing"][:5], 1):
        lines.append(f"{i}. {p['symbol']}  score {p['score']}  price Rs {p['close']}  "
                     f"stop Rs {p['stop']}  target Rs {p['target']}  ({p['pattern']})"
                     f"{' [checked]' if p.get('verified') else ''}")
        for fl in p.get("flags", [])[:1]:
            lines.append(f"   WATCH: {fl[:110]}")
    if not res["swing"]:
        lines.append("No stocks met the setup rules today.")
    if res["etf"]:
        lines.append("Top ETFs: " + ", ".join(f"{p['symbol']} ({p['score']})" for p in res["etf"][:3]))
    else:
        lines.append("No ETF data today.")
    if res["long"]:
        lines.append("Long-term (price-based, fundamentals not checked): "
                     + ", ".join(f"{p['symbol']} ({p['score']})" for p in res["long"][:3]))
    lines.extend(notes)
    lines.append("Research aid, not advice. Place stop-losses at your broker.")
    tg("\n".join(lines))
    state["last_date"] = str(latest.date())
    state["features"] = FEATURES
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tg(f"Swing Radar scan failed: {e}")
        record_failure(e)
        sys.exit(1)
