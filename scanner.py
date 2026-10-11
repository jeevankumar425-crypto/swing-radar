"""Swing Radar daily scanner.

NSE daily file -> data checks -> indicators -> top swing picks -> Telegram.
Research aid only, not investment advice.
"""
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

DATA = "data"
HIST, IDEAS, STATE = f"{DATA}/history.csv", f"{DATA}/ideas.json", f"{DATA}/state.json"
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
SCHEMA = 2              # bump this to rebuild stored history when what we keep changes
BACKFILL_DAYS = 330     # about 1 year of history for the 200-day average
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


def fetch_day(d):
    """Return that day's data, or None if NSE has no file (holiday or not out yet)."""
    r = requests.get(URL.format(d), headers=HEAD, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return normalize(pd.read_csv(z.open(z.namelist()[0])), d)


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


def scan(hist):
    latest = hist.date.max()
    swing, etf = [], []
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
    rank = lambda rows: sorted(rows, key=lambda r: -r["score"])
    return {"swing": rank(swing)[:10], "etf": rank(etf)[:5]}


# ---------- 4. run ----------
def main():
    today = datetime.now(IST).date()
    state = load_state()
    hist = update_history(today, state)
    latest = hist.date.max()
    if (today - latest.date()).days > 4:
        raise DataProblem(f"Latest NSE data is from {latest.date()}, older than expected. Picks paused.")
    if state.get("last_date") == str(latest.date()):
        print("No new trading session since the last run.")
        save_state(state)
        return
    res = scan(hist)
    json.dump({"asof": str(latest.date()), **res}, open(IDEAS, "w"), indent=1)
    lines = [f"Swing Radar picks for {latest.date():%d %b %Y} (end-of-day data)"]
    for i, p in enumerate(res["swing"][:5], 1):
        lines.append(f"{i}. {p['symbol']}  score {p['score']}  price Rs {p['close']}  "
                     f"stop Rs {p['stop']}  target Rs {p['target']}  ({p['pattern']})")
    if not res["swing"]:
        lines.append("No stocks met the setup rules today.")
    if res["etf"]:
        lines.append("Top ETFs: " + ", ".join(f"{p['symbol']} ({p['score']})" for p in res["etf"][:3]))
    else:
        lines.append("No ETF data today.")
    lines.append("Research aid, not advice. Place stop-losses at your broker.")
    tg("\n".join(lines))
    state["last_date"] = str(latest.date())
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tg(f"Swing Radar scan failed: {e}")
        sys.exit(1)
