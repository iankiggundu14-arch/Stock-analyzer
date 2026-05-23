"""
AlphaScout — Small & Mid-Cap Outperformer Screener
Backend: FastAPI + yfinance + SEC EDGAR + FRED
"""
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
CACHE  = BASE / "cache"
FRONT  = BASE / "frontend"
CACHE.mkdir(exist_ok=True)
FRONT.mkdir(exist_ok=True)

CACHE_FILE  = CACHE / "screener.json"
STATUS_FILE = CACHE / "status.json"
PRICES_FILE = CACHE / "prices_live.json"

# ── Market-cap tiers (USD) ────────────────────────────────────────────────────
TIER_THRESHOLDS = [
    ("mega",  200_000_000_000),
    ("large",  10_000_000_000),
    ("mid",     2_000_000_000),
    ("small",     300_000_000),
]
TIER_COLORS = {
    "mega":    "#7c3aed",
    "large":   "#2563eb",
    "mid":     "#0891b2",
    "small":   "#059669",
    "micro":   "#d97706",
    "unknown": "#6b7280",
}

# ── Utility functions ─────────────────────────────────────────────────────────
def get_tier(mc: float) -> str:
    if not mc or mc <= 0:
        return "unknown"
    for name, threshold in TIER_THRESHOLDS:
        if mc >= threshold:
            return name
    return "micro"

def fmt_cap(mc: float) -> str:
    if not mc or mc <= 0:
        return "N/A"
    if mc >= 1e12: return f"${mc/1e12:.1f}T"
    if mc >= 1e9:  return f"${mc/1e9:.1f}B"
    if mc >= 1e6:  return f"${mc/1e6:.0f}M"
    return f"${mc:,.0f}"

def set_status(status: str, message: str = "", progress: int = 0):
    STATUS_FILE.write_text(json.dumps({
        "status":   status,
        "message":  message,
        "progress": progress,
        "ts":       datetime.now().isoformat(),
    }), encoding="utf-8")

# ── Market-cap fetcher (runs in thread pool) ──────────────────────────────────
def _fetch_mc(ticker: str):
    try:
        return ticker, getattr(yf.Ticker(ticker).fast_info, "market_cap", None)
    except Exception:
        return ticker, None

# ── Core analysis pipeline ────────────────────────────────────────────────────
def compute():
    try:
        # ── Step 1: S&P 500 constituents from Wikipedia ───────────────────────
        set_status("running", "Fetching S&P 500 constituents from Wikipedia…", 5)
        import requests, io
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            timeout=30,
        )
        resp.raise_for_status()
        wiki = pd.read_html(io.StringIO(resp.text)
        )[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
        wiki.columns = ["symbol", "name", "sector", "sub"]
        wiki["symbol"] = wiki["symbol"].str.replace(".", "-", regex=False)
        tickers = wiki["symbol"].tolist()

        # ── Step 2: Batch price download (6 months, daily) ───────────────────
        set_status("running", f"Downloading 6-month price history for {len(tickers)} stocks…", 15)
        raw = yf.download(
            tickers, period="6mo", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )

        # yfinance returns MultiIndex columns when multiple tickers requested
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw

        if not isinstance(closes, pd.DataFrame):
            closes = closes.to_frame()

        # ── Step 3: Market caps (parallel, 20 workers) ───────────────────────
        set_status("running", "Fetching market caps in parallel…", 45)
        mcs: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=20) as ex:
            for tkr, mc in ex.map(_fetch_mc, tickers):
                if mc:
                    mcs[tkr] = mc

        # ── Step 4: Per-stock records ─────────────────────────────────────────
        set_status("running", "Computing returns and building records…", 65)
        records = []

        for _, row in wiki.iterrows():
            sym = row["symbol"]
            if sym not in closes.columns:
                continue
            s = closes[sym].dropna()
            if len(s) < 10:
                continue

            price = float(s.iloc[-1])

            def ret(n: int):
                if len(s) <= n:
                    return None
                return round((price / float(s.iloc[-n]) - 1) * 100, 2)

            mc   = mcs.get(sym) or 0
            tier = get_tier(mc)

            # Sparkline: last 30 days normalised to 100
            raw30  = s.tail(30).tolist()
            base   = raw30[0] if raw30 else 1.0
            spark  = [round(v / base * 100, 2) for v in raw30] if base else []

            records.append({
                "ticker": sym,
                "name":   str(row["name"]),
                "sector": str(row["sector"]),
                "sub":    str(row["sub"]),
                "mc":     mc,
                "mc_fmt": fmt_cap(mc),
                "tier":   tier,
                "color":  TIER_COLORS.get(tier, "#6b7280"),
                "price":  round(price, 2),
                "r1m":    ret(21),
                "r3m":    ret(63),
                "r6m":    ret(126),
                "spark":  spark,
                # filled in next step
                "rs3m":   None,
                "rs1m":   None,
                "score":  None,
                "pct":    None,
            })

        df = pd.DataFrame(records)

        # ── Reclassify tiers relative to the S&P 500 universe ────────────────
        # Use known market caps only; unknown/zero treated as bottom tier
        known_mc = df.loc[df["mc"] > 0, "mc"]
        if len(known_mc) >= 10:
            p75 = known_mc.quantile(0.75)
            p40 = known_mc.quantile(0.40)
            def sp500_tier(mc):
                if mc <= 0:     return "small"   # unknown → treat as small
                if mc >= p75:   return "mega"
                if mc >= p40:   return "large"
                return "small"
            df["tier"]  = df["mc"].apply(sp500_tier)
            df["color"] = df["tier"].map(TIER_COLORS).fillna("#6b7280")

        # ── Step 5: Relative strength vs large/mega-cap sector peers ──────────
        set_status("running", "Computing relative-strength scores…", 80)

        # Global large-cap averages as fallback
        glc = df[df["tier"].isin(["large", "mega"])]
        g3  = float(glc["r3m"].mean()) if len(glc) else 0.0
        g1  = float(glc["r1m"].mean()) if len(glc) else 0.0

        for sector, grp in df.groupby("sector"):
            lc  = grp[grp["tier"].isin(["large", "mega"])]
            lc3 = float(lc["r3m"].mean()) if len(lc) and not lc["r3m"].isna().all() else g3
            lc1 = float(lc["r1m"].mean()) if len(lc) and not lc["r1m"].isna().all() else g1
            m   = df["sector"] == sector
            df.loc[m, "rs3m"] = (df.loc[m, "r3m"] - lc3).round(2)
            df.loc[m, "rs1m"] = (df.loc[m, "r1m"] - lc1).round(2)

        # Composite score: 60% weight on 3-month RS, 40% on 1-month RS
        df["score"] = (
            0.6 * df["rs3m"].fillna(0) +
            0.4 * df["rs1m"].fillna(0)
        ).round(2)

        # Percentile rank within the small + mid universe
        sm = df["tier"].isin(["small", "mid"])
        if sm.sum() > 0:
            df.loc[sm, "pct"] = (
                df.loc[sm, "score"].rank(pct=True) * 100
            ).round(1)

        # ── Step 6: Sector summary ────────────────────────────────────────────
        sector_rows = []
        for sec, g in df.groupby("sector"):
            sm_g = g[g["tier"].isin(["small", "mid"])]
            top  = (sm_g.sort_values("score", ascending=False).iloc[0]["ticker"]
                    if len(sm_g) else None)
            avg  = (round(float(sm_g["rs3m"].mean()), 2)
                    if len(sm_g) and not sm_g["rs3m"].isna().all() else None)
            sector_rows.append({
                "sector":    sec,
                "total":     int(len(g)),
                "sm_count":  int(len(sm_g)),
                "avg_rs3m":  avg,
                "top":       top,
            })

        # ── Step 7: Persist ───────────────────────────────────────────────────
        set_status("running", "Saving results…", 95)
        out = {
            "updated_at": datetime.now().isoformat(),
            "n":          len(df),
            "stocks":     df.replace({np.nan: None}).to_dict(orient="records"),
            "sectors":    sector_rows,
        }
        CACHE_FILE.write_text(json.dumps(out), encoding="utf-8")
        set_status("ready", f"Ready — {len(df)} stocks analysed", 100)
        print(f"[AlphaScout] OK {len(df)} stocks processed at {datetime.now():%H:%M:%S}")

    except Exception as exc:
        set_status("error", str(exc), 0)
        print(f"[AlphaScout] ERR {exc}")
        raise


# ── Live price refresh (lightweight — runs every 5 min) ──────────────────────
def refresh_live_prices():
    """Batch-download latest closes and store change% for every tracked stock."""
    if not CACHE_FILE.exists():
        return
    d       = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    tickers = [s["ticker"] for s in d["stocks"]]

    raw = yf.download(
        tickers, period="2d", interval="1d",
        auto_adjust=True, progress=False, threads=True,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if not isinstance(closes, pd.DataFrame):
        closes = closes.to_frame()

    prices: dict = {}
    for sym in tickers:
        if sym not in closes.columns:
            continue
        s = closes[sym].dropna()
        if len(s) >= 2:
            today = round(float(s.iloc[-1]), 2)
            prev  = float(s.iloc[-2])
            prices[sym] = {"price": today, "chg": round((today / prev - 1) * 100, 2)}
        elif len(s) == 1:
            prices[sym] = {"price": round(float(s.iloc[-1]), 2), "chg": None}

    PRICES_FILE.write_text(
        json.dumps({"updated_at": datetime.now().isoformat(), "prices": prices}),
        encoding="utf-8",
    )
    print(f"[AlphaScout] Prices updated: {len(prices)} tickers")


def _price_loop():
    """Background daemon: refresh live prices every 5 minutes."""
    time.sleep(10)                        # let startup settle first
    while True:
        try:
            if CACHE_FILE.exists():
                refresh_live_prices()
        except Exception as exc:
            print(f"[AlphaScout] Price loop error: {exc}")
        time.sleep(300)                   # 5 minutes


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AlphaScout")

VALID_SORTS = {"score", "rs3m", "rs1m", "r3m", "r1m", "r6m", "mc", "price"}


@app.get("/api/status")
def api_status():
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    if CACHE_FILE.exists():
        d = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return {"status": "ready", "ts": d["updated_at"], "n": d["n"], "progress": 100}
    return {"status": "no_data", "message": "No data yet", "progress": 0}


@app.post("/api/refresh")
def api_refresh():
    threading.Thread(target=compute, daemon=True).start()
    return {"ok": True}


@app.get("/api/screener")
def api_screener(
    tier:    str = Query("all"),
    sector:  str = Query("all"),
    sort_by: str = Query("score"),
    limit:   int = Query(500),
):
    if not CACHE_FILE.exists():
        return JSONResponse({"error": "No data — call /api/refresh first"}, 503)

    d = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    stocks = d["stocks"]

    if tier   != "all": stocks = [s for s in stocks if s.get("tier")   == tier]
    if sector != "all": stocks = [s for s in stocks if s.get("sector") == sector]

    if sort_by not in VALID_SORTS:
        sort_by = "score"

    stocks = sorted(
        stocks,
        key=lambda x: (x.get(sort_by) is None, -(x.get(sort_by) or 0)),
    )

    return {
        "updated_at": d["updated_at"],
        "n":          len(stocks),
        "stocks":     stocks[:limit],
        "sectors":    d.get("sectors", []),
    }


@app.get("/api/prices/live")
def api_live_prices():
    """Latest price + daily change% for every tracked stock."""
    if not PRICES_FILE.exists():
        # Kick off an async refresh; return empty for now
        threading.Thread(target=refresh_live_prices, daemon=True).start()
        return {"prices": {}, "updated_at": None, "refreshing": True}
    return json.loads(PRICES_FILE.read_text(encoding="utf-8"))


@app.get("/api/chart/{ticker}")
def api_chart(ticker: str, period: str = Query("3mo")):
    """OHLCV daily bars for a single ticker (for the chart panel)."""
    valid = {"1mo", "3mo", "6mo", "1y", "2y"}
    if period not in valid:
        period = "3mo"
    ticker = ticker.upper()
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if hist.empty:
            return JSONResponse({"error": f"No data for {ticker}"}, 404)
        rows = []
        for dt, row in hist.iterrows():
            try:
                rows.append({
                    "t": dt.strftime("%Y-%m-%d"),
                    "o": round(float(row["Open"]),   2),
                    "h": round(float(row["High"]),   2),
                    "l": round(float(row["Low"]),    2),
                    "c": round(float(row["Close"]),  2),
                    "v": int(row["Volume"]) if not pd.isna(row.get("Volume", 0)) else 0,
                })
            except (ValueError, TypeError):
                continue
        return {"ticker": ticker, "period": period, "data": rows}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)


# Serve the frontend (must come last — catches all unmatched routes)
app.mount("/", StaticFiles(directory=str(FRONT), html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_price_loop, daemon=True).start()
    if not CACHE_FILE.exists():
        print("[AlphaScout] No cache found — starting initial data fetch…")
        threading.Thread(target=compute, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
