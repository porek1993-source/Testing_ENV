# ─────────────────────────────────────────────────────────────
# spp_screener.py  –  Mini Stock Screener & Portfolio Tracker
# Stock Picker Pro v10.0
# ─────────────────────────────────────────────────────────────
"""
Provides:
 1) run_screener()          – scan a list of tickers with configurable filters
 2) get_portfolio()         – load portfolio from Supabase
 3) add_portfolio_holding() – add a holding to the portfolio
 4) remove_portfolio_holding() – remove a holding
 5) get_portfolio_summary() – aggregate P&L, sector breakdown
 6) fetch_twelvedata_indicators() – Twelve Data tech indicator fallback
"""

from __future__ import annotations
import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
import spp_database as db

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

# ──────────────────── helpers ────────────────────

def _get_secret(name: str, default: str = "") -> str:
    if st:
        try:
            return st.secrets.get(name, os.environ.get(name, default))
        except Exception:
            pass
    return os.environ.get(name, default)


def safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return v if v == v else None  # NaN check
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════
# 1.  TWELVE DATA – Technical Indicators Fallback
# ═══════════════════════════════════════════════════════════════

TWELVEDATA_BASE = "https://api.twelvedata.com"

def fetch_twelvedata_indicators(
    ticker: str,
    indicators: Optional[List[str]] = None,
    interval: str = "1day",
    outputsize: int = 30,
) -> Dict[str, Any]:
    """
    Fetch pre-computed indicators from Twelve Data API.
    Returns dict keyed by indicator name → list of values (most recent first).

    Default indicators: RSI, MACD, BBANDS, STOCH, ADX, ATR
    Free tier: 8 requests/min, 800/day
    """
    api_key = _get_secret("TWELVEDATA_API_KEY")
    if not api_key:
        return {}

    if indicators is None:
        indicators = ["rsi", "macd", "bbands", "stoch", "adx", "atr"]

    results: Dict[str, Any] = {}

    for ind in indicators:
        try:
            params = {
                "symbol": ticker,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": api_key,
            }
            resp = requests.get(
                f"{TWELVEDATA_BASE}/{ind}", params=params, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if "values" in data:
                    results[ind] = data["values"]
                elif "status" in data and data["status"] == "error":
                    results[ind] = {"error": data.get("message", "Unknown error")}
            time.sleep(0.15)  # rate limit
        except Exception as e:
            results[ind] = {"error": str(e)}

    return results


def get_twelvedata_rsi(ticker: str) -> Optional[float]:
    """Quick helper: get latest RSI from Twelve Data (fallback for yfinance)."""
    data = fetch_twelvedata_indicators(ticker, indicators=["rsi"], outputsize=1)
    try:
        return float(data["rsi"][0]["rsi"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def get_twelvedata_macd(ticker: str) -> Optional[Dict[str, float]]:
    """Quick helper: get latest MACD from Twelve Data."""
    data = fetch_twelvedata_indicators(ticker, indicators=["macd"], outputsize=1)
    try:
        v = data["macd"][0]
        return {
            "macd": float(v["macd"]),
            "signal": float(v["macd_signal"]),
            "histogram": float(v["macd_hist"]),
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════
# 2.  MINI STOCK SCREENER
# ═══════════════════════════════════════════════════════════════

# Default universe (popular stocks for quick screening)
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "TMO", "WMT", "MCD", "CSCO",
    "ACN", "LIN", "ABT", "CRM", "DHR", "NFLX", "TXN", "AMD", "NEE",
    "PM", "UNP", "INTC", "LOW", "HON", "QCOM", "RTX", "AMAT", "CAT",
    "BA", "GS", "SBUX", "BKNG", "ISRG",
]

def _fetch_quick_metrics(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch essential metrics for a single ticker (for screener table)."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or not info.get("regularMarketPrice"):
            return None

        pe = safe_float(info.get("trailingPE"))
        fwd_pe = safe_float(info.get("forwardPE"))
        pb = safe_float(info.get("priceToBook"))
        div_yield = safe_float(info.get("dividendYield"))
        mkt_cap = safe_float(info.get("marketCap"))
        rev_growth = safe_float(info.get("revenueGrowth"))
        op_margin = safe_float(info.get("operatingMargins"))
        roe = safe_float(info.get("returnOnEquity"))
        beta = safe_float(info.get("beta"))
        price = safe_float(info.get("regularMarketPrice"))
        target = safe_float(info.get("targetMeanPrice"))
        sector = info.get("sector", "—")

        upside = None
        if target and price and price > 0:
            upside = round((target / price - 1) * 100, 1)

        return {
            "Ticker": ticker,
            "Sector": sector,
            "Price": price,
            "P/E": pe,
            "Fwd P/E": fwd_pe,
            "P/B": pb,
            "Div %": round(div_yield * 100, 2) if div_yield else None,
            "Rev Grw %": round(rev_growth * 100, 1) if rev_growth else None,
            "Op Margin %": round(op_margin * 100, 1) if op_margin else None,
            "ROE %": round(roe * 100, 1) if roe else None,
            "Beta": round(beta, 2) if beta else None,
            "Mkt Cap": mkt_cap,
            "Upside %": upside,
        }
    except Exception:
        return None


def run_screener(
    universe: Optional[List[str]] = None,
    max_pe: Optional[float] = None,
    max_pb: Optional[float] = None,
    min_div_yield: Optional[float] = None,
    min_roe: Optional[float] = None,
    min_upside: Optional[float] = None,
    sector_filter: Optional[str] = None,
    max_workers: int = 6,
) -> "pd.DataFrame":
    """
    Scan *universe* tickers in parallel, apply filters, return sorted DataFrame.

    Args:
        universe:    list of tickers (default: 50 popular US stocks)
        max_pe:      filter P/E ≤ value
        max_pb:      filter P/B ≤ value
        min_div_yield: min dividend yield %
        min_roe:     min ROE %
        min_upside:  min analyst upside %
        sector_filter: sector name substring
        max_workers: parallel threads (default 6)
    """
    if pd is None:
        return []  # type: ignore

    tickers = universe or DEFAULT_UNIVERSE
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_quick_metrics, t): t for t in tickers
        }
        for future in as_completed(futures):
            data = future.result()
            if data is not None:
                results.append(data)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Apply filters
    if max_pe is not None:
        df = df[df["P/E"].notna() & (df["P/E"] <= max_pe)]
    if max_pb is not None:
        df = df[df["P/B"].notna() & (df["P/B"] <= max_pb)]
    if min_div_yield is not None:
        df = df[df["Div %"].notna() & (df["Div %"] >= min_div_yield)]
    if min_roe is not None:
        df = df[df["ROE %"].notna() & (df["ROE %"] >= min_roe)]
    if min_upside is not None:
        df = df[df["Upside %"].notna() & (df["Upside %"] >= min_upside)]
    if sector_filter:
        df = df[df["Sector"].str.contains(sector_filter, case=False, na=False)]

    # Sort by upside descending
    if "Upside %" in df.columns:
        df = df.sort_values("Upside %", ascending=False, na_position="last")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 3.  PORTFOLIO TRACKER (Supabase-backed)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 3.  PORTFOLIO TRACKER (Supabase-backed via spp_database)
# ═══════════════════════════════════════════════════════════════

def get_portfolio(user_id: str = "default") -> List[Dict[str, Any]]:
    """Load portfolio holdings via spp_database."""
    return db.get_portfolio(user_id)


def add_portfolio_holding(
    ticker: str,
    shares: float,
    buy_price: float,
    buy_date: Optional[str] = None,
    user_id: str = "default",
) -> bool:
    """Add or update a portfolio holding via spp_database."""
    b_date = buy_date or dt.date.today().isoformat()
    return db.add_portfolio_holding(ticker, shares, buy_price, b_date, user_id)


def remove_portfolio_holding(ticker: str, user_id: str = "default") -> bool:
    """Remove a holding via spp_database."""
    return db.remove_portfolio_holding(ticker, user_id)


def get_portfolio_summary(
    holdings: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Compute portfolio P&L, sector breakdown, total value.
    Returns dict with keys: holdings, total_value, total_cost, total_pnl,
                            total_pnl_pct, sector_breakdown
    """
    if holdings is None:
        holdings = get_portfolio(user_id)

    if not holdings:
        return {
            "holdings": [], "total_value": 0, "total_cost": 0,
            "total_pnl": 0, "total_pnl_pct": 0, "sector_breakdown": {},
        }

    enriched = []
    total_value = 0.0
    total_cost = 0.0
    sectors: Dict[str, float] = {}

    for h in holdings:
        ticker = h.get("ticker", "")
        shares = float(h.get("shares", 0))
        buy_price = float(h.get("buy_price", 0))
        cost = shares * buy_price

        current_price = None
        sector = "Unknown"
        if yf:
            try:
                info = yf.Ticker(ticker).info
                current_price = safe_float(info.get("regularMarketPrice"))
                sector = info.get("sector", "Unknown")
            except Exception:
                pass

        current_val = (shares * current_price) if current_price else 0
        pnl = current_val - cost
        pnl_pct = round((pnl / cost) * 100, 2) if cost > 0 else 0

        total_value += current_val
        total_cost += cost
        sectors[sector] = sectors.get(sector, 0) + current_val

        enriched.append({
            "ticker": ticker,
            "shares": shares,
            "buy_price": buy_price,
            "current_price": current_price,
            "cost": round(cost, 2),
            "value": round(current_val, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": pnl_pct,
            "sector": sector,
            "buy_date": h.get("buy_date", ""),
        })

    return {
        "holdings": enriched,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_value - total_cost, 2),
        "total_pnl_pct": round(((total_value / total_cost) - 1) * 100, 2) if total_cost > 0 else 0,
        "sector_breakdown": sectors,
    }
