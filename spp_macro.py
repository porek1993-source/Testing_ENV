"""
Stock Picker Pro – Macro Data Module (FRED API + Static Fallback)
=================================================================
Provides live macroeconomic indicators from the Federal Reserve
Economic Data (FRED) API. Falls back to static calendar when
FRED API key is not configured.
"""

import os
import datetime as dt
from typing import Any, Dict, List, Optional

import streamlit as st
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_fred_key() -> str:
    """Get FRED API key from Streamlit secrets or env."""
    try:
        return st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY", "")
    except Exception:
        return os.environ.get("FRED_API_KEY", "")


# ---------------------------------------------------------------------------
# FRED API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fred_series_latest(series_id: str) -> Optional[float]:
    """Fetch the latest value for a FRED series."""
    key = _get_fred_key()
    if not key:
        return None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        observations = data.get("observations", [])
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                return float(val)
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fred_series_history(series_id: str, start_date: str = "2020-01-01") -> List[Dict]:
    """Fetch historical values for a FRED series."""
    key = _get_fred_key()
    if not key:
        return []
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        result = []
        for obs in data.get("observations", []):
            val = obs.get("value", ".")
            if val != ".":
                result.append({
                    "date": obs["date"],
                    "value": float(val),
                })
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Live Macro Dashboard Data
# ---------------------------------------------------------------------------

# FRED series IDs for key macro indicators
MACRO_SERIES = {
    "Fed Funds Rate": {
        "series_id": "FEDFUNDS",
        "description": "Aktuální úroková sazba Fedu",
        "format": "pct",
        "importance": "Critical",
    },
    "10Y Treasury Yield": {
        "series_id": "DGS10",
        "description": "Výnos 10letého amerického dluhopisu",
        "format": "pct",
        "importance": "High",
    },
    "2Y Treasury Yield": {
        "series_id": "DGS2",
        "description": "Výnos 2letého amerického dluhopisu (inverzní křivka?)",
        "format": "pct",
        "importance": "High",
    },
    "CPI (YoY %)": {
        "series_id": "CPIAUCSL",
        "description": "Meziroční inflace (Consumer Price Index)",
        "format": "yoy_pct",
        "importance": "High",
    },
    "Unemployment Rate": {
        "series_id": "UNRATE",
        "description": "Míra nezaměstnanosti v USA",
        "format": "pct",
        "importance": "High",
    },
    "VIX (Fear Index)": {
        "series_id": "VIXCLS",
        "description": "Index strachu – volatilita S&P 500",
        "format": "num",
        "importance": "High",
    },
    "GDP Growth (QoQ %)": {
        "series_id": "A191RL1Q225SBEA",
        "description": "Reálný růst HDP (mezičtvrtletní, anualizovaný)",
        "format": "pct",
        "importance": "Medium",
    },
    "ISM Manufacturing PMI": {
        "series_id": "MANEMP",
        "description": "Zaměstnanost ve výrobním sektoru (proxy PMI)",
        "format": "num",
        "importance": "Medium",
    },
}


@st.cache_data(ttl=3600, show_spinner="📡 Stahuji živá makro data z FRED...")
def fetch_live_macro_indicators() -> List[Dict]:
    """
    Fetch all key macro indicators from FRED.
    Returns a list of dicts ready for display in a dashboard.
    """
    key = _get_fred_key()
    if not key:
        return []

    results = []
    for name, config in MACRO_SERIES.items():
        value = _fred_series_latest(config["series_id"])
        if value is not None:
            # Format the value
            if config["format"] == "pct":
                display = f"{value:.2f}%"
            elif config["format"] == "yoy_pct":
                display = f"{value:,.1f}"  # CPI is an index – show raw
            else:
                display = f"{value:,.1f}"

            results.append({
                "Indikátor": name,
                "Hodnota": display,
                "raw_value": value,
                "Popis": config["description"],
                "Důležitost": config["importance"],
            })

    return results


def get_yield_curve_data() -> Dict[str, Optional[float]]:
    """Get yield curve data points (2Y, 5Y, 10Y, 30Y) for inversion detection."""
    series = {
        "2Y": "DGS2",
        "5Y": "DGS5",
        "10Y": "DGS10",
        "30Y": "DGS30",
    }
    result = {}
    for label, sid in series.items():
        result[label] = _fred_series_latest(sid)
    return result


def is_yield_curve_inverted() -> Optional[bool]:
    """Check if the yield curve is inverted (2Y > 10Y)."""
    yc = get_yield_curve_data()
    y2 = yc.get("2Y")
    y10 = yc.get("10Y")
    if y2 is not None and y10 is not None:
        return y2 > y10
    return None


def get_market_regime_macro() -> str:
    """
    Determine macro-level market regime based on FRED data.
    Returns: 'Risk-On', 'Risk-Off', 'Neutral', or 'Unavailable'.
    """
    key = _get_fred_key()
    if not key:
        return "Unavailable"

    vix = _fred_series_latest("VIXCLS")
    fed_rate = _fred_series_latest("FEDFUNDS")
    unemployment = _fred_series_latest("UNRATE")

    signals = []

    if vix is not None:
        if vix > 30:
            signals.append("risk-off")
        elif vix < 15:
            signals.append("risk-on")
        else:
            signals.append("neutral")

    if unemployment is not None:
        if unemployment > 5.5:
            signals.append("risk-off")
        elif unemployment < 4.0:
            signals.append("risk-on")
        else:
            signals.append("neutral")

    if not signals:
        return "Unavailable"

    risk_off_count = signals.count("risk-off")
    risk_on_count = signals.count("risk-on")

    if risk_off_count > risk_on_count:
        return "🔴 Risk-Off (Defenzivní)"
    elif risk_on_count > risk_off_count:
        return "🟢 Risk-On (Růstový)"
    else:
        return "🟡 Neutrální"


# ---------------------------------------------------------------------------
# Static Macro Calendar (fallback + upcoming events)
# ---------------------------------------------------------------------------

STATIC_MACRO_CALENDAR = [
    {"date": "2026-02-20", "event": "FOMC Minutes Release", "importance": "High"},
    {"date": "2026-03-06", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-03-11", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-03-18", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-03-25", "event": "US GDP (Q4 2025 Final)", "importance": "Medium"},
    {"date": "2026-04-03", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-04-08", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-04-29", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-05-15", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-06-10", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
]


def get_macro_calendar() -> List[Dict]:
    """Return macro calendar events (static for now, could integrate economic calendar API)."""
    return STATIC_MACRO_CALENDAR
