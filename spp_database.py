"""
Stock Picker Pro – Database Module (Supabase REST API + JSON fallback)
======================================================================
Handles persistent storage for watchlists, memos, snapshots, alerts,
and AI report cache. Falls back to local JSON files when Supabase is
not configured.

Refactored to use `requests` directly to avoid dependency hell with
`supabase` python client (which requires C++ build tools for pyiceberg).
"""

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Conf & Helpers
# ---------------------------------------------------------------------------

def _get_secrets():
    """Retrieve Supabase URL and KEY from secrets or env."""
    url = st.secrets.get("SUPABASE_URL", "") if hasattr(st, "secrets") else os.environ.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "") if hasattr(st, "secrets") else os.environ.get("SUPABASE_KEY", "")
    return url, key

def is_supabase_available() -> bool:
    """Check whether we have configuration for Supabase."""
    url, key = _get_secrets()
    return bool(url and key)

def _sb_request(method: str, endpoint: str, params: Dict = None, json_data: Dict = None, headers_extra: Dict = None) -> Any:
    """Execute a raw REST API request to Supabase."""
    url, key = _get_secrets()
    if not url or not key:
        return None

    # Ensure URL ends with /rest/v1/
    base_url = url.rstrip("/") + "/rest/v1/"
    api_url = base_url + endpoint

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # To get back the inserted/updated data
    }
    if headers_extra:
        headers.update(headers_extra)

    try:
        if method == "GET":
            resp = requests.get(api_url, headers=headers, params=params, timeout=5)
        elif method == "POST":
            resp = requests.post(api_url, headers=headers, json=json_data, params=params, timeout=5)
        elif method == "PATCH":
            resp = requests.patch(api_url, headers=headers, json=json_data, params=params, timeout=5)
        elif method == "DELETE":
            resp = requests.delete(api_url, headers=headers, params=params, timeout=5)
        else:
            return None

        # Check for success
        if 200 <= resp.status_code < 300:
            return resp.json()
        
        # Log error if debugging (optional)
        # st.write(f"Supabase Error {resp.status_code}: {resp.text}")
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Local JSON helpers (fallback)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
MEMOS_PATH = os.path.join(DATA_DIR, "memos.json")

def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, obj: Any) -> None:
    _ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================================
# WATCHLIST
# ============================================================================

def get_watchlist(user_id: str = "default") -> Dict[str, Any]:
    """Return watchlist items."""
    if is_supabase_available():
        # GET /watchlists?user_id=eq.default&select=*
        data = _sb_request("GET", "watchlists", params={"user_id": f"eq.{user_id}", "select": "*"})
        if data is not None:
            items = {}
            for row in data:
                items[row["ticker"]] = {
                    "company_name": row.get("company_name", ""),
                    "added_at": row.get("added_at", ""),
                    "notes": row.get("notes", ""),
                    "target_price": row.get("target_price"),
                }
            return {"items": items}
            
    # Fallback
    return _load_json(WATCHLIST_PATH, {"items": {}})


def add_to_watchlist(ticker: str, company_name: str = "", notes: str = "", target_price: Optional[float] = None, user_id: str = "default") -> bool:
    """Add or update a ticker in the watchlist."""
    if is_supabase_available():
        payload = {
            "user_id": user_id,
            "ticker": ticker.upper(),
            "company_name": company_name,
            "notes": notes,
            "target_price": target_price,
            # triggers 'added_at' default on server if new
        }
        # UPSERT via POST with Prefer: resolution=merge-duplicates in headers?
        # Supabase generic upsert usually works via POST with on_conflict param if using client,
        # but via REST it's often POST with Prefer: resolution=merge-duplicates.
        resp = _sb_request("POST", "watchlists", json_data=payload, headers_extra={"Prefer": "resolution=merge-duplicates"})
        if resp is not None:
            return True

    # Fallback
    data = _load_json(WATCHLIST_PATH, {"items": {}})
    data["items"][ticker.upper()] = {
        "company_name": company_name,
        "notes": notes,
        "target_price": target_price,
        "added_at": dt.datetime.now().isoformat(),
    }
    _save_json(WATCHLIST_PATH, data)
    return True


def remove_from_watchlist(ticker: str, user_id: str = "default") -> bool:
    """Remove a ticker."""
    if is_supabase_available():
        # DELETE /watchlists?user_id=eq...&ticker=eq...
        resp = _sb_request("DELETE", "watchlists", params={"user_id": f"eq.{user_id}", "ticker": f"eq.{ticker.upper()}"})
        if resp is not None:
            return True

    # Fallback
    data = _load_json(WATCHLIST_PATH, {"items": {}})
    data["items"].pop(ticker.upper(), None)
    _save_json(WATCHLIST_PATH, data)
    return True


def set_watchlist(data: Dict[str, Any], user_id: str = "default") -> None:
    """Batch setter."""
    if is_supabase_available():
        items = data.get("items", {})
        for ticker, meta in items.items():
            add_to_watchlist(ticker, meta.get("company_name"), meta.get("notes"), meta.get("target_price"), user_id)
        return

    _save_json(WATCHLIST_PATH, data)


# ============================================================================
# MEMOS
# ============================================================================

def get_memos(user_id: str = "default") -> Dict[str, Any]:
    if is_supabase_available():
        data = _sb_request("GET", "memos", params={"user_id": f"eq.{user_id}", "select": "*"})
        if data is not None:
            memos = {}
            for row in data:
                memos[row["ticker"]] = {
                    "thesis": row.get("thesis", ""),
                    "drivers": row.get("drivers", ""),
                    "risks": row.get("risks", ""),
                    "catalysts": row.get("catalysts", ""),
                    "buy_conditions": row.get("buy_conditions", ""),
                    "notes": row.get("notes", ""),
                    "created_at": row.get("created_at", ""),
                    "updated_at": row.get("updated_at", ""),
                }
            return {"memos": memos}

    return _load_json(MEMOS_PATH, {"memos": {}})


def save_memo(ticker: str, thesis: str = "", drivers: str = "", risks: str = "", catalysts: str = "", buy_conditions: str = "", notes: str = "", user_id: str = "default") -> bool:
    """Save or update a memo."""
    if is_supabase_available():
        payload = {
            "user_id": user_id,
            "ticker": ticker.upper(),
            "thesis": thesis,
            "drivers": drivers,
            "risks": risks,
            "catalysts": catalysts,
            "buy_conditions": buy_conditions,
            "notes": notes,
            "updated_at": dt.datetime.now().isoformat(),
        }
        resp = _sb_request("POST", "memos", json_data=payload, headers_extra={"Prefer": "resolution=merge-duplicates"})
        if resp is not None:
            return True

    data = _load_json(MEMOS_PATH, {"memos": {}})
    data["memos"][ticker.upper()] = {
        "thesis": thesis, "drivers": drivers, "risks": risks,
        "catalysts": catalysts, "buy_conditions": buy_conditions, "notes": notes,
        "updated_at": dt.datetime.now().isoformat(),
    }
    _save_json(MEMOS_PATH, data)
    return True


def set_memos(data: Dict[str, Any], user_id: str = "default") -> None:
    if is_supabase_available():
        memos = data.get("memos", {})
        for ticker, memo in memos.items():
            save_memo(
                ticker, memo.get("thesis"), memo.get("drivers"), memo.get("risks"),
                memo.get("catalysts"), memo.get("buy_conditions"), memo.get("notes"), user_id
            )
        return
    _save_json(MEMOS_PATH, data)


# ============================================================================
# SNAPSHOTS
# ============================================================================

def save_snapshot(ticker: str, current_price: Optional[float] = None, fair_value_dcf: Optional[float] = None, scorecard: float = 0.0, mos_dcf: Optional[float] = None, verdict: str = "", piotroski: int = 0, altman_z: Optional[float] = None, pe: Optional[float] = None, insider_signal: float = 0.0, extra_data: Optional[Dict] = None) -> bool:
    if not is_supabase_available():
        return False
        
    payload = {
        "ticker": ticker.upper(),
        "snapshot_date": dt.date.today().isoformat(),
        "current_price": current_price,
        "fair_value_dcf": fair_value_dcf,
        "scorecard": scorecard,
        "mos_dcf": mos_dcf,
        "verdict": verdict,
        "piotroski": piotroski,
        "altman_z": altman_z,
        "pe": pe,
        "insider_signal": insider_signal,
        "data": json.dumps(extra_data) if extra_data else None,
    }
    # No upsert here usually, we just insert. But if unique constraint exists on (ticker, date), then use upsert.
    # Assuming standard insert logic for history.
    resp = _sb_request("POST", "snapshots", json_data=payload)
    return resp is not None


def get_snapshot_history(ticker: str, days: int = 90) -> List[Dict]:
    if not is_supabase_available():
        return []

    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    # GET /snapshots?ticker=eq...&snapshot_date=gte...&order=snapshot_date.desc
    params = {
        "ticker": f"eq.{ticker.upper()}",
        "snapshot_date": f"gte.{cutoff}",
        "order": "snapshot_date.desc",
        "select": "*"
    }
    data = _sb_request("GET", "snapshots", params=params)
    return data or []


# ============================================================================
# ALERTS
# ============================================================================

def create_alert(ticker: str, alert_type: str, threshold: float, user_id: str = "default") -> bool:
    if not is_supabase_available():
        return False
        
    payload = {
        "user_id": user_id,
        "ticker": ticker.upper(),
        "alert_type": alert_type,
        "threshold": threshold,
        "is_active": True,
    }
    resp = _sb_request("POST", "alerts", json_data=payload)
    return resp is not None


def get_active_alerts(user_id: str = "default") -> List[Dict]:
    if not is_supabase_available():
        return []
        
    params = {
        "user_id": f"eq.{user_id}",
        "is_active": "eq.TRUE",
        "select": "*"
    }
    data = _sb_request("GET", "alerts", params=params)
    return data or []


def check_and_trigger_alerts(ticker: str, current_price: float, scorecard: float = 0.0, user_id: str = "default") -> List[Dict]:
    # Reuse get_active_alerts logic
    alerts = get_active_alerts(user_id)
    triggered = []
    
    for alert in alerts:
        if alert["ticker"] != ticker.upper():
            continue
            
        fire = False
        if alert["alert_type"] == "price_below" and current_price <= alert["threshold"]:
            fire = True
        elif alert["alert_type"] == "price_above" and current_price >= alert["threshold"]:
            fire = True
        elif alert["alert_type"] == "scorecard_change" and abs(scorecard - alert["threshold"]) > 10:
            fire = True
            
        if fire:
            triggered.append(alert)
            # Deactivate
            _sb_request("PATCH", "alerts", params={"id": f"eq.{alert['id']}"}, json_data={
                "is_active": False,
                "triggered_at": dt.datetime.now().isoformat()
            })
            
    return triggered


# ============================================================================
# AI REPORT CACHE
# ============================================================================

def cache_ai_report(ticker: str, report_data: Dict, model: str = "") -> bool:
    if not is_supabase_available():
        return False
        
    payload = {
        "ticker": ticker.upper(),
        "report_data": json.dumps(report_data, ensure_ascii=False),
        "model_used": model,
        # created_at is auto
    }
    resp = _sb_request("POST", "ai_reports", json_data=payload)
    return resp is not None


def get_cached_ai_report(ticker: str, max_age_hours: int = 24) -> Optional[Dict]:
    if not is_supabase_available():
        return None

    cutoff = (dt.datetime.now() - dt.timedelta(hours=max_age_hours)).isoformat()
    params = {
        "ticker": f"eq.{ticker.upper()}",
        "created_at": f"gte.{cutoff}",
        "order": "created_at.desc",
        "limit": 1,
        "select": "*"
    }
    data = _sb_request("GET", "ai_reports", params=params)
    
    if data and len(data) > 0:
        raw = data[0].get("report_data")
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    return None

# ============================================================================
# PORTFOLIO (New in v10)
# ============================================================================

def get_portfolio(user_id: str = "default") -> List[Dict]:
    if not is_supabase_available():
        return []
        
    params = {"user_id": f"eq.{user_id}", "select": "*"}
    data = _sb_request("GET", "portfolio", params=params)
    return data or []

def add_portfolio_holding(ticker: str, shares: float, buy_price: float, buy_date: str, user_id: str = "default") -> bool:
    if not is_supabase_available():
        return False

    payload = {
        "user_id": user_id,
        "ticker": ticker.upper(),
        "shares": shares,
        "buy_price": buy_price,
        "buy_date": buy_date
    }
    resp = _sb_request("POST", "portfolio", json_data=payload)
    return resp is not None

def remove_portfolio_holding(ticker: str, user_id: str = "default") -> bool:
    if not is_supabase_available():
        return False
        
    params = {"user_id": f"eq.{user_id}", "ticker": f"eq.{ticker.upper()}"}
    _sb_request("DELETE", "portfolio", params=params)
    return True
