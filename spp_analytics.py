"""
Stock Picker Pro – Analytické výpočetní funkce
===============================================
Obsahuje: fundamental analýza, technická analýza, DCF, scorecard,
          nové metriky (Net Debt/EBITDA, Rule of 40, FCF Margin,
          Dividend Safety, historické P/E, Radar Chart data), opravené bugy.
"""

import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pomocné funkce (forward-deklarace – jsou definovány v hlavním souboru,
# ale importujeme je sem přes parametry nebo předáváme explicitně)
# ---------------------------------------------------------------------------

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


# ============================================================================
# FUNDAMENTÁLNÍ ANALÝZA
# ============================================================================

def calculate_roic(info: Dict[str, Any]) -> Optional[float]:
    """Aproximace ROIC: NOPAT / Invested Capital."""
    try:
        ebit = safe_float(info.get("ebit"))
        if ebit is None:
            ebitda = safe_float(info.get("ebitda"))
            da = safe_float(info.get("depreciationAndAmortization") or info.get("totalDepreciationAndAmortization"))
            if ebitda is not None:
                ebit = ebitda - (da or 0)
        nopat = ebit * 0.79 if ebit is not None else None
        invested_capital = (safe_float(info.get("totalDebt")) or 0) + (safe_float(info.get("totalStockholderEquity")) or 0)
        return safe_div(nopat, invested_capital)
    except Exception:
        return None


def calculate_graham_number(info: Dict[str, Any]) -> Optional[float]:
    """Graham Number: konzervativní fair value = sqrt(22.5 * EPS * BVPS)."""
    try:
        eps = safe_float(info.get("trailingEps"))
        bvps = safe_float(info.get("bookValue"))
        if eps and bvps and eps > 0 and bvps > 0:
            return math.sqrt(22.5 * eps * bvps)
        return None
    except Exception:
        return None


def calculate_net_debt_ebitda(info: Dict[str, Any]) -> Optional[float]:
    """
    Čistý dluh / EBITDA – intuitivnější než D/E.
    Vrací počet let, za kolik firma splatí čistý dluh z provozního zisku.
    """
    try:
        total_debt = safe_float(info.get("totalDebt")) or 0
        total_cash = safe_float(info.get("totalCash")) or 0
        ebitda = safe_float(info.get("ebitda"))
        if ebitda is None or ebitda <= 0:
            return None
        net_debt = total_debt - total_cash
        return round(net_debt / ebitda, 2)
    except Exception:
        return None


def calculate_rule_of_40(info: Dict[str, Any]) -> Optional[float]:
    """
    Rule of 40 pro SaaS/tech firmy: Rev. Growth (%) + Op. Margin (%).
    > 40 = firma vyvažuje růst a ziskovost.
    """
    try:
        rev_growth = safe_float(info.get("revenueGrowth"))
        op_margin = safe_float(info.get("operatingMargins"))
        if rev_growth is None or op_margin is None:
            return None
        return round((rev_growth * 100) + (op_margin * 100), 1)
    except Exception:
        return None


def calculate_fcf_margin(info: Dict[str, Any], fcf: Optional[float]) -> Optional[float]:
    """
    FCF Margin = FCF / Revenue.
    Lepší než FCF Yield pro mezifiremní srovnání.
    """
    try:
        revenue = safe_float(info.get("totalRevenue"))
        if revenue is None or revenue <= 0 or fcf is None:
            return None
        return fcf / revenue
    except Exception:
        return None


def calculate_dividend_safety_score(info: Dict[str, Any], fcf: Optional[float]) -> Tuple[int, str]:
    """
    Dividend Safety Score (0–5): hodnotí udržitelnost dividendy.
    Vrací (score, popis).
    """
    try:
        div_yield = safe_float(info.get("dividendYield"))
        if not div_yield or div_yield <= 0:
            return 0, "Firma nevyplácí dividendu"

        score = 0
        details = []

        # 1. FCF Payout Ratio (ideálně < 75%)
        market_cap = safe_float(info.get("marketCap"))
        div_total = div_yield * market_cap if (div_yield and market_cap) else None
        if fcf and div_total:
            fcf_payout = div_total / abs(fcf)
            if fcf_payout < 0.50:
                score += 2
                details.append("✅ FCF payout < 50%")
            elif fcf_payout < 0.75:
                score += 1
                details.append("👍 FCF payout < 75%")
            else:
                details.append(f"⚠️ FCF payout {fcf_payout*100:.0f}% – vysoké")
        else:
            # Fallback: earnings payout ratio
            payout = safe_float(info.get("payoutRatio"))
            if payout is not None:
                if payout < 0.50:
                    score += 2
                    details.append("✅ Payout ratio < 50%")
                elif payout < 0.75:
                    score += 1
                    details.append("👍 Payout ratio < 75%")
                else:
                    details.append(f"⚠️ Payout ratio {payout*100:.0f}%")

        # 2. D/E (nízký dluh = bezpečnější dividenda)
        de = safe_float(info.get("debtToEquity"))
        if de is not None:
            de_normalized = de / 100.0 if de > 10 else de
            if de_normalized < 0.5:
                score += 1
                details.append("✅ Nízká zadluženost (D/E < 0.5)")
            elif de_normalized < 1.5:
                details.append("👍 Střední zadluženost")
            else:
                details.append("⚠️ Vysoká zadluženost – riziko pro dividendu")

        # 3. Provozní cash flow pozitivní
        ocf = safe_float(info.get("operatingCashflow"))
        if ocf and ocf > 0:
            score += 1
            details.append("✅ Kladné provozní cash flow")
        elif ocf is not None:
            details.append("🚨 Záporné provozní cash flow!")

        # 4. Dividend growth (5-year)
        div_growth = safe_float(info.get("fiveYearAvgDividendYield"))
        if div_growth and div_growth > 0:
            details.append("✅ Dividenda vyplácena min. 5 let")

        labels = {5: "Velmi bezpečná 🟢", 4: "Bezpečná 🟢", 3: "Přiměřená 🟡", 2: "Opatrnost ⚠️", 1: "Riziková 🔴", 0: "Velmi riziková 🚨"}
        label = labels.get(score, f"Skóre {score}/5")
        return score, label + "\n" + "\n".join(details)
    except Exception:
        return 0, "Data nedostupná"


def calculate_insider_ownership(info: Dict[str, Any]) -> Optional[float]:
    """Insider ownership jako % z celkového float."""
    try:
        held = safe_float(info.get("heldPercentInsiders"))
        return held  # already 0-1 decimal in yfinance
    except Exception:
        return None


def calculate_historical_pe_ratio(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Porovnání aktuálního P/E s 5letým historickým průměrem.
    
    Metoda: vypočítáme P/E z čtvrtletních dat (EPS TTM + historická cena).
    Vrací: (current_pe, hist_avg_pe, percentile_rank)
    - percentile_rank = 0–100, kde je aktuální P/E v historii (100 = historické maximum)
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        current_pe = safe_float(info.get("trailingPE"))

        # Načíst historické ceny a EPS
        hist = t.history(period="5y", interval="1mo", auto_adjust=True)
        if hist.empty or len(hist) < 12:
            return current_pe, None, None

        # Načíst čtvrtletní EPS
        try:
            earnings = t.quarterly_earnings
            if earnings is None or earnings.empty:
                return current_pe, None, None
        except Exception:
            return current_pe, None, None

        # Spočítat TTM EPS pro každý měsíc
        if "EPS" not in earnings.columns and "Reported EPS" not in earnings.columns:
            return current_pe, None, None

        eps_col = "EPS" if "EPS" in earnings.columns else "Reported EPS"
        earnings_sorted = earnings.sort_index(ascending=True)
        eps_ttm_series = earnings_sorted[eps_col].rolling(4).sum()

        if eps_ttm_series.dropna().empty:
            return current_pe, None, None

        # Párování cen s EPS
        hist_pe_values = []
        for price_date, price_row in hist.iterrows():
            close_price = safe_float(price_row.get("Close"))
            if close_price is None or close_price <= 0:
                continue
            # Najít nejbližší dostupné TTM EPS (ne budoucí)
            past_eps = eps_ttm_series[eps_ttm_series.index <= price_date]
            if past_eps.empty:
                continue
            ttm_eps = safe_float(past_eps.iloc[-1])
            if ttm_eps and ttm_eps > 0:
                pe = close_price / ttm_eps
                if 2 < pe < 200:  # sanity filter
                    hist_pe_values.append(pe)

        if len(hist_pe_values) < 6:
            return current_pe, None, None

        hist_avg = float(np.mean(hist_pe_values))
        if current_pe and hist_pe_values:
            pct_rank = float(np.percentile(
                [100 * (v <= current_pe) for v in hist_pe_values], 50
            ))
            # Alternativně: kde je current_pe v distribuci
            pct_rank = float(sum(v <= current_pe for v in hist_pe_values) / len(hist_pe_values) * 100)
        else:
            pct_rank = None

        return current_pe, round(hist_avg, 1), (round(pct_rank, 0) if pct_rank is not None else None)
    except Exception:
        return None, None, None


def calculate_piotroski_fscore(
    info: Dict[str, Any],
    income: "pd.DataFrame",
    balance: "pd.DataFrame",
    cashflow: "pd.DataFrame"
) -> Tuple[int, Dict[str, int]]:
    """
    Piotroski F-Score (0-9): 9-bodový fundamental quality check.
    Vyšší = lepší kvalita fundamentů.
    OPRAVA: D/E threshold používá normalizovanou hodnotu (< 1.0, ne < 100).
    """
    score = 0
    breakdown: Dict[str, int] = {}

    def _row(df: "pd.DataFrame", candidates: List[str]) -> Optional["pd.Series"]:
        if df is None or df.empty:
            return None
        for c in candidates:
            if c in df.index:
                return df.loc[c]
        return None

    try:
        # --- Profitabilita (4 body) ---
        roa = safe_float(info.get("returnOnAssets"))
        if roa is not None:
            p1 = 1 if roa > 0 else 0
            score += p1
            breakdown["ROA > 0"] = p1

        ocf = safe_float(info.get("operatingCashflow"))
        if ocf is not None:
            p2 = 1 if ocf > 0 else 0
            score += p2
            breakdown["OCF > 0"] = p2

        # Change in ROA (YoY)
        if income is not None and not income.empty and len(income.columns) >= 2:
            ni_row = _row(income, ["Net Income", "Net Income Applicable To Common Shares"])
            ta_row = _row(balance, ["Total Assets"]) if (balance is not None and not balance.empty) else None
            if ni_row is not None and ta_row is not None:
                try:
                    ni_curr = safe_float(ni_row.iloc[0])
                    ni_prev = safe_float(ni_row.iloc[1])
                    ta_curr = safe_float(ta_row.iloc[0])
                    ta_prev = safe_float(ta_row.iloc[1])
                    if all(v is not None and v != 0 for v in [ni_curr, ni_prev, ta_curr, ta_prev]):
                        roa_curr = ni_curr / ta_curr
                        roa_prev = ni_prev / ta_prev
                        p3 = 1 if roa_curr > roa_prev else 0
                        score += p3
                        breakdown["ΔROAᵧ > 0"] = p3
                except Exception:
                    pass

        # Accruals: OCF/Assets > ROA
        ta_val = safe_float(info.get("totalAssets"))
        if ocf is not None and roa is not None and ta_val and ta_val > 0:
            p4 = 1 if (ocf / ta_val) > roa else 0
            score += p4
            breakdown["OCF/Assets > ROA"] = p4

        # --- Leverage & Liquidity (3 body) ---
        de_raw = safe_float(info.get("debtToEquity"))
        if de_raw is not None:
            # Normalizace: Yahoo vrací ×100 formát
            de_normalized = de_raw / 100.0 if de_raw > 10 else de_raw
            # OPRAVA: threshold 1.0 (ne 100)
            p5 = 1 if de_normalized < 1.0 else 0
            score += p5
            breakdown["D/E < 1.0"] = p5

        cr = safe_float(info.get("currentRatio"))
        if cr is not None:
            p6 = 1 if cr > 1 else 0
            score += p6
            breakdown["Current Ratio > 1"] = p6

        # Ředění: buyback check
        shares_curr = safe_float(info.get("sharesOutstanding"))
        buyback = safe_float(info.get("repurchaseOfStock") or info.get("commonStockRepurchased"))
        if shares_curr and buyback is not None:
            p7 = 1 if buyback < 0 else 0
            score += p7
            breakdown["Zpětné odkupy (bez ředění)"] = p7
        elif shares_curr:
            breakdown["Zpětné odkupy (bez ředění)"] = 0

        # --- Efektivita (2 body) ---
        gm_curr = safe_float(info.get("grossMargins"))
        if gm_curr is not None:
            p8 = 1 if gm_curr > 0.30 else 0
            score += p8
            breakdown["Gross Margin > 30%"] = p8

        asset_turnover = safe_div(
            safe_float(info.get("totalRevenue")),
            safe_float(info.get("totalAssets"))
        )
        if asset_turnover is not None:
            p9 = 1 if asset_turnover > 0.5 else 0
            score += p9
            breakdown["Asset Turnover > 0.5"] = p9

    except Exception:
        pass

    return score, breakdown


def calculate_altman_zscore(
    info: Dict[str, Any],
    income: Optional["pd.DataFrame"] = None,
    balance: Optional["pd.DataFrame"] = None,
    market_cap: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """
    Altman Z-Score: bankruptcy risk indicator.
    Classic (public manufacturing) model.
    """
    try:
        sector = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        if "financial" in sector or any(k in industry for k in ["bank", "insurance", "capital markets"]):
            return None, "N/A pro finanční sektor"

        def _df_value(df: Optional["pd.DataFrame"], candidates: List[str]) -> Optional[float]:
            if df is None or getattr(df, "empty", True):
                return None
            for c in candidates:
                if c in df.index:
                    try:
                        return safe_float(df.loc[c].iloc[0])
                    except Exception:
                        continue
            return None

        missing = []

        total_assets = safe_float(info.get("totalAssets")) or _df_value(balance, ["Total Assets"])
        if not total_assets or total_assets <= 0:
            return None, "Data nedostupná"

        current_assets = safe_float(info.get("totalCurrentAssets")) or _df_value(
            balance, ["Total Current Assets", "Current Assets"]
        )
        current_liab = safe_float(info.get("totalCurrentLiabilities")) or _df_value(
            balance, ["Total Current Liabilities", "Current Liabilities"]
        )
        if current_assets is None or current_liab is None:
            missing.append("WC")
        working_capital = (current_assets or 0) - (current_liab or 0)

        retained_earnings = (
            safe_float(info.get("retainedEarnings") or info.get("retainedEarningsAccumulatedDeficit"))
            or _df_value(balance, ["Retained Earnings", "Retained Earnings (Accumulated Deficit)"])
            or 0
        )
        if retained_earnings == 0:
            missing.append("RE")

        ebit = safe_float(info.get("ebit")) or _df_value(income, ["Ebit", "EBIT", "Operating Income"]) or 0
        if ebit == 0:
            ebit = safe_float(info.get("ebitda")) or 0
        if ebit == 0:
            missing.append("EBIT")

        revenue = safe_float(info.get("totalRevenue")) or _df_value(income, ["Total Revenue", "Operating Revenue"]) or 0
        if revenue == 0:
            missing.append("Sales")

        if market_cap is None:
            market_cap = safe_float(info.get("marketCap"))
        if market_cap is None or market_cap == 0:
            price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            shares = safe_float(info.get("sharesOutstanding"))
            if price and shares:
                market_cap = price * shares
        market_cap = market_cap or 0
        if market_cap == 0:
            missing.append("MVE")

        total_liabilities = safe_float(info.get("totalLiabilities")) or _df_value(
            balance,
            ["Total Liab", "Total Liabilities Net Minority Interest", "Total Liabilities"],
        )
        if total_liabilities is None or total_liabilities <= 0:
            total_equity = _df_value(
                balance,
                ["Total Stockholder Equity", "Stockholders Equity",
                 "Total Equity Gross Minority Interest", "Total Equity"],
            )
            if total_equity is not None:
                total_liabilities = total_assets - total_equity

        if total_liabilities is None or total_liabilities <= 0:
            return None, "Data nedostupná (liabilities)"

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities
        x5 = revenue / total_assets

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        if z > 2.99:
            zone = "✅ Bezpečná zóna"
        elif z > 1.81:
            zone = "⚠️ Šedá zóna"
        else:
            zone = "🚨 Riziko bankrotu"

        if missing:
            zone = f"{zone} (odhad: chybí {', '.join(sorted(set(missing)))})"

        return round(float(z), 2), zone

    except Exception:
        return None, "Chyba výpočtu"


def calculate_earnings_quality(info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Earnings Quality: CFO / Net Income ratio."""
    try:
        cfo = safe_float(info.get("operatingCashflow"))
        net_income = safe_float(info.get("netIncomeToCommon"))
        if cfo is None or net_income is None or net_income == 0:
            return None, "Data nedostupná"
        ratio = cfo / abs(net_income)
        if ratio >= 1.1:
            label = "✅ Výborná (CFO > Net Income)"
        elif ratio >= 0.8:
            label = "👍 Dobrá"
        elif ratio >= 0.5:
            label = "⚠️ Průměrná - prověř accruals"
        else:
            label = "🚨 Slabá - možné manipulace"
        return round(ratio, 2), label
    except Exception:
        return None, "Chyba výpočtu"


def detect_market_regime(price_history: "pd.DataFrame") -> str:
    """Detekce režimu na základě volatility a trendu za 6 měsíců."""
    if price_history is None or price_history.empty or len(price_history) < 20:
        return "Stable / Neutral"
    returns = price_history["Close"].pct_change().dropna()
    vol = returns.std() * math.sqrt(252)
    avg_ret = returns.mean() * 252
    if vol > 0.28 and avg_ret < -0.10:
        return "High Volatility / Bear"
    if vol < 0.18 and avg_ret > 0.05:
        return "Low Volatility / Bull"
    return "Stable / Transition"


def _detect_value_trap_impl(
    info: Dict[str, Any], metrics: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Detekce potenciální 'pasti na hodnotu'.
    OPRAVA: D/E threshold používá normalizovanou hodnotu (> 2.0 ne > 200).
    """
    pe = metrics.get("pe").value if metrics.get("pe") else None
    revenue_growth = metrics.get("revenue_growth").value if metrics.get("revenue_growth") else None
    debt_to_equity = metrics.get("debt_to_equity").value if metrics.get("debt_to_equity") else None
    eps = safe_float(info.get("trailingEps"))

    is_trap = False
    warnings_list = []

    if pe and pe < 10:
        if revenue_growth is not None and revenue_growth < -0.05:
            is_trap = True
            warnings_list.append("Klesající tržby (YoY)")
        # OPRAVA: normalized D/E (0.50 = low, 2.0 = high) – ne Yahoo raw (50, 200)
        if debt_to_equity is not None and debt_to_equity > 2.0:
            is_trap = True
            warnings_list.append("Vysoká zadluženost (D/E > 2)")
        if eps is not None and eps <= 0:
            is_trap = True
            warnings_list.append("Negativní/nulové EPS")

    if is_trap:
        msg = (
            f"⚠️ **Potenciální Value Trap**: {', '.join(warnings_list)}. "
            "Nízká valuace může být oprávněná kvůli úpadku byznysu."
        )
        return True, msg
    return False, ""


def detect_value_trap(
    info: Dict[str, Any], metrics: Dict[str, Any]
) -> Tuple[bool, str]:
    return _detect_value_trap_impl(info, metrics)


# ============================================================================
# TECHNICKÁ ANALÝZA
# ============================================================================

def calculate_rsi(price_series: "pd.Series", period: int = 14) -> Optional[float]:
    """Relative Strength Index (RSI)."""
    try:
        if len(price_series) < period + 1:
            return None
        delta = price_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def calculate_macd(
    price_series: "pd.Series",
) -> Tuple[Optional[float], Optional[float], str]:
    """MACD (12/26/9). Returns (macd_line, signal_line, trend_label)."""
    try:
        if len(price_series) < 35:
            return None, None, "Nedostatek dat"
        ema12 = price_series.ewm(span=12, adjust=False).mean()
        ema26 = price_series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        m = float(macd_line.iloc[-1])
        s = float(signal_line.iloc[-1])
        if pd.isna(m) or pd.isna(s):
            return None, None, "—"
        trend = "📈 Bullish" if m > s else "📉 Bearish"
        return m, s, trend
    except Exception:
        return None, None, "—"


def calculate_technical_signals(price_history: "pd.DataFrame") -> Dict[str, Any]:
    """Kompletní sada technických indikátorů."""
    signals: Dict[str, Any] = {}
    try:
        if price_history is None or price_history.empty or "Close" not in price_history.columns:
            return signals
        close = price_history["Close"].dropna()
        if len(close) < 20:
            return signals

        signals["rsi"] = calculate_rsi(close)
        signals["macd"], signals["macd_signal"], signals["macd_label"] = calculate_macd(close)

        # Moving averages
        if len(close) >= 50:
            signals["ma50"] = float(close.rolling(50).mean().iloc[-1])
        if len(close) >= 200:
            signals["ma200"] = float(close.rolling(200).mean().iloc[-1])

        # Bollinger Bands (20d)
        if len(close) >= 20:
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            signals["bb_upper"] = float((ma20 + 2 * std20).iloc[-1])
            signals["bb_lower"] = float((ma20 - 2 * std20).iloc[-1])
            signals["bb_mid"] = float(ma20.iloc[-1])

        # 52W High/Low
        if len(close) >= 252:
            signals["high_52w"] = float(close.rolling(252).max().iloc[-1])
            signals["low_52w"] = float(close.rolling(252).min().iloc[-1])
        else:
            signals["high_52w"] = float(close.max())
            signals["low_52w"] = float(close.min())

        # Price from MA200
        if "ma200" in signals:
            signals["pct_from_ma200"] = (float(close.iloc[-1]) / signals["ma200"]) - 1

        # Volume trend (20d vs 60d)
        if "Volume" in price_history.columns:
            vol = price_history["Volume"].dropna()
            if len(vol) >= 60:
                vol20 = float(vol.iloc[-20:].mean())
                vol60 = float(vol.iloc[-60:].mean())
                if vol60 > 0:
                    signals["vol_trend"] = (vol20 / vol60) - 1

    except Exception:
        pass
    return signals


# ============================================================================
# DCF A OCEŇOVÁNÍ
# ============================================================================

def calculate_dcf_fair_value(
    fcf: float,
    growth_rate: float = 0.10,
    terminal_growth: float = 0.03,
    wacc: float = 0.10,
    years: int = 5,
    shares_outstanding: Optional[float] = None,
    total_cash: float = 0.0,
    total_debt: float = 0.0,
) -> Optional[float]:
    """
    DCF výpočet s Gordon Growth Terminal Value.
    OPRAVENO: Nyní přičítá cash a odečítá dluh (jako hlavní DCF výpočet).
    Parametry total_cash a total_debt jsou volitelné pro zpětnou kompatibilitu.
    """
    if fcf <= 0 or shares_outstanding is None or shares_outstanding <= 0:
        return None
    try:
        pv_sum = 0.0
        current_fcf = fcf
        for year in range(1, years + 1):
            current_fcf *= (1 + growth_rate)
            pv_sum += current_fcf / ((1 + wacc) ** year)
        terminal_fcf = current_fcf * (1 + terminal_growth)
        if wacc <= terminal_growth:
            return None
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** years)
        enterprise_value = pv_sum + pv_terminal
        # Přidáme cash, odečteme dluh (Bridge to Equity)
        equity_value = enterprise_value + total_cash - total_debt
        if equity_value <= 0:
            return None
        return equity_value / shares_outstanding
    except Exception:
        return None


def monte_carlo_dcf(
    fcf: float,
    growth_rate: float,
    terminal_growth: float,
    wacc: float,
    years: int,
    shares_outstanding: float,
    total_cash: float = 0.0,
    total_debt: float = 0.0,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """Monte Carlo simulace DCF (1 000 scénářů)."""
    try:
        results = []
        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            sim_growth = rng.normal(growth_rate, growth_rate * 0.3)
            sim_wacc = rng.normal(wacc, wacc * 0.15)
            sim_terminal = rng.normal(terminal_growth, 0.005)
            sim_wacc = max(0.05, min(0.25, sim_wacc))
            sim_terminal = max(0.0, min(0.05, sim_terminal))
            if sim_wacc <= sim_terminal:
                continue
            fv = calculate_dcf_fair_value(
                fcf, sim_growth, sim_terminal, sim_wacc, years, shares_outstanding,
                total_cash, total_debt
            )
            if fv and fv > 0:
                results.append(fv)
        if not results:
            return {}
        arr = np.array(results)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(np.std(arr)),
            "n": len(results),
        }
    except Exception:
        return {}


def reverse_dcf_implied_growth(
    current_price: float,
    fcf: float,
    terminal_growth: float = 0.03,
    wacc: float = 0.10,
    years: int = 5,
    shares_outstanding: Optional[float] = None,
    total_cash: float = 0.0,
    total_debt: float = 0.0,
) -> Optional[float]:
    """Calculate implied growth rate from current price (binary search)."""
    if fcf <= 0 or shares_outstanding is None or shares_outstanding <= 0:
        return None
    try:
        def dcf_at_growth(g: float) -> float:
            fv = calculate_dcf_fair_value(
                fcf, g, terminal_growth, wacc, years, shares_outstanding,
                total_cash, total_debt
            )
            return fv if fv else 0.0

        low, high = -0.5, 1.0
        for _ in range(50):
            mid = (low + high) / 2.0
            fv = dcf_at_growth(mid)
            if abs(fv - current_price) < 0.01:
                return mid
            if fv < current_price:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0
    except Exception:
        return None


# ============================================================================
# SCORECARD
# ============================================================================

def calculate_metric_score(metric: Any) -> float:
    """Calculate 0-10 score for a single metric."""
    if metric.value is None:
        return 3.0
    val = metric.value
    if metric.target_below is not None:
        if val <= metric.target_below * 0.7:
            return 10.0
        elif val <= metric.target_below:
            return 8.0
        elif val <= metric.target_below * 1.5:
            return 5.0
        else:
            return 2.0
    if metric.target_above is not None:
        if val >= metric.target_above * 1.5:
            return 10.0
        elif val >= metric.target_above:
            return 8.0
        elif val >= metric.target_above * 0.5:
            return 5.0
        else:
            return 2.0
    return 5.0


def build_scorecard_advanced(
    metrics: Dict[str, Any], info: Dict[str, Any]
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Build advanced scorecard (0-100) with category breakdown.
    Returns: (total_score, category_scores, individual_metric_scores)
    """
    categories = {
        "Valuace": ["pe", "pb", "ps", "peg", "ev_ebitda"],
        "Kvalita": ["roe", "roa", "operating_margin", "profit_margin", "gross_margin"],
        "Růst": ["revenue_growth", "earnings_growth"],
        "Fin. zdraví": ["current_ratio", "quick_ratio", "debt_to_equity", "fcf_yield"],
    }
    category_scores: Dict[str, float] = {}
    individual_scores: Dict[str, float] = {}

    for cat_name, metric_keys in categories.items():
        cat_scores = []
        for key in metric_keys:
            metric = metrics.get(key)
            if metric and metric.weight > 0:
                score = calculate_metric_score(metric)
                individual_scores[metric.name] = score
                cat_scores.append((score, metric.weight))
        if cat_scores:
            weighted_sum = sum(s * w for s, w in cat_scores)
            total_weight = sum(w for _, w in cat_scores)
            category_scores[cat_name] = (weighted_sum / total_weight) * 10
        else:
            category_scores[cat_name] = 50.0

    total_score = sum(category_scores.values()) / len(category_scores)
    return total_score, category_scores, individual_scores


def build_radar_data(
    category_scores: Dict[str, float],
    piotroski_score: int,
    insider_signal: float,
    tech_signals: Dict[str, Any],
) -> Dict[str, float]:
    """
    Připraví data pro Spider/Radar Chart (6 os, hodnoty 0–100).
    Osy: Valuace, Kvalita, Růst, Fin. zdraví, Insider, Technická
    """
    # Technická os: kombinace RSI neutrálnosti, MACD, MA200 pozice
    tech_score = 50.0
    try:
        rsi = tech_signals.get("rsi")
        macd_label = tech_signals.get("macd_label", "")
        pct_ma200 = tech_signals.get("pct_from_ma200")
        components = []
        if rsi is not None:
            # RSI 40-60 = neutrální (50 bodů), 30 = přeprodáno (80), 70 = překoupeno (20)
            if rsi < 30:
                components.append(75)
            elif rsi < 45:
                components.append(60)
            elif rsi < 55:
                components.append(50)
            elif rsi < 70:
                components.append(40)
            else:
                components.append(25)
        if "Bullish" in macd_label:
            components.append(65)
        elif "Bearish" in macd_label:
            components.append(35)
        if pct_ma200 is not None:
            if pct_ma200 > 0.10:
                components.append(60)
            elif pct_ma200 > 0:
                components.append(55)
            else:
                components.append(40)
        if components:
            tech_score = float(np.mean(components))
    except Exception:
        pass

    # Insider os: převod z -100..+100 na 0..100
    insider_norm = max(0.0, min(100.0, (insider_signal + 100) / 2.0))

    # Piotroski os: 0-9 → 0-100
    piotroski_norm = (piotroski_score / 9.0) * 100.0

    return {
        "Valuace": round(category_scores.get("Valuace", 50), 1),
        "Kvalita": round(category_scores.get("Kvalita", 50), 1),
        "Růst": round(category_scores.get("Růst", 50), 1),
        "Fin. zdraví": round(category_scores.get("Fin. zdraví", 50), 1),
        "Piotroski": round(piotroski_norm, 1),
        "Insider": round(insider_norm, 1),
        "Technická": round(tech_score, 1),
    }


# ============================================================================
# INVESTIČNÍ SIMULÁTOR
# ============================================================================

def simulate_investment(
    ticker: str, amount_czk: float, years_back: int = 5
) -> Optional[Dict[str, Any]]:
    """'Co kdybych investoval X Kč?' – porovnání s SPY (S&P 500 ETF)."""
    try:
        import yfinance as yf
        period = f"{years_back}y"
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty or len(hist) < 10:
            return None
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period=period, auto_adjust=True)
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        stock_return = (end_price / start_price) - 1
        final_value = amount_czk * (1 + stock_return)
        spy_return = None
        if not spy_hist.empty:
            spy_return = (float(spy_hist["Close"].iloc[-1]) / float(spy_hist["Close"].iloc[0])) - 1
        return {
            "initial": amount_czk,
            "final": final_value,
            "stock_return": stock_return,
            "spy_return": spy_return,
            "years": years_back,
            "start_date": hist.index[0].strftime("%d.%m.%Y"),
            "end_date": hist.index[-1].strftime("%d.%m.%Y"),
        }
    except Exception:
        return None


# ============================================================================
# VERDICT
# ============================================================================

def get_advanced_verdict(
    scorecard: float,
    mos_dcf: Optional[float],
    mos_analyst: Optional[float],
    insider_signal: float,
    implied_growth: Optional[float],
) -> Tuple[str, str, List[str]]:
    """Advanced verdict with multiple signals."""
    warnings: List[str] = []

    if scorecard >= 85:
        base, color = "STRONG BUY", "#00ff88"
    elif scorecard >= 60:
        base, color = "BUY", "#88ff00"
    elif scorecard >= 45:
        base, color = "HOLD", "#ffaa00"
    elif scorecard >= 30:
        base, color = "CAUTION", "#ff8800"
    else:
        base, color = "AVOID", "#ff4444"

    if mos_dcf is not None:
        if mos_dcf >= 0.20 and base in ["HOLD", "CAUTION"]:
            base, color = "BUY", "#88ff00"
        elif mos_dcf < -0.15 and base in ["STRONG BUY", "BUY"]:
            base, color = "HOLD", "#ffaa00"
            warnings.append("⚠️ DCF model ukazuje přeceněnost (-15% MOS)")

    if mos_analyst is not None and mos_dcf is not None:
        if mos_analyst > 0.15 and mos_dcf < -0.10:
            warnings.append("🚨 MISMATCH: Analytici vidí upside +15%, ale DCF ukazuje overvalued -10%!")
            warnings.append("   → Trh možná implikuje vyšší růst než je ve tvém DCF modelu")

    if insider_signal > 50:
        warnings.append(f"✅ Silný insider buying (+{insider_signal:.0f}) podporuje BUY tezi")
    elif insider_signal < -30:
        warnings.append(f"⚠️ Negativní insider selling ({insider_signal:.0f})")

    if implied_growth is not None:
        if implied_growth > 0.25:
            warnings.append(f"⚠️ Trh implikuje agresivní růst FCF ({implied_growth*100:.0f}%) – vysoká očekávání!")
        elif implied_growth < 0:
            warnings.append(f"📉 Trh implikuje pokles FCF ({implied_growth*100:.0f}%) – možná příležitost")

    return base, color, warnings


# ============================================================================
# ODHAD SMART DCF PARAMETRŮ
# ============================================================================

def estimate_smart_params(info: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Konzervativní odhad DCF parametrů.
    OPRAVA: D/E check používá normalizovanou hodnotu (< 0.5, ne < 50).
    """
    market_cap = safe_float(info.get("marketCap")) or 0.0
    sector = str(info.get("sector") or "").strip()

    is_mega_cap = market_cap > 200e9
    is_large_cap = market_cap > 50e9

    # WACC z beta
    beta = safe_float(info.get("beta"))
    if beta is None or beta <= 0:
        base_wacc = 0.10
    else:
        base_wacc = 0.042 + (beta * 0.05)
    wacc = max(0.09, min(0.15, base_wacc))
    if market_cap < 10e9 and market_cap > 0:
        wacc += 0.015

    # Růst – vážený průměr (70% tržby, 30% EPS)
    rev_g: Optional[float] = None
    earn_g: Optional[float] = None
    try:
        m = metrics.get("revenue_growth")
        if m and m.value is not None:
            rev_g = float(m.value)
    except Exception:
        pass
    try:
        m = metrics.get("earnings_growth")
        if m and m.value is not None:
            earn_g = float(m.value)
    except Exception:
        pass

    if rev_g is not None and earn_g is not None:
        raw_growth = 0.7 * rev_g + 0.3 * earn_g
    elif rev_g is not None:
        raw_growth = rev_g
    elif earn_g is not None:
        raw_growth = earn_g
    else:
        raw_growth = 0.10

    growth_cap = 0.08 if is_mega_cap else (0.12 if is_large_cap else 0.20)
    growth = max(0.03, min(growth_cap, raw_growth))

    # Exit multiple (sektorový základ + quality premium)
    sector_l = sector.lower()
    if "technology" in sector_l:
        base_multiple = 20.0
    elif "communication" in sector_l:
        base_multiple = 18.0
    elif "consumer cyclical" in sector_l:
        base_multiple = 20.0
    elif "financial" in sector_l or "energy" in sector_l:
        base_multiple = 12.0
    elif "healthcare" in sector_l:
        base_multiple = 18.0
    else:
        base_multiple = 15.0

    quality_score = 0
    try:
        roe = safe_float(metrics.get("roe").value) if metrics.get("roe") else 0
        if roe and roe > 0.15:
            quality_score += 2
        elif roe and roe > 0.10:
            quality_score += 1
    except Exception:
        pass
    try:
        pm = safe_float(metrics.get("profit_margin").value) if metrics.get("profit_margin") else 0
        if pm and pm > 0.20:
            quality_score += 2
        elif pm and pm > 0.10:
            quality_score += 1
    except Exception:
        pass
    try:
        roa = safe_float(metrics.get("roa").value) if metrics.get("roa") else 0
        if roa and roa > 0.15:
            quality_score += 2
        elif roa and roa > 0.10:
            quality_score += 1
    except Exception:
        pass
    try:
        # OPRAVA: normalizovaná D/E (< 0.5, ne < 50)
        debt_eq = safe_float(metrics.get("debt_to_equity").value) if metrics.get("debt_to_equity") else None
        if debt_eq is not None and debt_eq < 0.5:
            quality_score += 1
    except Exception:
        pass

    exit_multiple = min(25.0, base_multiple + quality_score)

    return {
        "wacc": float(wacc),
        "growth": float(growth),
        "exit_multiple": float(exit_multiple),
        "is_mega_cap": bool(is_mega_cap),
        "market_cap": float(market_cap),
        "sector": sector,
    }


# ============================================================================
# CRYPTO DETEKCE
# ============================================================================

def is_crypto(info: Dict[str, Any]) -> bool:
    """Detekuje, zda je aktivum kryptoměna."""
    qt = (info.get("quoteType") or "").upper()
    return qt in ("CRYPTOCURRENCY", "CRYPTO")


def get_crypto_display_metrics(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrahuje metriky relevantní pro kryptoměny.
    Vrátí dict s hodnotami pro zobrazení v Overview.
    """
    price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
    market_cap = safe_float(info.get("marketCap"))
    volume_24h = safe_float(info.get("volume24Hr") or info.get("regularMarketVolume"))
    circulating_supply = safe_float(info.get("circulatingSupply"))
    total_supply = safe_float(info.get("maxSupply") or info.get("totalSupply"))
    change_24h = safe_float(info.get("regularMarketChangePercent"))
    change_52w_high = safe_float(info.get("fiftyTwoWeekHigh"))
    change_52w_low = safe_float(info.get("fiftyTwoWeekLow"))

    return {
        "price": price,
        "market_cap": market_cap,
        "volume_24h": volume_24h,
        "circulating_supply": circulating_supply,
        "total_supply": total_supply,
        "change_24h": change_24h,
        "high_52w": change_52w_high,
        "low_52w": change_52w_low,
        "is_crypto": True,
    }


# ============================================================================
# WATCHLIST TRACKING (snapshot logika)
# ============================================================================

def make_snapshot(
    ticker: str,
    current_price: Optional[float],
    scorecard: float,
    fair_value_dcf: Optional[float],
    mos_dcf: Optional[float],
    verdict: str,
) -> Dict[str, Any]:
    """Vytvoří snapshot metriky pro historické sledování ve watchlistu."""
    return {
        "date": dt.datetime.now().strftime("%Y-%m-%d"),
        "price": current_price,
        "scorecard": round(scorecard, 1),
        "dcf_fair": fair_value_dcf,
        "mos": mos_dcf,
        "verdict": verdict,
    }
