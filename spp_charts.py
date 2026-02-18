"""
Stock Picker Pro – Charts Module
==================================
Interactive Plotly candlestick charts with technical indicator overlays.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Candlestick Chart Builder
# ---------------------------------------------------------------------------

def build_candlestick_chart(
    price_history: pd.DataFrame,
    ticker: str = "",
    show_ma50: bool = True,
    show_ma200: bool = True,
    show_bollinger: bool = True,
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    fair_value_dcf: Optional[float] = None,
    height: int = 800,
) -> go.Figure:
    """
    Build a comprehensive interactive candlestick chart with overlays.

    Parameters
    ----------
    price_history : DataFrame with columns Open, High, Low, Close, Volume
    ticker : str — ticker symbol for title
    show_ma50/ma200 : bool — show moving averages
    show_bollinger : bool — show Bollinger Bands
    show_volume : bool — show volume subplot
    show_rsi : bool — show RSI subplot
    show_macd : bool — show MACD subplot
    fair_value_dcf : Optional[float] — horizontal line for DCF fair value

    Returns
    -------
    plotly Figure
    """
    df = price_history.copy()

    # Ensure we have the right columns (handle various yfinance formats)
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower == "open":
            col_map["Open"] = col
        elif col_lower == "high":
            col_map["High"] = col
        elif col_lower == "low":
            col_map["Low"] = col
        elif col_lower in ("close", "adj close"):
            col_map["Close"] = col
        elif col_lower == "volume":
            col_map["Volume"] = col

    if "Close" not in col_map:
        # Try to find any price-like column
        for col in df.columns:
            if "close" in str(col).lower():
                col_map["Close"] = col
                break

    if "Close" not in col_map:
        # Fallback: create a simple line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df.iloc[:, 0],
            mode="lines", name="Price",
            line=dict(color="#00ff88", width=2),
        ))
        fig.update_layout(
            title=f"📈 {ticker} — Price Chart",
            template="plotly_dark",
            height=400,
        )
        return fig

    # Determine subplot counts
    n_subplots = 1
    subplot_titles = [f"📈 {ticker}"]
    row_heights = [0.5]

    if show_volume:
        n_subplots += 1
        subplot_titles.append("Volume")
        row_heights.append(0.12)

    if show_rsi:
        n_subplots += 1
        subplot_titles.append("RSI")
        row_heights.append(0.15)

    if show_macd:
        n_subplots += 1
        subplot_titles.append("MACD")
        row_heights.append(0.18)

    # Normalise heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    # ---- Main candlestick ----
    has_ohlc = all(k in col_map for k in ("Open", "High", "Low", "Close"))
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df[col_map["Open"]],
                high=df[col_map["High"]],
                low=df[col_map["Low"]],
                close=df[col_map["Close"]],
                name="Price",
                increasing_line_color="#00ff88",
                decreasing_line_color="#ff4444",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col_map["Close"]],
                mode="lines", name="Close",
                line=dict(color="#00ff88", width=2),
            ),
            row=1, col=1,
        )

    close = df[col_map["Close"]]

    # ---- Moving Averages ----
    if show_ma50 and len(close) >= 50:
        ma50 = close.rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ma50,
                mode="lines", name="MA50",
                line=dict(color="#ffaa00", width=1.5, dash="dot"),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    if show_ma200 and len(close) >= 200:
        ma200 = close.rolling(200).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ma200,
                mode="lines", name="MA200",
                line=dict(color="#ff6600", width=1.5, dash="dash"),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    # ---- Bollinger Bands ----
    if show_bollinger and len(close) >= 20:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb_upper,
                mode="lines", name="BB Upper",
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb_lower,
                mode="lines", name="BB Lower",
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.08)",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # ---- DCF Fair Value Line ----
    if fair_value_dcf is not None and fair_value_dcf > 0:
        fig.add_hline(
            y=fair_value_dcf,
            line_dash="dash",
            line_color="#00ccff",
            line_width=2,
            annotation_text=f"DCF Fair Value: ${fair_value_dcf:.2f}",
            annotation_position="top left",
            annotation_font_color="#00ccff",
            row=1, col=1,
        )

    # ---- Volume ----
    current_row = 2
    if show_volume and "Volume" in col_map:
        vol = df[col_map["Volume"]]
        colors = ["#00ff88" if c >= o else "#ff4444"
                  for c, o in zip(df[col_map["Close"]], df[col_map.get("Open", "Close")])]
        fig.add_trace(
            go.Bar(
                x=df.index, y=vol,
                name="Volume",
                marker_color=colors,
                opacity=0.6,
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        current_row += 1

    # ---- RSI ----
    if show_rsi and len(close) >= 14:
        rsi = _compute_rsi(close, 14)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=rsi,
                mode="lines", name="RSI(14)",
                line=dict(color="#aa88ff", width=1.5),
            ),
            row=current_row, col=1,
        )
        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,68,68,0.5)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,136,0.5)", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # ---- MACD ----
    if show_macd and len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_line,
                mode="lines", name="MACD",
                line=dict(color="#00ccff", width=1.5),
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=signal_line,
                mode="lines", name="Signal",
                line=dict(color="#ff8800", width=1.5),
            ),
            row=current_row, col=1,
        )
        colors = ["#00ff88" if v >= 0 else "#ff4444" for v in histogram]
        fig.add_trace(
            go.Bar(
                x=df.index, y=histogram,
                name="Histogram",
                marker_color=colors,
                opacity=0.6,
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        current_row += 1

    # ---- Layout ----
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=50, r=20, t=60, b=30),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    
    # Subtle grid lines
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)")

    # Remove weekend gaps
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    return fig


# ---------------------------------------------------------------------------
# Helper – RSI Calculation
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Use EMA smoothing after initial SMA
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Earnings Surprise Chart
# ---------------------------------------------------------------------------

def build_earnings_surprise_chart(
    ticker: str,
    earnings_data: Optional[pd.DataFrame] = None,
) -> Optional[go.Figure]:
    """
    Build a bar chart showing actual vs expected EPS (earnings surprise).

    Parameters
    ----------
    ticker : str
    earnings_data : DataFrame with columns like 'epsActual', 'epsEstimate', 'quarter'

    Returns
    -------
    plotly Figure or None
    """
    try:
        import yfinance as yf

        if earnings_data is None or earnings_data.empty:
            t = yf.Ticker(ticker)
            earnings_data = getattr(t, "earnings_history", None)
            if earnings_data is None or earnings_data.empty:
                # Try quarterly earnings
                qe = getattr(t, "quarterly_earnings", None)
                if qe is not None and not qe.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=qe.index.astype(str),
                        y=qe.get("Revenue", qe.iloc[:, 0]) if "Revenue" in qe.columns else qe.iloc[:, 0],
                        name="Revenue/Earnings",
                        marker_color="#00ccff",
                    ))
                    fig.update_layout(
                        title=f"📊 {ticker} — Kvartální výsledky",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=350,
                    )
                    return fig
                return None

        if earnings_data.empty:
            return None

        # Normalize column names
        df = earnings_data.copy()
        cols_lower = {str(c).lower(): c for c in df.columns}

        actual_col = None
        estimate_col = None
        for key, col in cols_lower.items():
            if "actual" in key:
                actual_col = col
            elif "estimate" in key or "expected" in key:
                estimate_col = col

        if actual_col is None:
            return None

        fig = go.Figure()

        # Actual EPS
        fig.add_trace(go.Bar(
            x=df.index.astype(str) if not isinstance(df.index[0], str) else df.index,
            y=df[actual_col],
            name="Actual EPS",
            marker_color="#00ff88",
        ))

        # Estimated EPS
        if estimate_col:
            fig.add_trace(go.Bar(
                x=df.index.astype(str) if not isinstance(df.index[0], str) else df.index,
                y=df[estimate_col],
                name="Expected EPS",
                marker_color="rgba(255, 170, 0, 0.6)",
            ))

        fig.update_layout(
            title=f"📊 {ticker} — Earnings Surprise (Actual vs Expected EPS)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            height=350,
            font=dict(color="white"),
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        return fig

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Yield Curve Visualization
# ---------------------------------------------------------------------------

def build_yield_curve_chart(yield_data: Dict[str, Optional[float]]) -> Optional[go.Figure]:
    """Build a yield curve chart from FRED data."""
    labels = []
    values = []
    for label in ["2Y", "5Y", "10Y", "30Y"]:
        val = yield_data.get(label)
        if val is not None:
            labels.append(label)
            values.append(val)

    if len(values) < 2:
        return None

    # Detect inversion
    is_inverted = len(values) >= 2 and values[0] > values[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        mode="lines+markers",
        name="Yield Curve",
        line=dict(
            color="#ff4444" if is_inverted else "#00ff88",
            width=3,
        ),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title=f"📉 US Treasury Yield Curve {'(⚠️ INVERTED!)' if is_inverted else ''}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        yaxis_title="Yield (%)",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig
