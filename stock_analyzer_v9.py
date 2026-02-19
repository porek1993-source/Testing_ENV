"""
Stock Picker Pro v9.0
================================
Pokročilá česká aplikace pro kvantitativní analýzu akcií.
Funkce: DCF (Monte Carlo), Insider signal, Scorecard, Piotroski, Altman Z-Score,
Graham Number, technická analýza (RSI/MACD/BB), peer comparison, AI analyst,
Radar Chart, Net Debt/EBITDA, Rule of 40, FCF Margin, Dividend Safety Score,
Historické P/E, podpora kryptoměn (BTC-USD), Watchlist snapshoty.

Jazyk: pouze čeština
Moduly: spp_constants.py, spp_analytics.py
"""

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'google\.generativeai\..*')
import requests
import re
import json
import html
import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

# --- New v10.0 modules ---
import spp_database as db
import spp_macro
import spp_news
import spp_charts
import spp_screener
from spp_constants import METRIC_TOOLTIPS


# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Stock Picker Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DESIGN SYSTEM & CSS OVERHAUL (v10.0 Modern UI)
# =============================================================================

def load_custom_css():
    """Premium Design System v10.0 – Modern dark UI with glassmorphism, animations, and responsive layout."""
    st.markdown("""
        <style>
        /* ================================================================
           1. IMPORTS & DESIGN TOKENS
           ================================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

        :root {
            /* Surface colors */
            --bg-primary: #06080d;
            --bg-secondary: #0c0f17;
            --bg-elevated: #111520;
            --card-bg: rgba(255, 255, 255, 0.035);
            --card-bg-hover: rgba(255, 255, 255, 0.06);
            --card-border: rgba(255, 255, 255, 0.08);
            --card-border-hover: rgba(255, 255, 255, 0.18);

            /* Text colors */
            --text-primary: #f0f2f5;
            --text-secondary: #8b92a5;
            --text-muted: #555d70;

            /* Accent palette */
            --accent-green: #00e68a;
            --accent-green-dim: rgba(0, 230, 138, 0.12);
            --accent-red: #ff5a5a;
            --accent-red-dim: rgba(255, 90, 90, 0.12);
            --accent-blue: #3b9eff;
            --accent-blue-dim: rgba(59, 158, 255, 0.12);
            --accent-violet: #8b5cf6;
            --accent-violet-dim: rgba(139, 92, 246, 0.12);
            --accent-amber: #f59e0b;
            --accent-amber-dim: rgba(245, 158, 11, 0.12);

            /* Gradients */
            --gradient-primary: linear-gradient(135deg, #00e68a 0%, #00b4d8 100%);
            --gradient-hero: linear-gradient(135deg, rgba(0,230,138,0.08) 0%, rgba(139,92,246,0.06) 100%);
            --gradient-card: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            --gradient-sidebar: linear-gradient(180deg, #08090e 0%, #0a0c14 100%);

            /* Spacing & Radius */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;

            /* Shadows */
            --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.25), 0 1px 2px rgba(0, 0, 0, 0.2);
            --shadow-elevated: 0 8px 40px rgba(0, 0, 0, 0.35), 0 2px 8px rgba(0, 0, 0, 0.25);
            --shadow-glow-green: 0 0 20px rgba(0, 230, 138, 0.15);
            --shadow-glow-violet: 0 0 20px rgba(139, 92, 246, 0.15);

            /* Typography */
            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* ================================================================
           2. KEYFRAME ANIMATIONS
           ================================================================ */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 4px rgba(0, 230, 138, 0.2); }
            50% { box-shadow: 0 0 16px rgba(0, 230, 138, 0.35); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-12px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes scaleIn {
            from { opacity: 0; transform: scale(0.92); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* ================================================================
           3. GLOBAL RESET & BASE
           ================================================================ */
        .stApp {
            background: var(--bg-primary) !important;
            font-family: var(--font-main) !important;
            color: var(--text-primary);
        }

        h1, h2, h3, h4, h5, h6, p, label, button, input, textarea, div {
            font-family: var(--font-main) !important;
        }
        /* Preserve Streamlit / Material icon fonts */
        span[data-testid] *,
        [data-baseweb] [role="presentation"],
        .material-symbols-outlined,
        .e1nzilvr5,
        [class*="Icon"] {
            font-family: inherit !important;
        }

        h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
        h2 { font-weight: 700 !important; letter-spacing: -0.01em; }
        h3 { font-weight: 600 !important; }

        a { color: var(--accent-blue); transition: color 0.2s ease; }
        a:hover { color: var(--accent-green); }

        /* ================================================================
           4. LAYOUT & CONTAINERS
           ================================================================ */
        .block-container {
            max-width: 96% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 4rem !important;
        }

        div[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* Main content area */
        .main .block-container {
            animation: fadeInUp 0.4s ease-out;
        }

        /* ================================================================
           5. GLASSMORPHISM CARDS
           ================================================================ */
        .glass-card {
            background: var(--gradient-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 20px;
            margin-bottom: 16px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-card);
            animation: fadeInUp 0.5s ease-out both;
        }
        .glass-card:hover {
            border-color: var(--card-border-hover);
            transform: translateY(-3px);
            box-shadow: var(--shadow-elevated);
            background: var(--card-bg-hover);
        }

        /* Stat cards for header metrics */
        .stat-card {
            background: var(--gradient-card);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-card);
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .stat-card:hover {
            border-color: var(--card-border-hover);
            transform: translateY(-2px);
        }
        .stat-card:hover::before { opacity: 1; }

        /* Metric card layout */
        .metric-card {
            background: var(--gradient-card);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: all 0.25s ease;
        }
        .metric-card:hover {
            border-color: var(--card-border-hover);
        }

        /* ================================================================
           6. TYPOGRAPHY COMPONENTS
           ================================================================ */
        .metric-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: clamp(1.4rem, 4vw, 2rem);
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1.2;
        }
        .metric-delta {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 4px;
        }

        .text-green { color: var(--accent-green) !important; }
        .text-red { color: var(--accent-red) !important; }
        .text-blue { color: var(--accent-blue) !important; }
        .text-violet { color: var(--accent-violet) !important; }
        .text-amber { color: var(--accent-amber) !important; }
        .text-muted { color: var(--text-muted) !important; }

        /* Section header with gradient underline */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 24px 0 14px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid transparent;
            border-image: var(--gradient-primary) 1;
            animation: fadeInUp 0.4s ease-out both;
        }

        /* ================================================================
           7. BADGES & PILLS
           ================================================================ */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 12px;
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }
        .badge-green { background: var(--accent-green-dim); color: var(--accent-green); border: 1px solid rgba(0,230,138,0.2); }
        .badge-red { background: var(--accent-red-dim); color: var(--accent-red); border: 1px solid rgba(255,90,90,0.2); }
        .badge-blue { background: var(--accent-blue-dim); color: var(--accent-blue); border: 1px solid rgba(59,158,255,0.2); }
        .badge-violet { background: var(--accent-violet-dim); color: var(--accent-violet); border: 1px solid rgba(139,92,246,0.2); }
        .badge-amber { background: var(--accent-amber-dim); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.2); }

        /* ================================================================
           8. ALERT BOXES
           ================================================================ */
        .warning-box {
            background: var(--accent-red-dim);
            border-left: 3px solid var(--accent-red);
            padding: 14px 18px;
            border-radius: var(--radius-sm);
            color: #ffc9c9;
            margin-bottom: 12px;
            animation: slideInLeft 0.3s ease-out;
        }
        .success-box {
            background: var(--accent-green-dim);
            border-left: 3px solid var(--accent-green);
            padding: 14px 18px;
            border-radius: var(--radius-sm);
            color: #c8ffe0;
            margin-bottom: 12px;
            animation: slideInLeft 0.3s ease-out;
        }
        .info-box {
            background: var(--accent-blue-dim);
            border-left: 3px solid var(--accent-blue);
            padding: 14px 18px;
            border-radius: var(--radius-sm);
            color: #c0e0ff;
            margin-bottom: 12px;
            animation: slideInLeft 0.3s ease-out;
        }

        /* ================================================================
           8b. MATERIAL ICON FIX
           ================================================================ */
        /* Force Material Symbols font for Streamlit icon elements */

        /* Collapse / expand sidebar button and toolbar icons */
        [data-testid="stSidebarCollapsedControl"] span,
        [data-testid="collapsedControl"] span,
        button[kind="header"] span,
        [data-testid="stHeader"] span,
        [data-testid="stToolbar"] span,
        .stDeployButton span,
        details summary span[class] {
            font-family: 'Material Symbols Outlined', sans-serif !important;
            font-size: 24px !important;
            -webkit-font-feature-settings: 'liga' 1;
            font-feature-settings: 'liga' 1;
        }

        /* ================================================================
           9. SIDEBAR
           ================================================================ */
        [data-testid="stSidebar"] {
            background: var(--gradient-sidebar) !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
        }
        section[data-testid="stSidebar"] hr {
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
            margin: 1.5rem 0 !important;
            border: none !important;
            height: 1px !important;
        }

        /* ================================================================
           10. TABS
           ================================================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 0;
            background: transparent;
            overflow-x: auto;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: var(--radius-sm) var(--radius-sm) 0 0;
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.85rem;
            padding: 8px 16px;
            transition: all 0.25s ease;
            border-bottom: 2px solid transparent;
            white-space: nowrap;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary);
            background: rgba(255,255,255,0.04);
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 230, 138, 0.06) !important;
            color: var(--accent-green) !important;
            border-bottom: 2px solid var(--accent-green) !important;
            font-weight: 600;
        }

        /* ================================================================
           11. FORM ELEMENTS
           ================================================================ */
        .stTextInput input, .stNumberInput input {
            background-color: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: var(--radius-sm) !important;
            transition: all 0.25s ease !important;
            font-size: 0.9rem !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: var(--radius-sm) !important;
        }

        input:focus, textarea:focus, select:focus {
            border-color: var(--accent-green) !important;
            box-shadow: 0 0 0 2px rgba(0, 230, 138, 0.12) !important;
            transition: all 0.25s ease !important;
        }

        /* ================================================================
           12. BUTTONS
           ================================================================ */
        div.stButton > button {
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            color: var(--text-primary);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: var(--radius-sm);
            font-weight: 600;
            font-size: 0.88rem;
            padding: 0.5rem 1.2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            min-height: 44px;
        }
        div.stButton > button:hover {
            border-color: var(--accent-green);
            color: var(--accent-green);
            background: rgba(0, 230, 138, 0.06);
            transform: translateY(-1px);
            box-shadow: var(--shadow-glow-green);
        }
        div.stButton > button[kind="primary"],
        div.stButton > button:has([data-testid="stBaseButton-primary"]) {
            background: linear-gradient(135deg, #00e68a 0%, #00b4d8 100%);
            color: #06080d;
            border: none;
            font-weight: 700;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(0, 230, 138, 0.3);
            color: #06080d;
        }

        /* ================================================================
           13. DATA TABLES
           ================================================================ */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: var(--radius-sm) !important;
            overflow: hidden;
        }

        /* ================================================================
           14. EXPANDERS
           ================================================================ */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: var(--text-primary) !important;
        }
        details {
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: var(--radius-sm) !important;
            background: rgba(255,255,255,0.02) !important;
        }

        /* ================================================================
           15. METRICS (st.metric override)
           ================================================================ */
        [data-testid="stMetricValue"] {
            font-size: clamp(1.2rem, 3.5vw, 1.8rem) !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            color: var(--text-muted) !important;
        }
        [data-testid="stMetricDelta"] > div {
            font-weight: 600 !important;
        }

        /* ================================================================
           16. TOAST NOTIFICATIONS
           ================================================================ */
        div[data-testid="stToast"] {
            background: rgba(12, 15, 23, 0.96) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-left: 3px solid var(--accent-green) !important;
            color: var(--text-primary) !important;
            border-radius: var(--radius-sm) !important;
            backdrop-filter: blur(20px);
            box-shadow: var(--shadow-elevated);
        }

        /* ================================================================
           17. SCROLLBARS
           ================================================================ */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-green);
        }
        * { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.15) transparent; }

        /* ================================================================
           18. HERO / WELCOME COMPONENTS
           ================================================================ */
        .hero-container {
            background: var(--gradient-hero);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-xl);
            padding: 48px 40px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
            animation: scaleIn 0.5s ease-out;
        }
        .hero-container::after {
            content: '';
            position: absolute;
            top: -50%; right: -50%;
            width: 100%; height: 100%;
            background: radial-gradient(circle, rgba(139,92,246,0.05) 0%, transparent 60%);
            pointer-events: none;
        }
        .hero-title {
            font-size: clamp(2rem, 5vw, 3.2rem);
            font-weight: 900;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            line-height: 1.15;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
            line-height: 1.6;
            max-width: 640px;
        }

        /* Feature cards for welcome screen */
        .feature-card {
            background: var(--gradient-card);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 20px;
            transition: all 0.3s ease;
            cursor: default;
        }
        .feature-card:hover {
            border-color: var(--card-border-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-card);
        }
        .feature-icon {
            font-size: 1.8rem;
            margin-bottom: 10px;
            display: block;
        }
        .feature-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        .feature-desc {
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Ticker button for welcome screen */
        .ticker-btn {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: var(--radius-sm);
            padding: 10px 16px;
            text-align: center;
            transition: all 0.25s ease;
            cursor: pointer;
        }
        .ticker-btn:hover {
            background: rgba(0, 230, 138, 0.08);
            border-color: rgba(0, 230, 138, 0.3);
            transform: translateY(-1px);
        }

        /* ================================================================
           19. SCORE RING
           ================================================================ */
        .score-ring {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 80px; height: 80px;
            border-radius: 50%;
            font-size: 1.6rem;
            font-weight: 900;
            position: relative;
        }
        .score-ring::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            border: 3px solid rgba(255,255,255,0.08);
        }

        /* ================================================================
           20. RESPONSIVE DESIGN
           ================================================================ */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            .glass-card, .stat-card, .metric-card {
                padding: 14px 16px;
            }
            .hero-container {
                padding: 28px 20px;
            }
            .hero-title {
                font-size: 1.8rem;
            }
            section[data-testid="stSidebar"] {
                background: rgba(8, 9, 14, 0.98) !important;
                backdrop-filter: none !important;
            }
            section[data-testid="stSidebar"] * {
                color: var(--text-primary);
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 0.78rem;
                padding: 6px 10px;
            }
        }

        @media (max-width: 480px) {
            .metric-value {
                font-size: 1.3rem;
            }
            .section-header {
                font-size: 1.2rem;
            }
        }

        /* ================================================================
           21. LOADING / SKELETON SHIMMER
           ================================================================ */
        .shimmer {
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.04) 50%, transparent 100%);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
        }

        /* ================================================================
           22. MISC POLISH
           ================================================================ */
        /* Divider override */
        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
            margin: 1.5rem 0 !important;
        }

        /* Smooth transitions on all interactive elements */
        button, input, select, textarea, a, details {
            transition: all 0.25s ease;
        }

        /* Plotly chart background fix */
        .js-plotly-plot .plotly .main-svg {
            border-radius: var(--radius-sm);
        }

        /* ================================================================
           23. LIGHT THEME OVERRIDE
           ================================================================ */
        [data-theme="light"] {
            --bg-primary: #f5f6fa;
            --bg-secondary: #ebedf3;
            --bg-elevated: #ffffff;
            --card-bg: rgba(0, 0, 0, 0.03);
            --card-bg-hover: rgba(0, 0, 0, 0.06);
            --card-border: rgba(0, 0, 0, 0.08);
            --card-border-hover: rgba(0, 0, 0, 0.16);
            --text-primary: #1a1d26;
            --text-secondary: #5a6275;
            --text-muted: #9098ad;
            --shadow-card: 0 2px 12px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
            --shadow-elevated: 0 4px 20px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.08);
            --shadow-glow-green: 0 0 12px rgba(0, 230, 138, 0.10);
            --shadow-glow-violet: 0 0 12px rgba(139, 92, 246, 0.10);
            --gradient-hero: linear-gradient(135deg, rgba(0,230,138,0.06) 0%, rgba(139,92,246,0.04) 100%);
            --gradient-card: linear-gradient(180deg, rgba(0,0,0,0.03) 0%, rgba(0,0,0,0.01) 100%);
            --gradient-sidebar: linear-gradient(180deg, #f0f1f5 0%, #e8eaf0 100%);
        }
        [data-theme="light"] .stApp {
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        [data-theme="light"] section[data-testid="stSidebar"] {
            background: var(--gradient-sidebar) !important;
            border-right: 1px solid var(--card-border) !important;
        }
        [data-theme="light"] .stTabs [role="tab"] {
            color: var(--text-secondary) !important;
        }
        [data-theme="light"] .stTabs [role="tab"][aria-selected="true"] {
            color: var(--text-primary) !important;
        }

        /* ================================================================
           24. MOBILE TAB RESPONSIVENESS
           ================================================================ */
        @media (max-width: 768px) {
            .stTabs [role="tablist"] {
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
                -ms-overflow-style: none;
                flex-wrap: nowrap !important;
                gap: 2px !important;
                padding-bottom: 4px !important;
                mask-image: linear-gradient(to right, transparent 0%, black 3%, black 97%, transparent 100%);
                -webkit-mask-image: linear-gradient(to right, transparent 0%, black 3%, black 97%, transparent 100%);
            }
            .stTabs [role="tablist"]::-webkit-scrollbar { display: none; }
            .stTabs [role="tab"] {
                padding: 6px 10px !important;
                font-size: 0.78rem !important;
                white-space: nowrap !important;
                min-width: fit-content !important;
            }
        }

        /* ================================================================
           25. SKELETON / SHIMMER PLACEHOLDERS
           ================================================================ */
        .spp-skeleton {
            background: var(--card-bg);
            border-radius: var(--radius-md);
            border: 1px solid var(--card-border);
            padding: 20px;
            margin-bottom: 12px;
            position: relative;
            overflow: hidden;
        }
        .spp-skeleton::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
        }
        .spp-skeleton-line {
            height: 14px;
            background: rgba(255,255,255,0.06);
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .spp-skeleton-line.short { width: 60%; }
        .spp-skeleton-line.medium { width: 80%; }
        .spp-skeleton-circle {
            width: 48px; height: 48px;
            border-radius: 50%;
            background: rgba(255,255,255,0.06);
            margin-bottom: 12px;
        }

        /* ================================================================
           26. GLASS METRIC CARD
           ================================================================ */
        .glass-metric-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            padding: 18px 20px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .glass-metric-card:hover {
            border-color: var(--card-border-hover);
            box-shadow: var(--shadow-card);
            transform: translateY(-2px);
        }
        .glass-metric-card .metric-label {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .glass-metric-card .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.2;
        }
        .glass-metric-card .metric-delta {
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 4px;
        }
        .glass-metric-card .metric-delta.positive { color: var(--accent-green); }
        .glass-metric-card .metric-delta.negative { color: var(--accent-red); }
        .glass-metric-card .metric-icon {
            position: absolute;
            top: 14px; right: 16px;
            font-size: 1.4rem;
            opacity: 0.35;
        }

        /* ================================================================
           27. ERROR STATE CARD
           ================================================================ */
        .error-state-card {
            background: var(--accent-red-dim);
            border: 1px solid rgba(255,90,90,0.25);
            border-radius: var(--radius-lg);
            padding: 24px;
            text-align: center;
        }
        .error-state-card .error-icon { font-size: 2rem; margin-bottom: 8px; }
        .error-state-card .error-title {
            font-size: 1rem; font-weight: 600; color: var(--accent-red); margin-bottom: 4px;
        }
        .error-state-card .error-msg {
            font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px;
        }

        /* ================================================================
           28. TOAST NOTIFICATION
           ================================================================ */
        .spp-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 99999;
            padding: 14px 24px;
            border-radius: var(--radius-md);
            font-size: 0.88rem;
            font-weight: 500;
            animation: fadeInUp 0.4s ease, fadeOutDown 0.4s ease 2.6s forwards;
            box-shadow: var(--shadow-elevated);
            backdrop-filter: blur(12px);
            pointer-events: none;
        }
        .spp-toast.success { background: rgba(0,230,138,0.15); border: 1px solid rgba(0,230,138,0.3); color: var(--accent-green); }
        .spp-toast.error { background: rgba(255,90,90,0.15); border: 1px solid rgba(255,90,90,0.3); color: var(--accent-red); }
        .spp-toast.info { background: rgba(59,158,255,0.15); border: 1px solid rgba(59,158,255,0.3); color: var(--accent-blue); }
        @keyframes fadeOutDown {
            from { opacity: 1; transform: translateY(0); }
            to { opacity: 0; transform: translateY(-16px); }
        }

        /* ================================================================
           29. COMPARISON VIEW
           ================================================================ */
        .comparison-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 24px;
            margin-bottom: 24px;
        }
        .comparison-vs {
            font-size: 1.2rem;
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ─── UI HELPER FUNCTIONS ────────────────────────────────────────────────────

def render_skeleton(rows: int = 3, show_circle: bool = False) -> None:
    """Render a shimmer skeleton placeholder while data loads."""
    lines_html = ""
    if show_circle:
        lines_html += '<div class="spp-skeleton-circle"></div>'
    for i in range(rows):
        cls = "short" if i % 3 == 0 else ("medium" if i % 3 == 1 else "")
        lines_html += f'<div class="spp-skeleton-line {cls}"></div>'
    st.markdown(f'<div class="spp-skeleton">{lines_html}</div>', unsafe_allow_html=True)


def show_toast(message: str, toast_type: str = "success") -> None:
    """Show a temporary toast notification (success/error/info)."""
    valid = {"success", "error", "info"}
    t = toast_type if toast_type in valid else "info"
    st.markdown(
        f'<div class="spp-toast {t}">{message}</div>',
        unsafe_allow_html=True,
    )


def glass_metric(label: str, value: str, delta: str = "", icon: str = "",
                 delta_positive: bool = True) -> None:
    """Render a premium glass-blur metric card."""
    delta_cls = "positive" if delta_positive else "negative"
    delta_html = f'<div class="metric-delta {delta_cls}">{delta}</div>' if delta else ""
    icon_html = f'<div class="metric-icon">{icon}</div>' if icon else ""
    st.markdown(f"""
        <div class="glass-metric-card">
            {icon_html}
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_error_card(title: str, message: str, icon: str = "⚠️",
                      retry_hint: str = "") -> None:
    """Render a styled error state card."""
    retry_html = (f'<div style="font-size:0.78rem;color:var(--text-muted);'
                  f'margin-top:8px;">{retry_hint}</div>') if retry_hint else ""
    st.markdown(f"""
        <div class="error-state-card">
            <div class="error-icon">{icon}</div>
            <div class="error-title">{title}</div>
            <div class="error-msg">{message}</div>
            {retry_html}
        </div>
    """, unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Render dark/light mode toggle in sidebar using session state."""
    if "spp_theme" not in st.session_state:
        st.session_state["spp_theme"] = "dark"
    current = st.session_state["spp_theme"]
    label = "☀️ Light Mode" if current == "dark" else "🌙 Dark Mode"
    if st.button(label, key="theme_toggle_btn", use_container_width=True):
        st.session_state["spp_theme"] = "light" if current == "dark" else "dark"
        st.rerun()
    # Inject data-theme attribute via JS
    theme = st.session_state["spp_theme"]
    st.markdown(f"""
        <script>
        (function() {{
            document.documentElement.setAttribute('data-theme', '{theme}');
        }})();
        </script>
    """, unsafe_allow_html=True)


def inject_keyboard_shortcuts() -> None:
    """Inject global keyboard shortcuts (Ctrl+K = focus search)."""
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            // Ctrl+K or Cmd+K → focus ticker search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                var inputs = document.querySelectorAll('input[type="text"]');
                if (inputs.length > 0) inputs[0].focus();
            }
        });
        </script>
    """, unsafe_allow_html=True)


def render_comparison_view(ticker_a: str, ticker_b: str,
                           data_a: dict, data_b: dict) -> None:
    """Render a side-by-side stock comparison table."""
    st.markdown(f"""
        <div class="comparison-header">
            <span style="font-size:1.3rem;font-weight:700;">{ticker_a}</span>
            <span class="comparison-vs">VS</span>
            <span style="font-size:1.3rem;font-weight:700;">{ticker_b}</span>
        </div>
    """, unsafe_allow_html=True)

    metrics_to_compare = [
        ("P/E", "P/E"), ("P/B", "P/B"), ("ROE", "ROE"), ("ROA", "ROA"),
        ("Debt/Equity", "D/E"), ("Profit Margin", "Profit Margin"),
        ("Revenue Growth", "Rev. Growth"), ("EPS Growth", "EPS Growth"),
    ]
    rows = []
    for label, key in metrics_to_compare:
        val_a = data_a.get(key, "N/A")
        val_b = data_b.get(key, "N/A")
        rows.append({"Metric": label, ticker_a: val_a, ticker_b: val_b})

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def normalize_ticker(raw: str) -> str:
    """Normalizuje běžné aliasy a zápisy tickerů (bez web lookup)."""
    t = (raw or "").strip().upper().replace(" ", "")
    # Common dot-class tickers on Yahoo
    t = t.replace(".A", "-A").replace(".B", "-B")
    if t in ("BRK.B", "BRK B"):
        t = "BRK-B"
    if t in ("BRK.A", "BRK A"):
        t = "BRK-A"
    crypto_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD"}
    if t in crypto_map:
        return crypto_map[t]
    m = re.fullmatch(r"([A-Z]{2,6})USD", t)
    if m:
        sym = m.group(1)
        if sym in crypto_map:
            return crypto_map[sym]
        return f"{sym}-USD"
    return t


def detect_asset_flags(ticker: str, info: dict) -> dict:
    """Určí typ aktiva a co má smysl zobrazovat. DCF/Earnings/Insiders/Peers = jen EQUITY."""
    qt = (info.get("quoteType") or "").upper().strip()
    t = (ticker or "").upper().strip()

    asset_class = "EQUITY"
    if t.startswith("^") or qt in ("INDEX", "INDEXQUOTE"):
        asset_class = "INDEX"
    elif qt in ("CRYPTOCURRENCY", "CRYPTO") or (t.endswith("-USD") and info.get("circulatingSupply") and not info.get("sector")):
        asset_class = "CRYPTO"
    elif qt == "ETF":
        asset_class = "ETF"
    elif qt in ("MUTUALFUND", "MUTUAL FUND", "FUND"):
        asset_class = "FUND"
    elif qt in ("CURRENCY", "FX"):
        asset_class = "FX"

    eq_only = (asset_class == "EQUITY")
    return {
        "quoteType": qt or "—",
        "asset_class": asset_class,
        "supports_dcf": eq_only,
        "supports_earnings": eq_only,
        "supports_insiders": eq_only,
        "supports_peers": eq_only,
        "supports_fundamentals": eq_only,
    }


def _diag_init():
    if "_diagnostics" not in st.session_state:
        st.session_state["_diagnostics"] = []
    if "_diag_sources" not in st.session_state:
        st.session_state["_diag_sources"] = {}


def diag_log(msg: str, level: str = "INFO"):
    _diag_init()
    try:
        ts = dt.datetime.now().strftime("%H:%M:%S")
    except Exception:
        ts = "—"
    st.session_state["_diagnostics"].append({"t": ts, "level": level, "msg": str(msg)})
    st.session_state["_diagnostics"] = st.session_state["_diagnostics"][-200:]


def diag_set_source(key: str, value):
    _diag_init()
    st.session_state["_diag_sources"][key] = value


def render_diagnostics_panel():
    if not st.session_state.get("debug_mode", False):
        return
    _diag_init()
    with st.expander("🧪 Diagnostika & Zdroje", expanded=False):
        st.markdown("**Zdroje (provenance):**")
        st.json(st.session_state.get("_diag_sources", {}))
        st.markdown("---")
        st.markdown("**Události:**")
        events = list(reversed(st.session_state.get("_diagnostics", [])))
        if not events:
            st.caption("(žádné události)")
        else:
            for e in events[:80]:
                st.write(f"[{e.get('t','—')}] {e.get('level','INFO')}: {e.get('msg','')}")
        if st.button("🧹 Vyčistit diagnostiku", use_container_width=True):
            st.session_state["_diagnostics"] = []
            st.session_state["_diag_sources"] = {}
            st.rerun()


def na_box(title: str, reason: str, help_text: str = ""):
    st.info(f"**{title}:** N/A – {reason} {qmark(help_text) if help_text else ''}", icon="ℹ️")


def reverse_exit_multiple_implied_growth(price, fcf0, wacc, years, shares, cash, debt, exit_multiple):
    """Reverse DCF implied růst (Exit Multiple metoda) – binary search."""
    try:
        if not (price and fcf0 and wacc and years and shares and exit_multiple):
            return None
        if fcf0 <= 0 or price <= 0 or shares <= 0:
            return None
        target_equity = price * shares
        target_ev = target_equity - cash + debt

        def ev_for_g(g):
            cf = float(fcf0)
            pv_sum = 0.0
            for y in range(1, int(years) + 1):
                cf *= (1 + g)
                pv_sum += cf / ((1 + wacc) ** y)
            tv = cf * float(exit_multiple)
            pv_tv = tv / ((1 + wacc) ** int(years))
            return pv_sum + pv_tv

        lo, hi = -0.5, 1.0
        ev_lo, ev_hi = ev_for_g(lo), ev_for_g(hi)
        if not (min(ev_lo, ev_hi) <= target_ev <= max(ev_lo, ev_hi)):
            hi2 = 2.0
            ev_hi2 = ev_for_g(hi2)
            if min(ev_lo, ev_hi2) <= target_ev <= max(ev_lo, ev_hi2):
                hi, ev_hi = hi2, ev_hi2
            else:
                return None
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if ev_for_g(mid) < target_ev:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    except Exception:
        return None


def build_data_quality(ticker, info, asset_flags, fcf_raw, fcf_used, used_ocf_proxy, shares_estimated, cash_missing, debt_missing, one_off_flag):
    price_cur = info.get("currency")
    fin_cur = info.get("financialCurrency")
    mismatch = bool(price_cur and fin_cur and price_cur != fin_cur)
    return {
        "asset_class": asset_flags.get("asset_class"),
        "quoteType": asset_flags.get("quoteType"),
        "price_currency": price_cur,
        "financial_currency": fin_cur,
        "currency_mismatch": mismatch,
        "fcf_raw": fcf_raw,
        "fcf_used": fcf_used,
        "used_ocf_proxy": used_ocf_proxy,
        "shares_estimated": shares_estimated,
        "cash_missing": cash_missing,
        "debt_missing": debt_missing,
        "one_off_cashflow_flag": one_off_flag,
    }


def compute_dcf_confidence(dq):
    score = 100
    reasons = []
    if dq.get("currency_mismatch"):
        score -= 10
        reasons.append("Měna ceny ≠ měna výkazů (může zkreslit DCF).")
    if dq.get("shares_estimated"):
        score -= 15
        reasons.append("Počet akcií je odhad (marketCap/price).")
    if dq.get("used_ocf_proxy"):
        score -= 20
        reasons.append("Použit OCF proxy místo skutečného FCF.")
    if dq.get("cash_missing") or dq.get("debt_missing"):
        score -= 10
        reasons.append("Chybí cash/debt údaje (equity bridge méně přesný).")
    if dq.get("one_off_cashflow_flag"):
        score -= 10
        reasons.append("Detekován možný one-off / outlier v cashflow.")
    score = max(0, min(100, int(score)))
    return score, reasons


def js_close_sidebar():
    """Return HTML+JS that attempts to close Streamlit sidebar/drawer (mobile + desktop)."""
    return """
    <script>
      (function () {
        function getDoc() {
          try { return (window.parent && window.parent.document) ? window.parent.document : document; }
          catch (e) { return document; }
        }

        function isSidebarOpen(doc) {
          var sb = doc.querySelector('section[data-testid="stSidebar"], [data-testid="stSidebar"]');
          if (!sb) return false;
          try {
            var r = sb.getBoundingClientRect();
            // On desktop sidebar has width; on mobile drawer may overlay with width as well
            return (r.width && r.width > 40) || (r.right && r.right > 40);
          } catch (e) {
            return true;
          }
        }

        function findCloseButton(doc) {
          var selectors = [
            'button[aria-label="Close sidebar"]',
            'button[aria-label="Collapse sidebar"]',
            'button[title="Close sidebar"]',
            '[data-testid="stSidebarCollapseButton"]',
            '[data-testid="stSidebarToggleButton"]',
            'header button[aria-label="Close sidebar"]',
            'header button[aria-label="Collapse sidebar"]',
            'header [data-testid="stSidebarCollapseButton"]',
            'header [data-testid="stSidebarToggleButton"]'
          ];
          for (var i = 0; i < selectors.length; i++) {
            var el = doc.querySelector(selectors[i]);
            if (el) return el;
          }
          // Fallback: first button inside sidebar section
          var sb = doc.querySelector('section[data-testid="stSidebar"], [data-testid="stSidebar"]');
          if (sb) {
            var b = sb.querySelector('button');
            if (b) return b;
          }
          return null;
        }

        function attemptClose() {
          var doc = getDoc();
          if (!isSidebarOpen(doc)) return true; // nothing to do
          var btn = findCloseButton(doc);
          if (btn) {
            btn.click();
            // second click helps on some mobile browsers
            setTimeout(function(){ try { btn.click(); } catch(e){} }, 120);
            return true;
          }
          return false;
        }

        var tries = 0;
        var maxTries = 25;
        var timer = setInterval(function () {
          tries++;
          var ok = false;
          try { ok = attemptClose(); } catch (e) { ok = false; }
          if (ok || tries >= maxTries) {
            clearInterval(timer);
          }
        }, 120);

        // Also try shortly after start
        setTimeout(function(){ try { attemptClose(); } catch(e){} }, 60);
      })();
    </script>
    """

def js_persist_tab(force_label: str = "") -> str:
    """Return HTML+JS that persistently tracks the active tab.

    If *force_label* is supplied (e.g. after an explicit st.rerun()),
    that tab is selected first and saved.  Otherwise the last-known tab
    from sessionStorage is restored.

    A MutationObserver continually watches for tab clicks so that
    every user-initiated tab switch is saved automatically.
    """
    force = json.dumps(force_label) if force_label else "null"
    return f"""
<script>
(function() {{
  const STORAGE_KEY = "spp_active_tab";
  const forceLabel = {force};

  function norm(s) {{
    return (s || "")
      .toLowerCase()
      .replace(/[^a-z0-9 ]/g, " ")
      .replace(/[\\s]+/g, " ")
      .trim();
  }}

  function clickTab(label) {{
    const doc = window.parent.document;
    const tabs = doc.querySelectorAll('[role="tab"], button[role="tab"]');
    const want = norm(label);
    for (const t of tabs) {{
      const txt = norm(t.innerText || t.textContent);
      if (txt && (txt === want || txt.includes(want) || want.includes(txt))) {{
        t.click();
        return true;
      }}
    }}
    return false;
  }}

  /* Install click-listeners on every tab so we track user switches */
  function installListeners() {{
    const doc = window.parent.document;
    const tabs = doc.querySelectorAll('[role="tab"], button[role="tab"]');
    tabs.forEach(function(t) {{
      if (!t.dataset.sppTracked) {{
        t.dataset.sppTracked = "1";
        t.addEventListener("click", function() {{
          const raw = (t.innerText || t.textContent || "").trim();
          if (raw) {{
            try {{ window.parent.sessionStorage.setItem(STORAGE_KEY, raw); }} catch(e) {{}}
          }}
        }});
      }}
    }});
    return tabs.length > 0;
  }}

  /* Determine which tab to restore */
  var target = forceLabel;
  if (!target) {{
    try {{ target = window.parent.sessionStorage.getItem(STORAGE_KEY); }} catch(e) {{}}
  }}
  if (target) {{
    try {{ window.parent.sessionStorage.setItem(STORAGE_KEY, target); }} catch(e) {{}}
  }}

  /* Retry loop – DOM may not be ready immediately */
  var tries = 0;
  var timer = setInterval(function() {{
    tries++;
    var ready = installListeners();
    if (target && ready) {{
      clickTab(target);
    }}
    if (tries > 30) clearInterval(timer);
  }}, 200);
}})();
</script>
"""


def _get_secret(name: str, default: str = "") -> str:
    """Centralizované a bezpečné načítání Secrets přes st.secrets.

    Pozn.: Pro lokální běh použij .streamlit/secrets.toml (nedoporučujeme env fallback, aby
    se v cloudu předešlo nechtěnému logování / úniku).
    """
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return str(default or "")

# Read from Streamlit secrets (preferred) or env.
# In Streamlit Cloud > App settings > Secrets:
# GEMINI_API_KEY="..."
# FMP_API_KEY="..."
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
FMP_API_KEY = _get_secret("FMP_API_KEY", "")
SEC_USER_AGENT = _get_secret("SEC_USER_AGENT", "StockPickerPro/1.0 (contact: your_email@example.com)")
ALPHAVANTAGE_API_KEY = _get_secret("ALPHAVANTAGE_API_KEY", "")
FINNHUB_API_KEY = _get_secret("FINNHUB_API_KEY", "")
NINJAS_API_KEY = _get_secret("NINJAS_API_KEY", "") or _get_secret("Ninjas_API_KEY", "")

# --- New v10.0 API keys ---
SUPABASE_URL = _get_secret("SUPABASE_URL", "")
SUPABASE_KEY = _get_secret("SUPABASE_KEY", "")
FRED_API_KEY = _get_secret("FRED_API_KEY", "")
POLYGON_API_KEY = _get_secret("POLYGON_API_KEY", "")
TWELVEDATA_API_KEY = _get_secret("TWELVEDATA_API_KEY", "")
NEWSAPI_KEY = _get_secret("NEWSAPI_KEY", "")

# PDF Export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False


def generate_pdf_report(ticker: str, info: dict, metrics: dict) -> Optional[bytes]:
    """Generate a PDF report for a stock analysis. Returns bytes or None."""
    if not _HAS_PDF:
        return None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.lib.units import inch
        import io

        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        y = height - 1 * inch

        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(1 * inch, y, f"Stock Picker Pro – {ticker}")
        y -= 0.4 * inch
        c.setFont("Helvetica", 10)
        company = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "N/A")
        c.drawString(1 * inch, y, f"{company} | Sector: {sector}")
        y -= 0.5 * inch

        # Key Metrics table
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, y, "Key Metrics")
        y -= 0.3 * inch
        c.setFont("Helvetica", 10)

        metric_keys = [
            ("Price", "currentPrice"), ("P/E", "trailingPE"),
            ("P/B", "priceToBook"), ("EV/EBITDA", "enterpriseToEbitda"),
            ("ROE", "returnOnEquity"), ("ROA", "returnOnAssets"),
            ("Profit Margin", "profitMargins"), ("Debt/Equity", "debtToEquity"),
            ("Revenue Growth", "revenueGrowth"), ("Dividend Yield", "dividendYield"),
        ]
        for label, key in metric_keys:
            val = info.get(key) or metrics.get(key, "N/A")
            if isinstance(val, float):
                if key in ("returnOnEquity", "returnOnAssets", "profitMargins",
                           "revenueGrowth", "dividendYield"):
                    val = f"{val * 100:.2f}%"
                else:
                    val = f"{val:,.2f}"
            c.drawString(1 * inch, y, f"{label}:")
            c.drawString(3.5 * inch, y, str(val))
            y -= 0.22 * inch
            if y < 1 * inch:
                c.showPage()
                y = height - 1 * inch
                c.setFont("Helvetica", 10)

        # Custom metrics from analysis
        if metrics:
            y -= 0.3 * inch
            c.setFont("Helvetica-Bold", 14)
            c.drawString(1 * inch, y, "Analysis Scores")
            y -= 0.3 * inch
            c.setFont("Helvetica", 10)
            score_keys = ["Piotroski", "Altman Z", "DCF", "MOS", "Graham Number"]
            for sk in score_keys:
                if sk in metrics:
                    c.drawString(1 * inch, y, f"{sk}:")
                    c.drawString(3.5 * inch, y, str(metrics[sk]))
                    y -= 0.22 * inch

        # Footer
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(1 * inch, 0.5 * inch,
                     "Generated by Stock Picker Pro · Not financial advice")
        c.save()
        return buf.getvalue()
    except Exception:
        return None

# Constants
APP_NAME = "Stock Picker Pro"
APP_VERSION = "v9.0"

GEMINI_MODEL = "gemini-2.5-flash-lite"  # Optimized for Free Tier
MAX_AI_RETRIES = 3  # Retry logic for rate limits
RETRY_DELAY = 2  # seconds

# METRIC_TOOLTIPS imported from spp_constants (single source of truth)

def metric_help(key: str) -> Optional[str]:
    """Vrátí tooltip text pro danou metriku nebo None."""
    return METRIC_TOOLTIPS.get(key)


def qmark(help_text: str) -> str:
    """Small inline question-mark tooltip for section headers (HTML title attribute)."""
    try:
        safe = html.escape(str(help_text or ""))
    except Exception:
        safe = str(help_text or "")
    return f"<span style='font-size:0.9rem; opacity:0.65; cursor:help;' title='{safe}'>❔</span>"

def section_title(title: str, help_text: str = "", level: int = 3) -> None:
    """Render a section title with a hover tooltip question mark."""
    tag = f"h{max(1, min(6, int(level)))}"
    st.markdown(
        f"<{tag} style='margin-top:0.8rem; margin-bottom:0.35rem;'>{title} {qmark(help_text) if help_text else ''}</{tag}>",
        unsafe_allow_html=True,
    )



# ============================================================================


# -----------------------------------------------------------------------------
# Social & Guru (X/Twitter) handles
# -----------------------------------------------------------------------------
GURUS = {
    "CZ/SK Scéna": {
        "Jaroslav Brychta": "JaroslavBrychta",
        "Dominik Stroukal": "stroukal",
        "Jaroslav Šura": "jarsura",
        "Tomáš Plecháč": "TPlechac",
        "Akciový Guru": "akciovyguru",
        "Nicnevim": "Nicnevim11",
        "Bulios": "Bulios_cz",
        "Michal Semotan": "MichalSemotan",
    },
    "Global & News": {
        "Walter Bloomberg (News)": "DeItaone",
        "Brian Feroldi (Education)": "BrianFeroldi",
        "App Economy Insights": "AppEconomyInsights",
    },
}

DATA_DIR = os.path.join(os.path.dirname(__file__), ".stock_picker_pro")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
MEMOS_PATH = os.path.join(DATA_DIR, "memos.json")

# Sector to peers mapping (expand as needed)
SECTOR_PEERS = {
    "Technology": {
        "AAPL": ["MSFT", "GOOGL", "META", "NVDA"],
        "MSFT": ["AAPL", "GOOGL", "META", "AMZN"],
        "GOOGL": ["AAPL", "MSFT", "META", "AMZN"],
        "META": ["AAPL", "GOOGL", "SNAP", "PINS"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
        "TSLA": ["RIVN", "LCID", "F", "GM"],
        "NFLX": ["DIS", "PARA", "WBD"],
    },
    "Consumer Cyclical": {
        "AMZN": ["WMT", "TGT", "EBAY", "BABA"],
        "TSLA": ["F", "GM", "RIVN", "LCID"],
    },
    "Healthcare": {
        "JNJ": ["PFE", "UNH", "ABT", "MRK"],
        "PFE": ["JNJ", "MRK", "ABBV", "LLY"],
    },
    "Financial Services": {
        "JPM": ["BAC", "WFC", "C", "GS"],
        "V": ["MA", "PYPL", "SQ"],
        "KOMB.PR": ["MONETA.PR", "JPM", "BAC"],  # Czech: Komerční banka
        "MONETA.PR": ["KOMB.PR", "JPM", "BAC"],  # Czech: Moneta Money Bank
    },
    "Communication Services": {
        "T": ["VZ", "TMUS"],
    },
    "Utilities": {
        "CEZ.PR": ["NEE", "DUK", "SO", "D"],  # Czech: ČEZ
        "NEE": ["DUK", "SO", "D", "AEP"],
    },
}

# Macro Calendar Events (Feb-Mar 2026)
MACRO_CALENDAR = [
    {"date": "2026-02-20", "event": "FOMC Minutes Release", "importance": "High"},
    {"date": "2026-03-06", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-03-11", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-03-18", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-03-25", "event": "US GDP (Q4 2025 Final)", "importance": "Medium"},
]



# ============================================================================
# UTILITIES & QUANT LOGIC
# ============================================================================

def calculate_roic(info: Dict[str, Any]) -> Optional[float]:
    """Aproximace ROIC: NOPAT / Invested Capital.
    Používá EBIT (ne EBITDA) pro správný výpočet NOPAT.
    """
    try:
        # Prefer EBIT; fallback to EBITDA - D&A proxy if not available
        ebit = safe_float(info.get("ebit"))
        if ebit is None:
            ebitda = safe_float(info.get("ebitda"))
            da = safe_float(info.get("depreciationAndAmortization") or info.get("totalDepreciationAndAmortization"))
            if ebitda is not None:
                ebit = ebitda - (da or 0)
        nopat = ebit * 0.79 if ebit is not None else None  # 21% US Tax proxy
        invested_capital = (safe_float(info.get("totalDebt")) or 0) + (safe_float(info.get("totalStockholderEquity")) or 0)
        return safe_div(nopat, invested_capital)
    except:
        return None

# ============================================================================
# NOVÉ ANALYTICKÉ FUNKCE v6.0
# ============================================================================

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


def calculate_piotroski_fscore(info: Dict[str, Any], income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> Tuple[int, Dict[str, int]]:
    """
    Piotroski F-Score (0-9): 9-bodový fundamental quality check.
    Vyšší = lepší kvalita fundamentů.
    """
    score = 0
    breakdown: Dict[str, int] = {}

    def _row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
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
            score += p1; breakdown["ROA > 0"] = p1

        ocf = safe_float(info.get("operatingCashflow"))
        if ocf is not None:
            p2 = 1 if ocf > 0 else 0
            score += p2; breakdown["OCF > 0"] = p2

        # Change in ROA (YoY) - from income statement
        if not income.empty and len(income.columns) >= 2:
            ni_row = _row(income, ["Net Income", "Net Income Applicable To Common Shares"])
            ta_row = _row(balance, ["Total Assets"]) if not balance.empty else None
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
                        score += p3; breakdown["ΔROAᵧ > 0"] = p3
                except Exception:
                    pass

        # Accruals: OCF/Assets > ROA
        ta_val = safe_float(info.get("totalAssets"))
        if ocf is not None and roa is not None and ta_val and ta_val > 0:
            p4 = 1 if (ocf / ta_val) > roa else 0
            score += p4; breakdown["OCF/Assets > ROA"] = p4

        # --- Leverage & Liquidity (3 body) ---
        de_curr = safe_float(info.get("debtToEquity"))
        if de_curr is not None:
            # Normalizace: Yahoo Finance vrací ×100 formát
            de_normalized = de_curr / 100.0 if de_curr > 10 else de_curr
            # Ideálně bychom porovnali s předchozím rokem, ale info dává jen aktuální
            # Jako proxy: nízký D/E je dobré znamení (< 1.0 po normalizaci)
            p5 = 1 if de_normalized < 1.0 else 0  # D/E < 1.0 (normalizováno)
            score += p5; breakdown["D/E < 1.0"] = p5

        cr = safe_float(info.get("currentRatio"))
        if cr is not None:
            p6 = 1 if cr > 1 else 0
            score += p6; breakdown["Current Ratio > 1"] = p6

        # Dilution: shares outstanding - pokud rostou, je to negativní
        # Ředění: porovnáme impliedShares (z market cap / price) vs sharesOutstanding
        # Pokud firma aktivně odkupuje (buybacks), shares klesají = pozitivní
        shares_curr = safe_float(info.get("sharesOutstanding"))
        # Proxy: buyback yield > 0 nebo nízká % změna impliujeme z treasury
        buyback = safe_float(info.get("repurchaseOfStock") or info.get("commonStockRepurchased"))
        if shares_curr and buyback is not None:
            p7 = 1 if buyback < 0 else 0  # negativní = firma zpětně odkupuje (pozitivní sign)
            score += p7; breakdown["Zpětné odkupy (bez ředění)"] = p7
        elif shares_curr:
            # Fallback: pokud není info o buybacku, neutrálně přiřadíme 0
            breakdown["Zpětné odkupy (bez ředění)"] = 0

        # --- Efektivita (2 body) ---
        gm_curr = safe_float(info.get("grossMargins"))
        if gm_curr is not None:
            p8 = 1 if gm_curr > 0.30 else 0
            score += p8; breakdown["Gross Margin > 30%"] = p8

        asset_turnover = safe_div(safe_float(info.get("totalRevenue")), safe_float(info.get("totalAssets")))
        if asset_turnover is not None:
            p9 = 1 if asset_turnover > 0.5 else 0
            score += p9; breakdown["Asset Turnover > 0.5"] = p9

    except Exception:
        pass

    return score, breakdown


def calculate_altman_zscore(
    info: Dict[str, Any],
    income: Optional[pd.DataFrame] = None,
    balance: Optional[pd.DataFrame] = None,
    market_cap: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """
    Altman Z-Score: bankruptcy risk indicator (Manufacturing variant).
    Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    
    A = Working Capital / Total Assets
    B = Retained Earnings / Total Assets
    C = EBIT / Total Assets
    D = Market Value of Equity / Total Liabilities
    E = Sales / Total Assets
    """
    try:
        sector = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        # Finanční sektor se nehodnotí Altmanem (mají jinou strukturu rozvahy)
        if "financial" in sector or any(k in industry for k in ["bank", "insurance", "capital markets", "asset management"]):
            return None, "N/A (Finanční sektor)"

        # Helper: Robust DataFrame Value Extraction
        def _df_get(df: Optional[pd.DataFrame], keys: List[str]) -> Optional[float]:
            if df is None or getattr(df, "empty", True):
                return None
            
            # 1. Exact match
            for k in keys:
                if k in df.index:
                    try:
                        val = safe_float(df.loc[k].iloc[0])
                        if val is not None: return val
                    except: pass
            
            # 2. Case-insensitive / strip match
            idx_map = {str(i).strip().lower(): i for i in df.index}
            for k in keys:
                clean_k = k.strip().lower()
                if clean_k in idx_map:
                    try:
                        val = safe_float(df.loc[idx_map[clean_k]].iloc[0])
                        if val is not None: return val
                    except: pass
            return None

        # ── 1. Total Assets ──
        ta = safe_float(info.get("totalAssets")) or _df_get(balance, ["Total Assets", "Assets"])
        if not ta or ta <= 0:
            return None, "Chybí Total Assets"

        # ── 2. Working Capital (CA - CL) ──
        ca = safe_float(info.get("totalCurrentAssets")) or _df_get(balance, ["Total Current Assets", "Current Assets"])
        cl = safe_float(info.get("totalCurrentLiabilities")) or _df_get(balance, ["Total Current Liabilities", "Current Liabilities"])
        
        wc = 0.0
        if ca is not None and cl is not None:
            wc = ca - cl
        else:
            # Fallback estimation if explicit WC is missing
            wc_guess = safe_float(info.get("workingCapital"))
            if wc_guess: wc = wc_guess

        # ── 3. Retained Earnings ──
        re_val = (
            safe_float(info.get("retainedEarnings") or info.get("retainedEarningsAccumulatedDeficit"))
            or _df_get(balance, ["Retained Earnings", "Retained Earnings (Accumulated Deficit)", "Retained Earnings (Accumulated Deficit)"])
            or 0.0
        )

        # ── 4. EBIT ──
        ebit = safe_float(info.get("ebit")) or _df_get(income, ["EBIT", "Ebit", "Operating Income", "Operating Profit"])
        if ebit is None:
            # Fallback: EBITDA - D&A
            ebitda = safe_float(info.get("ebitda")) or _df_get(income, ["EBITDA", "Normalized EBITDA"])
            if ebitda:
                ebit = ebitda  # Approximation if D&A missing, better than 0

        # ── 5. Market Value of Equity (Market Cap) ──
        mve = market_cap
        if not mve:
            mve = safe_float(info.get("marketCap"))
        if not mve:
            p = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
            sh = safe_float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
            if p and sh:
                mve = p * sh
        
        # ── 6. Total Liabilities ──
        tl = safe_float(info.get("totalLiabilities")) or _df_get(balance, ["Total Liabilities", "Total Liab", "Total Liabilities Net Minority Interest"])
        if not tl:
            # Estimate: Assets - Equity
            te = safe_float(info.get("totalStockholderEquity")) or _df_get(balance, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"])
            if te and ta:
                tl = ta - te
        
        # ── 7. Sales (Revenue) ──
        sales = safe_float(info.get("totalRevenue")) or _df_get(income, ["Total Revenue", "Operating Revenue", "Revenue"]) or 0.0

        # Data Validation for critical components
        if mve is None or mve <= 0: return None, "Chybí Market Cap"
        if tl is None or tl <= 0: return None, "Chybí Total Liabilities"
        if ebit is None: ebit = 0.0 # Conservative fallback

        # ── Calculation ──
        A = wc / ta
        B = re_val / ta
        C = ebit / ta
        D = mve / tl
        E = sales / ta

        Z = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)
        
        # Verdict
        if Z > 2.99:
            zone = "✅ Bezpečná zóna"
        elif Z > 1.81:
            zone = "⚠️ Šedá zóna"
        else:
            zone = "🚨 Riziko bankrotu"

        return round(float(Z), 2), zone

    except Exception:
        return None, "Chyba výpočtu"

def calculate_rsi(price_series: pd.Series, period: int = 14) -> Optional[float]:
    """Relative Strength Index (RSI)."""
    try:
        if len(price_series) < period + 1:
            return None
        delta = price_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def calculate_macd(price_series: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
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
        if m > s:
            label = "📈 Bullish crossover"
        elif m < s:
            label = "📉 Bearish crossover"
        else:
            label = "➡️ Neutrální"
        return m, s, label
    except Exception:
        return None, None, "Chyba"


def calculate_technical_signals(price_history: pd.DataFrame) -> Dict[str, Any]:
    """Vypočítá sadu technických indikátorů z cenové historie."""
    result: Dict[str, Any] = {}
    if price_history.empty or "Close" not in price_history.columns:
        return result

    close = price_history["Close"].dropna()
    if len(close) < 20:
        return result

    # RSI
    result["rsi"] = calculate_rsi(close)

    # MACD
    macd_val, signal_val, macd_label = calculate_macd(close)
    result["macd"] = macd_val
    result["macd_signal"] = signal_val
    result["macd_label"] = macd_label

    # Moving averages
    result["ma50"] = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    result["ma200"] = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    result["current_price"] = float(close.iloc[-1])

    # 52W High/Low
    result["high_52w"] = float(close.rolling(252).max().iloc[-1]) if len(close) >= 20 else float(close.max())
    result["low_52w"] = float(close.rolling(252).min().iloc[-1]) if len(close) >= 20 else float(close.min())

    # Bollinger Bands (20-day)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    result["bb_upper"] = float((ma20 + 2 * std20).iloc[-1]) if len(close) >= 20 else None
    result["bb_lower"] = float((ma20 - 2 * std20).iloc[-1]) if len(close) >= 20 else None
    result["bb_mid"] = float(ma20.iloc[-1]) if len(close) >= 20 else None

    # Distance from 200MA
    if result.get("ma200") and result["ma200"] > 0:
        result["pct_from_ma200"] = (result["current_price"] / result["ma200"] - 1)

    # Volume trend (avg last 20 days vs avg last 60 days)
    if "Volume" in price_history.columns:
        vol = price_history["Volume"].dropna()
        if len(vol) >= 60:
            result["vol_trend"] = float(vol.iloc[-20:].mean() / vol.iloc[-60:].mean() - 1)

    return result




def calculate_dcf_fair_value(
    fcf: float,
    growth_rate: float,
    terminal_growth: float,
    wacc: float,
    years: int,
    shares_outstanding: float,
    total_cash: float = 0.0,
    total_debt: float = 0.0,
    exit_multiple: Optional[float] = None,
) -> Optional[float]:
    """Spočítá férovou cenu akcie (DCF) na základě FCF.

    Podporované terminální metody:
    - Exit Multiple (doporučeno): TV = FCF_YearN * exit_multiple
    - Gordon Growth (fallback):   TV = FCF_YearN * (1+g_term) / (WACC - g_term)

    Vrací fair value *na akcii* (Equity Value / shares).
    Pozn.: terminal_growth se použije jen když exit_multiple není zadán.
    """
    try:
        if fcf is None or shares_outstanding is None or shares_outstanding <= 0:
            return None
        if wacc is None or float(wacc) <= 0:
            return None

        cf = float(fcf)
        pv_sum = 0.0
        yrs = int(years)

        for year in range(1, yrs + 1):
            cf *= (1.0 + float(growth_rate))
            pv_sum += cf / ((1.0 + float(wacc)) ** year)

        if exit_multiple is not None:
            tv = cf * float(exit_multiple)
        else:
            if float(wacc) <= float(terminal_growth):
                return None
            tv = cf * (1.0 + float(terminal_growth)) / (float(wacc) - float(terminal_growth))

        pv_tv = tv / ((1.0 + float(wacc)) ** yrs)
        enterprise_value = pv_sum + pv_tv

        equity_value = enterprise_value + float(total_cash or 0.0) - float(total_debt or 0.0)
        return equity_value / float(shares_outstanding)
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
    exit_multiple: Optional[float] = None,
    n_simulations: int = 1000
) -> Dict[str, Any]:
    """Monte Carlo simulace DCF - vrací distribuci fair values.

    - Pokud je `exit_multiple` zadán, simulace používá Exit Multiple metodu (konzistentní s hlavním DCF).
      Náhodně rozptyluje: growth, WACC a exit multiple.
    - Pokud `exit_multiple` není zadán, použije se Gordon Growth terminál (terminal_growth) a rozptyluje se i terminal growth.

    Vrací agregace (mean/median/p10/p90/...) nad validními scénáři.
    """
    try:
        results: List[float] = []
        rng = np.random.default_rng(42)

        base_g = float(growth_rate)
        base_w = float(wacc)
        base_exit = float(exit_multiple) if exit_multiple is not None else None
        base_term = float(terminal_growth)

        # Nastavíme minimální volatilitu, aby simulace fungovala i když je růst ~0
        g_sigma = max(abs(base_g) * 0.30, 0.01)
        w_sigma = max(abs(base_w) * 0.15, 0.005)

        for _ in range(int(n_simulations)):
            sim_growth = float(rng.normal(base_g, g_sigma))
            sim_wacc = float(rng.normal(base_w, w_sigma))

            # Clamp na rozumné hranice
            sim_wacc = max(0.05, min(0.25, sim_wacc))

            if base_exit is not None:
                exit_sigma = max(abs(base_exit) * 0.20, 1.0)
                sim_exit = float(rng.normal(base_exit, exit_sigma))
                sim_exit = max(5.0, min(80.0, sim_exit))

                fv = calculate_dcf_fair_value(
                    fcf, sim_growth, base_term, sim_wacc, years, shares_outstanding,
                    total_cash, total_debt, exit_multiple=sim_exit
                )
            else:
                sim_terminal = float(rng.normal(base_term, 0.005))
                sim_terminal = max(0.0, min(0.05, sim_terminal))
                if sim_wacc <= sim_terminal:
                    continue

                fv = calculate_dcf_fair_value(
                    fcf, sim_growth, sim_terminal, sim_wacc, years, shares_outstanding,
                    total_cash, total_debt, exit_multiple=None
                )

            if fv and fv > 0:
                results.append(float(fv))

        if not results:
            return {}

        arr = np.array(results, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(np.std(arr)),
            "n": int(arr.shape[0]),
        }
    except Exception:
        return {}


def calculate_mean_reversion_pe(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Porovnání aktuálního P/E s historickým průměrem (5 let).
    Vrací (current_pe, hist_avg_pe).
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5y", interval="3mo", auto_adjust=False)
        if hist.empty:
            return None, None
        # yfinance nevrací historické PE přímo - používáme cenu a EPS odhad
        # Jako proxy: porovnáme P/B nebo P/S přes dobu
        info = t.info
        curr_pe = safe_float(info.get("trailingPE"))
        return curr_pe, None  # simplified - plná implementace by potřebovala historical EPS
    except Exception:
        return None, None


def calculate_earnings_quality(info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Earnings Quality: CFO / Net Income ratio.
    Ratio < 0.8 = možné manipulace s čísly.
    """
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


def simulate_investment(ticker: str, amount_czk: float, years_back: int = 5) -> Optional[Dict[str, Any]]:
    """
    'Co kdybych investoval X Kč?' simulátor.
    Porovnání s SPY (S&P 500 ETF).
    """
    try:
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


def get_short_interest(info: Dict[str, Any]) -> Optional[float]:
    """Short interest jako % float shares."""
    return safe_float(info.get("shortPercentOfFloat"))


def detect_value_trap(info: Dict[str, Any], metrics: Dict[str, "Metric"]) -> Tuple[bool, str]:
    return _detect_value_trap_impl(info, metrics)


def detect_market_regime(price_history: pd.DataFrame) -> str:
    """Detekce režimu na základě volatility a trendu za 6 měsíců."""
    if price_history.empty or len(price_history) < 20: return "Stable / Neutral"
    returns = price_history['Close'].pct_change().dropna()
    vol = returns.std() * math.sqrt(252)
    avg_ret = returns.mean() * 252
    
    if vol > 0.28 and avg_ret < -0.10: return "High Volatility / Bear"
    if vol < 0.18 and avg_ret > 0.05: return "Low Volatility / Bull"
    return "Stable / Transition"
    
# JSON helpers removed – use spp_database module instead (db._load_json / db._save_json)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (np.generic,)):
            x = x.item()
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return float(x)
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x == "":
                return None
            v = float(x)
            if math.isfinite(v):
                return v
        return None
    except Exception:
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    a = safe_float(a)
    b = safe_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def fmt_pct(x: Any, digits: int = 1) -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{v*100:.{digits}f}%"


def fmt_money(x: Any, digits: int = 2, prefix: str = "$") -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{prefix}{v:,.{digits}f}"


def clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


# ============================================================================
# DATA FETCHING (CACHED)
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Fetch basic info from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_price_history(ticker: str, period: str = "1y", auto_adjust: bool = False) -> pd.DataFrame:
    """Fetch historical price data (cache). auto_adjust=True -> upravené ceny (splity/dividendy)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=bool(auto_adjust))
        return df if isinstance(df, pd.DataFrame) and (not df.empty) else pd.DataFrame()
    except Exception as e:
        diag_log(f"fetch_price_history error: {e}", "WARN")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch income statement, balance sheet, and cash flow."""
    try:
        t = yf.Ticker(ticker)
        income = t.financials
        balance = t.balance_sheet
        cashflow = t.cashflow
        return (
            income if income is not None else pd.DataFrame(),
            balance if balance is not None else pd.DataFrame(),
            cashflow if cashflow is not None else pd.DataFrame()
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def get_fcf_ttm_yfinance(ticker: str, market_cap: Optional[float] = None) -> Tuple[Optional[float], List[str]]:
    """Robustně spočítá roční Free Cash Flow (TTM) z yfinance quarterly_cashflow.

    Pravidla:
    - Primárně sečte poslední 4 dostupné kvartály (TTM).
    - Když chybí řádek 'Free Cash Flow', spočítá FCF jako Operating Cash Flow - |CapEx|.
    - Pokud jsou dostupná jen 1-3 kvartální čísla, annualizuje průměrem ×4.
    - Sanity check: pro obří firmy (MarketCap > $1T) a podezřele nízké FCF (< $30B)
      aplikuje pojistku násobení 4× (typicky když provider vrátí jen 1 kvartál).
    - Vrací (fcf_ttm, dbg) kde dbg je list informativních zpráv.
    """
    dbg: List[str] = []
    try:
        t = yf.Ticker(ticker)
        qcf = getattr(t, "quarterly_cashflow", None)
        if qcf is None or not isinstance(qcf, pd.DataFrame) or qcf.empty:
            dbg.append("FCF: quarterly_cashflow není k dispozici (prázdné). Zkouším fallback.")
            qcf = pd.DataFrame()

        def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            if df is None or df.empty:
                return None
            idx = set(map(str, df.index))
            for c in candidates:
                if c in idx:
                    return c
            # zkus case-insensitive match
            low_map = {str(i).strip().lower(): str(i) for i in df.index}
            for c in candidates:
                key = c.strip().lower()
                if key in low_map:
                    return low_map[key]
            return None

        def _sorted_quarter_cols(df: pd.DataFrame) -> List[Any]:
            cols = list(df.columns)
            if not cols:
                return []
            dts = pd.to_datetime(cols, errors="coerce")
            if dts.notna().any():
                order = sorted(range(len(cols)), key=lambda i: dts[i], reverse=True)
                return [cols[i] for i in order]
            return cols  # fallback: keep original order

        # 1) vyber poslední dostupné kvartály
        cols_sorted = _sorted_quarter_cols(qcf)
        cols_sel = cols_sorted[:4] if cols_sorted else []
        if cols_sel:
            dbg.append(f"FCF: Načítám kvartály: {', '.join([str(c) for c in cols_sel])}")
        else:
            dbg.append("FCF: Nenalezeny žádné kvartální sloupce v quarterly_cashflow.")

        # 2) primárně: přímý řádek Free Cash Flow
        fcf_row = _pick_row(qcf, ["Free Cash Flow", "FreeCashFlow", "Free cash flow"])
        used_method = None

        fcf_quarters = None
        non_null = 0

        if fcf_row and cols_sel:
            s = pd.to_numeric(qcf.loc[fcf_row, cols_sel], errors="coerce")
            non_null = int(s.notna().sum())
            if non_null > 0:
                fcf_quarters = s
                used_method = f"quarterly row '{fcf_row}'"
        # 3) fallback: OCF - |CapEx|
        if fcf_quarters is None and cols_sel:
            ocf_row = _pick_row(qcf, [
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
                "Total Cash From Operating Activities (Continuing Operations)",
                "Cash Flow From Continuing Operating Activities",
                "Net Cash Provided By Operating Activities",
            ])
            capex_row = _pick_row(qcf, [
                "Capital Expenditures",
                "Capital Expenditure",
                "CapitalExpenditures",
                "Purchase Of PPE",
                "Purchase of Property Plant Equipment",
            ])
            if ocf_row and capex_row:
                ocf = pd.to_numeric(qcf.loc[ocf_row, cols_sel], errors="coerce")
                capex = pd.to_numeric(qcf.loc[capex_row, cols_sel], errors="coerce")
                non_null = int((ocf.notna() & capex.notna()).sum())
                if non_null > 0:
                    # CapEx bývá záporný; chceme: FCF = OCF - |CapEx|
                    fcf_quarters = ocf - capex.abs()
                    used_method = f"computed: '{ocf_row}' - |'{capex_row}'|"

        # 4) pokud pořád nic, fallback na annual cashflow / info
        if fcf_quarters is None:
            # annual cashflow
            acf = getattr(t, "cashflow", None)
            if isinstance(acf, pd.DataFrame) and not acf.empty:
                acf_cols = _sorted_quarter_cols(acf)[:1]  # nejnovější rok
                fcf_row_a = _pick_row(acf, ["Free Cash Flow", "FreeCashFlow", "Free cash flow"])
                if fcf_row_a and acf_cols:
                    v = safe_float(acf.loc[fcf_row_a, acf_cols[0]])
                    if v is not None:
                        dbg.append("FCF: Používám annual cashflow (nejnovější rok) – řádek Free Cash Flow.")
                        used_method = "annual row 'Free Cash Flow'"
                        fcf_ttm = float(v)
                        msg = f"Použité roční FCF (TTM): ${fcf_ttm/1e9:.1f} miliard ({used_method})"
                        dbg.append(msg)
                        return fcf_ttm, dbg

            # last resort: info['freeCashflow']
            try:
                info = getattr(t, "info", None) or {}
            except Exception:
                info = {}
            v = safe_float(info.get("freeCashflow"))
            if v is not None:
                used_method = "info['freeCashflow'] (fallback)"
                fcf_ttm = float(v)
                msg = f"Použité roční FCF (TTM): ${fcf_ttm/1e9:.1f} miliard ({used_method})"
                dbg.append(msg)
                return fcf_ttm, dbg

            dbg.append("FCF: Nepodařilo se získat FCF ani z quarterly ani z annual ani z info.")
            return None, dbg

        # 5) TTM / extrapolace
        fcf_vals = pd.to_numeric(fcf_quarters, errors="coerce").dropna()
        n = int(fcf_vals.shape[0])
        applied_extrap = False
        used_sum4 = False

        if n >= 4:
            fcf_ttm = float(fcf_vals.iloc[:4].sum())
            used_sum4 = True
        elif n > 0:
            # annualizace průměrem ×4
            fcf_ttm = float(fcf_vals.mean() * 4.0)
            applied_extrap = True
        else:
            dbg.append("FCF: kvartální hodnoty jsou všechny NaN.")
            return None, dbg

        # 6) Sanity check (market cap > 1T & FCF < 30B) -> 4×
        mc = safe_float(market_cap)
        if (not applied_extrap) and used_sum4 and mc and mc > 1e12 and fcf_ttm < 30e9:
            fcf_ttm *= 4.0
            dbg.append("FCF: Sanity check aktivován (MarketCap > $1T a FCF < $30B) -> násobím 4× (podezření na 1 kvartál).")

        # 7) Debug zprávy
        if used_method:
            dbg.append(f"FCF metoda: {used_method}. Kvartály použity: {n}.")
        if applied_extrap:
            dbg.append(f"FCF: Extrapolace do roční báze (k dispozici {n} kvartály) -> průměr ×4.")
        if used_sum4:
            dbg.append("FCF: TTM = součet posledních 4 kvartálů.")

        msg = f"Použité roční FCF (TTM): ${fcf_ttm/1e9:.1f} miliard"
        dbg.append(msg)

        return fcf_ttm, dbg
    except Exception as e:
        dbg.append(f"FCF: chyba při výpočtu TTM: {e}")
        return None, dbg
@st.cache_data(show_spinner=False, ttl=86400)  # ATH mění jednou denně max
def get_all_time_high(ticker: str) -> Optional[float]:
    """Get all-time high price."""
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="max", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        col = "High" if "High" in h.columns else ("Close" if "Close" in h.columns else None)
        if not col:
            return None
        return float(pd.to_numeric(h[col], errors="coerce").max())
    except Exception:
        return None


def _redact_apikey(url: str) -> str:
    try:
        return re.sub(r"(apikey=)[^&]+", r"\1***", url, flags=re.IGNORECASE)
    except Exception:
        return url


@st.cache_data(show_spinner=False, ttl=1800)
def _http_get_json(url: str, headers_items: Tuple[Tuple[str, str], ...] = ()) -> Tuple[int, Any, str]:
    """HTTP GET helper with Streamlit cache. Returns (status_code, json_or_None, error_text_or_empty)."""
    try:
        headers = dict(headers_items) if headers_items else None
        r = requests.get(url, headers=headers, timeout=25)
        status = int(getattr(r, "status_code", 0) or 0)
        try:
            return status, r.json(), ""
        except Exception:
            txt = getattr(r, "text", "") or ""
            return status, None, txt[:2000]
    except Exception as e:
        return 0, None, str(e)


@st.cache_data(show_spinner=False, ttl=86400)
def _http_get_text(url: str, headers_items: Tuple[Tuple[str, str], ...] = ()) -> Tuple[int, str, str]:
    """HTTP GET that returns raw text (needed for XML filings)."""
    try:
        headers = dict(headers_items) if headers_items else None
        r = requests.get(url, headers=headers, timeout=25)
        status = int(getattr(r, "status_code", 0) or 0)
        return status, (getattr(r, "text", "") or ""), ""
    except Exception as e:
        return 0, "", str(e)


@st.cache_data(show_spinner=False, ttl=86400)
def _sec_ticker_to_cik_map(user_agent: str) -> Dict[str, int]:
    """
    SEC 'company_tickers.json' -> mapping {TICKER: cik_int}.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = (
        ("User-Agent", user_agent),
        ("Accept", "application/json"),
        ("Accept-Encoding", "gzip, deflate"),
    )
    status, data, err = _http_get_json(url, headers)
    if status != 200 or not isinstance(data, dict):
        return {}
    out: Dict[str, int] = {}
    for _, v in data.items():
        try:
            t = str(v.get("ticker", "")).upper().strip()
            cik = int(v.get("cik_str"))
            if t:
                out[t] = cik
        except Exception:
            continue
    return out


def _coerce_dt(x: Any) -> Optional[pd.Timestamp]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return None


def _norm_tx_label(raw_tx: Any, acquired_disposed: Any = None) -> str:
    t = str(raw_tx or "").strip().lower()
    ad = str(acquired_disposed or "").strip().upper()

    # Prefer explicit acquired/disposed
    if ad == "A":
        return "Buy"
    if ad == "D":
        return "Sell"

    # Fallback heuristics
    if any(k in t for k in ["buy", "purchase", "acquire"]):
        return "Buy"
    if any(k in t for k in ["sell", "sale", "dispose"]):
        return "Sell"
    if t in {"p", "p - purchase"}:
        return "Buy"
    if t in {"s", "s - sale"}:
        return "Sell"
    return "Other"


def _df_from_records(records: List[Dict[str, Any]], source: str) -> pd.DataFrame:
    """Normalize disparate insider-trade payloads into a common dataframe."""

    # OPRAVA: _to_float definována mimo smyčku (dříve se předefinovávala při každé iteraci)
    def _to_float(x):
        try:
            if x is None:
                return None
            s = str(x).strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return None
            # remove commas
            s = s.replace(",", "")
            return float(s)
        except Exception:
            return None

    rows: List[Dict[str, Any]] = []
    for it in records or []:
        if not isinstance(it, dict):
            continue

        # Common-ish fields across providers
        date_raw = (
            it.get("transactionDate")
            or it.get("transaction_date")
            or it.get("filingDate")
            or it.get("filing_date")
            or it.get("acceptedDate")
            or it.get("date")
        )

        # "transactionType" might already be human readable ("Purchase"/"Sale") or a code
        tx_raw = (
            it.get("transactionType")
            or it.get("transaction_type")
            or it.get("transaction_name")
            or it.get("transactionName")
            or it.get("type")
            or it.get("transactionCode")
            or it.get("transaction_code")
        )
        ad = (
            it.get("acquisitionOrDisposition")
            or it.get("transactionAcquiredDisposedCode")
            or it.get("acquiredDisposedCode")
        )

        owner = (
            it.get("insider_name")
            or it.get("insiderName")
            or it.get("name")
            or it.get("reportingName")
            or it.get("reporting_name")
            or it.get("reportingOwner")
            or it.get("reportingOwnerName")
            or it.get("reporting_owner_name")
        )

        position = (
            it.get("insider_position")
            or it.get("insiderPosition")
            or it.get("insider_title")
            or it.get("reportingTitle")
            or it.get("ownerTitle")
            or it.get("typeOfOwner")
            or it.get("role")
        )

        code = it.get("transactionCode") or it.get("transaction_code") or it.get("transaction_code") or it.get("transactionCode")
        security = it.get("securityTitle") or it.get("security") or it.get("security_title") or it.get("securityTitleValue")

        # Quantities / prices / values
        shares = (
            it.get("securitiesTransacted")
            or it.get("securities_transacted")
            or it.get("transactionShares")
            or it.get("shares")
            or it.get("share")
        )
        price = (
            it.get("price")
            or it.get("transactionPrice")
            or it.get("transactionPricePerShare")
            or it.get("transaction_price")
            or it.get("transaction_price_per_share")
        )
        value = (
            it.get("transactionValue")
            or it.get("transaction_value")
            or it.get("value")
            or it.get("totalValue")
            or it.get("amount")
        )

        filing_url = (
            it.get("sec_filing_url")
            or it.get("secFilingUrl")
            or it.get("filingURL")
            or it.get("filingUrl")
            or it.get("url")
        )

        # Preserve raw fields / notes for better filtering (e.g., 10b5-1 planned trades)
        notes_raw = (
            it.get("remarks")
            or it.get("remark")
            or it.get("notes")
            or it.get("note")
            or it.get("transactionDescription")
            or it.get("transaction_description")
            or it.get("transactionDesc")
            or it.get("description")
            or it.get("comment")
            or it.get("tradingPlan")
            or it.get("trading_plan")
            or it.get("plan")
        )

        # Parse numerics robustly
        shares_f = _to_float(shares)
        price_f = _to_float(price)
        value_f = _to_float(value)

        if value_f is None and shares_f is not None and price_f is not None:
            value_f = shares_f * price_f

        dtv = _coerce_dt(date_raw)
        if dtv is None or pd.isna(dtv):
            continue

        rows.append({
            "Date": dtv.date(),
            "Transaction": _norm_tx_label(tx_raw, ad),
            "TransactionRaw": (str(tx_raw) if tx_raw is not None else None),
            "Notes": (str(notes_raw) if notes_raw is not None else None),
            "Position": position or "—",
            "Owner": owner,
            "Security": security,
            "Code": code,
            "Shares": shares_f,
            "Price": price_f,
            "Value": value_f,
            "Source": source,
            "FilingURL": filing_url,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date", ascending=False)
    return df


def _dedupe_insider_df(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate merged insider transactions across providers.

    Different providers may format the same trade slightly differently (whitespace, casing,
    numeric types). This normalizes key fields, dedupes, and aggregates Source so you keep
    provenance without double-counting.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    d = df.copy()

    # Ensure expected columns exist
    for col in ["Date", "Owner", "Code", "Shares", "Price", "Value", "Transaction", "TransactionRaw", "Notes", "Source", "Position", "Security", "FilingURL"]:
        if col not in d.columns:
            d[col] = None

    # Normalize date to date
    try:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.date
    except Exception:
        pass

    def _norm_text(x: Any) -> str:
        try:
            s = str(x) if x is not None else ""
            s = re.sub(r"\s+", " ", s).strip()
            return s
        except Exception:
            return ""

    d["_owner_n"] = d["Owner"].apply(_norm_text).str.upper()
    d["_code_n"] = d["Code"].apply(_norm_text).str.upper()
    d["_tx_n"] = d["Transaction"].apply(_norm_text).str.lower()

    # Numeric normalization
    def _to_num(x: Any) -> Optional[float]:
        try:
            v = safe_float(x)
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return float(v)
        except Exception:
            return None

    d["_shares_n"] = d["Shares"].apply(_to_num)
    d["_price_n"] = d["Price"].apply(_to_num)
    d["_value_n"] = d["Value"].apply(_to_num)

    # Round to stabilize floating differences
    d["_shares_r"] = d["_shares_n"].round(0)
    d["_price_r"] = d["_price_n"].round(4)
    d["_value_r"] = d["_value_n"].round(2)

    # Build dedupe key
    d["_key"] = (
        d["Date"].astype(str)
        + "|"
        + d["_owner_n"]
        + "|"
        + d["_code_n"]
        + "|"
        + d["_tx_n"]
        + "|"
        + d["_shares_r"].astype(str)
        + "|"
        + d["_price_r"].astype(str)
    )

    def _join_sources(s: pd.Series) -> str:
        vals = []
        for x in s.dropna().tolist():
            sx = _norm_text(x)
            if sx:
                vals.append(sx)
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        return ", ".join(uniq) if uniq else ""

    def _join_text(s: pd.Series) -> str:
        vals = []
        for x in s.dropna().tolist():
            sx = _norm_text(x)
            if sx:
                vals.append(sx)
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        return " | ".join(uniq) if uniq else ""

    # Aggregate: keep first non-null for most fields, but join sources
    agg = {
        "Date": "first",
        "Owner": "first",
        "Position": "first",
        "Code": "first",
        "Security": "first",
        "Shares": "first",
        "Price": "first",
        "Value": "first",
        "Transaction": "first",
        "TransactionRaw": _join_text,
        "Notes": _join_text,
        "FilingURL": "first",
        "Source": _join_sources,
    }

    d2 = d.groupby("_key", dropna=False, as_index=False).agg(agg)

    # Sort
    try:
        d2 = d2.sort_values("Date", ascending=False)
    except Exception:
        pass
    return d2


def _parse_fmp_company_outlook(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    # FMP legacy doc mentions "insideTrades" but this can vary.
    inside = (
        payload.get("insideTrades")
        or payload.get("insiderTrades")
        or payload.get("insiderTrading")
        or payload.get("insiderTrade")
    )
    if isinstance(inside, dict):
        # some variants might nest list deeper
        inside = inside.get("data") or inside.get("items") or inside.get("results")
    if not isinstance(inside, list):
        return pd.DataFrame()
    return _df_from_records(inside, source="FMP legacy: company-outlook")


def _parse_fmp_stable(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return _df_from_records(payload, source="FMP stable: insider-trading/search")
    if isinstance(payload, dict):
        # some endpoints wrap results
        inner = payload.get("data") or payload.get("items") or payload.get("results") or payload.get("insiderTrades")
        if isinstance(inner, list):
            return _df_from_records(inner, source="FMP stable: insider-trading/search")
    return pd.DataFrame()



def _extract_api_error(payload: Any, err_text: str = "") -> Optional[str]:
    """Normalize error text across providers."""
    if err_text:
        return str(err_text)[:500]
    if isinstance(payload, dict):
        for k in ("Error Message", "error", "Error", "message", "Information", "Note", "detail"):
            v = payload.get(k)
            if v:
                return str(v)[:500]
    return None


def _payload_to_records(payload: Any) -> Optional[List[Dict[str, Any]]]:
    """Try to find a list[dict] of transactions inside various payload shapes."""
    if isinstance(payload, list) and (not payload or isinstance(payload[0], dict)):
        return payload
    if isinstance(payload, dict):
        # common keys
        for k in ("data", "items", "results", "insiderTrades", "insideTrades", "insiderTransactions", "transactions"):
            v = payload.get(k)
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
        # fallback: first list of dicts found
        for v in payload.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
    return None


@st.cache_data(show_spinner=False, ttl=43200)
def _fetch_insider_from_alpha_vantage(ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Free-ish alternative: Alpha Vantage Insider Transactions.
    Docs: function=INSIDER_TRANSACTIONS&symbol=...&apikey=...
    """
    meta: Dict[str, Any] = {
        "provider": "AlphaVantage",
        "endpoint": "query?function=INSIDER_TRANSACTIONS",
        "ticker": ticker,
        "status": None,
        "items": 0,
        "note": None,
        "error": None,
        "url": None,
    }
    if not ALPHAVANTAGE_API_KEY:
        meta["note"] = "ALPHAVANTAGE_API_KEY není nastaven."
        return pd.DataFrame(), meta

    url = f"https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    meta["url"] = _redact_apikey(url)
    status, payload, err = _http_get_json(url)
    meta["status"] = status

    # AlphaVantage often returns "Note"/"Information" on rate limit
    meta["error"] = _extract_api_error(payload, err)
    if status != 200 or not payload:
        return pd.DataFrame(), meta

    recs = _payload_to_records(payload)
    if not recs:
        # some responses put everything in a single dict; keep debug info
        return pd.DataFrame(), meta

    df = _df_from_records(recs, source="Alpha Vantage: INSIDER_TRANSACTIONS")
    meta["items"] = int(len(df)) if df is not None else 0
    return df, meta


@st.cache_data(show_spinner=False, ttl=43200)
def _fetch_insider_from_finnhub(ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Alternative: Finnhub insider transactions.
    Endpoint: /stock/insider-transactions?symbol=...&token=...
    """
    meta: Dict[str, Any] = {
        "provider": "Finnhub",
        "endpoint": "api/v1/stock/insider-transactions",
        "ticker": ticker,
        "status": None,
        "items": 0,
        "note": None,
        "error": None,
        "url": None,
    }
    if not FINNHUB_API_KEY:
        meta["note"] = "FINNHUB_API_KEY není nastaven."
        return pd.DataFrame(), meta

    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={FINNHUB_API_KEY}"
    meta["url"] = re.sub(r"(token=)[^&]+", r"\1***", url, flags=re.IGNORECASE)
    status, payload, err = _http_get_json(url)
    meta["status"] = status
    meta["error"] = _extract_api_error(payload, err)
    if status != 200 or not payload:
        return pd.DataFrame(), meta

    recs = _payload_to_records(payload)
    if not recs:
        # Finnhub usually returns {"data":[...], ...}
        return pd.DataFrame(), meta

    df = _df_from_records(recs, source="Finnhub: insider-transactions")
    meta["items"] = int(len(df)) if df is not None else 0
    return df, meta



@st.cache_data(show_spinner=False, ttl=43200)
def _fetch_insider_from_api_ninjas(ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Alternative: API Ninjas insider transactions.
    Endpoint: https://api.api-ninjas.com/v1/insidertransactions?ticker=...
    Auth: X-Api-Key header.
    """
    meta: Dict[str, Any] = {
        "provider": "APINinjas",
        "endpoint": "v1/insidertransactions",
        "ticker": ticker,
        "status": None,
        "items": 0,
        "note": None,
        "error": None,
        "url": f"https://api.api-ninjas.com/v1/insidertransactions?ticker={ticker}",
    }
    if not NINJAS_API_KEY:
        meta["note"] = "NINJAS_API_KEY (nebo Ninjas_API_KEY) není nastaven."
        return pd.DataFrame(), meta

    headers = (("X-Api-Key", NINJAS_API_KEY), ("Accept", "application/json"))
    status, payload, err = _http_get_json(meta["url"], headers)
    meta["status"] = status
    meta["error"] = _extract_api_error(payload, err)

    if status != 200 or payload is None:
        return pd.DataFrame(), meta

    # API Ninjas returns a JSON array
    recs: Optional[List[Dict[str, Any]]]
    if isinstance(payload, list):
        recs = payload
    else:
        recs = _payload_to_records(payload)

    if not recs:
        return pd.DataFrame(), meta

    df = _df_from_records(recs, source="API Ninjas: insidertransactions")
    meta["items"] = int(len(df)) if df is not None else 0
    return df, meta

def _sec_pick_xml_from_index(index_payload: Any) -> Optional[str]:
    """Pick a likely Form 4 XML filename from SEC index.json listing."""
    if not isinstance(index_payload, dict):
        return None
    items = ((index_payload.get("directory") or {}).get("item") or [])
    names = []
    for it in items:
        try:
            nm = str(it.get("name", "")).strip()
            if nm:
                names.append(nm)
        except Exception:
            continue
    if not names:
        return None

    def score(n: str) -> Tuple[int, int]:
        nl = n.lower()
        s = 0
        if nl.endswith(".xml"):
            s += 100
        if "xsl" in nl:
            s -= 50
        if "form4" in nl or "f345" in nl:
            s += 30
        if "primary" in nl:
            s += 15
        # shorter names often better
        return (-s, len(nl))

    xmls = [n for n in names if n.lower().endswith(".xml")]
    if not xmls:
        return None
    return sorted(xmls, key=score)[0]


@st.cache_data(show_spinner=False, ttl=43200)
def _fetch_insider_from_sec(ticker: str, max_filings: int = 12, max_transactions: int = 250) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Free fallback: SEC EDGAR Form 4 parsing via:
    - company_tickers.json (ticker->CIK)
    - submissions CIK##########.json (recent filings)
    - index.json per filing directory to locate the real XML
    """
    meta: Dict[str, Any] = {"provider": "SEC", "ticker": ticker, "cik": None, "status": None, "items": 0, "note": None}

    ua = (SEC_USER_AGENT or "").strip()
    if not ua or "your_email" in ua:
        meta["note"] = "SEC_USER_AGENT není nastaven (doporučeno)."

    # Build ticker->CIK map
    cik_map = _sec_ticker_to_cik_map(ua or "StockPickerPro/1.0")
    cik_int = cik_map.get(ticker.upper())
    if not cik_int:
        meta["note"] = (meta["note"] or "") + " Ticker nenalezen v SEC mappingu."
        return pd.DataFrame(), meta

    meta["cik"] = cik_int
    cik_padded = str(cik_int).zfill(10)

    subs_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers_json = (
        ("User-Agent", ua or "StockPickerPro/1.0"),
        ("Accept", "application/json"),
        ("Accept-Encoding", "gzip, deflate"),
    )
    status, subs, err = _http_get_json(subs_url, headers_json)
    meta["status"] = status
    if status != 200 or not isinstance(subs, dict):
        meta["note"] = (meta["note"] or "") + f" Submissions error: {str(err)[:200]}"
        return pd.DataFrame(), meta

    recent = ((subs.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accs = recent.get("accessionNumber") or []
    fdates = recent.get("filingDate") or []

    # collect recent Form 4/4A
    idxs = [i for i, f in enumerate(forms) if str(f).startswith("4")]
    idxs = idxs[:max_filings]
    if not idxs:
        meta["note"] = (meta["note"] or "") + " Žádné Form 4 v recent submissions."
        return pd.DataFrame(), meta

    import xml.etree.ElementTree as ET

    rows: List[Dict[str, Any]] = []
    headers_xml = (
        ("User-Agent", ua or "StockPickerPro/1.0"),
        ("Accept", "application/xml,text/xml,text/plain,*/*"),
        ("Accept-Encoding", "gzip, deflate"),
    )
    # ── SEC debug countery ──────────────────────────────────────────────────
    _dbg_filings_tried  = 0
    _dbg_xml_downloaded = 0
    _dbg_xml_parsed_ok  = 0
    _dbg_tx_found       = 0
    _dbg_index_errors   = 0

    for i in idxs:
        try:
            accession = str(accs[i])
            accession_nodash = accession.replace("-", "")
            filing_date = fdates[i] if i < len(fdates) else None
            _dbg_filings_tried += 1

            # Use index.json to locate XML
            index_url = f"https://data.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
            st_i, index_payload, err_i = _http_get_json(index_url, headers_json)
            if st_i != 200 or not isinstance(index_payload, dict):
                _dbg_index_errors += 1
                continue

            xml_name = _sec_pick_xml_from_index(index_payload)
            if not xml_name:
                _dbg_index_errors += 1
                continue

            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{xml_name}"
            st_x, xml_text, err_x = _http_get_text(filing_url, headers_xml)
            if st_x != 200 or not xml_text:
                continue
            _dbg_xml_downloaded += 1

            # parse XML
            if "<ownershipDocument" not in xml_text and "<nonDerivativeTransaction" not in xml_text:
                if "<" not in xml_text[:50]:
                    continue

            root = ET.fromstring(xml_text.encode("utf-8", errors="ignore"))
            _dbg_xml_parsed_ok += 1

            owner = None
            officer_title = None
            try:
                owner = root.findtext(".//{*}reportingOwnerId/{*}rptOwnerName")
            except Exception:
                owner = None
            try:
                officer_title = root.findtext(".//{*}reportingOwnerRelationship/{*}officerTitle")
            except Exception:
                officer_title = None

            # Pull remarks/footnotes so we can filter non-informative (10b5-1, tax withholding, etc.)
            try:
                remarks_txt = root.findtext(".//{*}remarks") or ""
            except Exception:
                remarks_txt = ""
            footnotes_txt: List[str] = []
            try:
                for fn in root.findall(".//{*}footnote"):
                    try:
                        t = "".join(list(fn.itertext())).strip()
                        if t:
                            footnotes_txt.append(t)
                    except Exception:
                        continue
            except Exception:
                pass
            notes_blob = " ".join([remarks_txt] + footnotes_txt).strip() or None

            # non-derivative transactions
            for tx in root.findall(".//{*}nonDerivativeTransaction"):
                dt_val = tx.findtext(".//{*}transactionDate/{*}value") or filing_date
                code = tx.findtext(".//{*}transactionCoding/{*}transactionCode")
                ad = tx.findtext(".//{*}transactionAcquiredDisposedCode/{*}value")
                shares = tx.findtext(".//{*}transactionShares/{*}value")
                price = tx.findtext(".//{*}transactionPricePerShare/{*}value")
                sec_title = tx.findtext(".//{*}securityTitle/{*}value")

                try:
                    shares_f = float(shares) if shares else None
                except Exception:
                    shares_f = None
                try:
                    price_f = float(price) if price else None
                except Exception:
                    price_f = None

                val_f = (shares_f * price_f) if (shares_f is not None and price_f is not None) else None
                tx_label = _norm_tx_label(code, ad)

                dtp = _coerce_dt(dt_val)
                if dtp is None or pd.isna(dtp):
                    continue

                rows.append({
                    "Date": dtp.date(),
                    "Transaction": tx_label,
                    "TransactionRaw": (str(code) if code is not None else None),
                    "Notes": notes_blob,
                    "Position": officer_title or ("Director/Officer" if owner else "—"),
                    "Value": val_f,
                    "Shares": shares_f,
                    "Price": price_f,
                    "Owner": owner,
                    "Security": sec_title,
                    "Code": code,
                    "Source": "SEC Form 4",
                    "FilingURL": filing_url,
                })
                _dbg_tx_found += 1
                if len(rows) >= max_transactions:
                    break

            time.sleep(0.12)
            if len(rows) >= max_transactions:
                break
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date", ascending=False)
    meta["items"] = int(len(df))
    # Bohatší debug note pro případ 0 výsledků
    meta["note"] = (
        f"Filings zkuseno: {_dbg_filings_tried}/{len(idxs)} | "
        f"XML staženo: {_dbg_xml_downloaded} | "
        f"XML OK: {_dbg_xml_parsed_ok} | "
        f"Transakcí nalezeno: {_dbg_tx_found} | "
        f"Index chyby: {_dbg_index_errors}"
    )
    if _dbg_xml_downloaded == 0:
        meta["note"] += " ⚠️ Žádné XML nebylo staženo – zkontroluj SEC blok nebo User-Agent."
    elif _dbg_tx_found == 0:
        meta["note"] += " ℹ️ XML OK, ale žádné nonDerivativeTransaction – možná jen opce/granty."
    return df, meta


def fetch_insider_transactions_multi(ticker: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Multi-source insider fetch with rich debug.

    Sources (in priority order):
    1) FMP stable (often paid)  
    2) FMP legacy company-outlook (often blocked) 
    3) API Ninjas insidertransactions (free key) 
    4) Alpha Vantage INSIDER_TRANSACTIONS (free key) 
    5) Finnhub insider-transactions (token) 
    6) SEC EDGAR Form 4 parsing (free) 

    Returns a merged dataframe (deduplicated) if any source has data.
    """
    meta: Dict[str, Any] = {
        "ticker": ticker,
        "fmp_key_loaded": bool(FMP_API_KEY),
        "fmp_key_len": len(FMP_API_KEY) if FMP_API_KEY else 0,
        "av_key_loaded": bool(ALPHAVANTAGE_API_KEY),
        "av_key_len": len(ALPHAVANTAGE_API_KEY) if ALPHAVANTAGE_API_KEY else 0,
        "finnhub_key_loaded": bool(FINNHUB_API_KEY),
        "finnhub_key_len": len(FINNHUB_API_KEY) if FINNHUB_API_KEY else 0,
        "ninjas_key_loaded": bool(NINJAS_API_KEY),
        "ninjas_key_len": len(NINJAS_API_KEY) if NINJAS_API_KEY else 0,
        "chosen_source": None,
        "attempts": [],
    }

    def add_attempt(d: Dict[str, Any]) -> None:
        try:
            meta["attempts"].append(d)
        except Exception:
            pass

    dfs: List[pd.DataFrame] = []
    sources_used: List[str] = []

    # 1) FMP stable endpoint
    if FMP_API_KEY:
        url = f"https://financialmodelingprep.com/stable/insider-trading/search?symbol={ticker}&page=0&limit=100&apikey={FMP_API_KEY}"
        status, payload, err = _http_get_json(url)
        add_attempt({
            "provider": "FMP",
            "endpoint": "stable/insider-trading/search",
            "url": _redact_apikey(url),
            "status_code": status,
            "items": len(payload) if isinstance(payload, list) else None,
            "error": _extract_api_error(payload, err),
        })
        if status == 200:
            df = _parse_fmp_stable(payload)
            if df is not None and not df.empty:
                dfs.append(df)
                sources_used.append("FMP stable")

    # 2) FMP legacy company outlook
    if FMP_API_KEY:
        url = f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={ticker}&apikey={FMP_API_KEY}"
        status, payload, err = _http_get_json(url)
        add_attempt({
            "provider": "FMP",
            "endpoint": "api/v4/company-outlook",
            "url": _redact_apikey(url),
            "status_code": status,
            "items": None,
            "error": _extract_api_error(payload, err),
        })
        if status == 200:
            df = _parse_fmp_company_outlook(payload)
            if df is not None and not df.empty:
                dfs.append(df)
                sources_used.append("FMP legacy")

    # 3) API Ninjas
    df_nj, nj_meta = _fetch_insider_from_api_ninjas(ticker)
    add_attempt(nj_meta)
    if df_nj is not None and not df_nj.empty:
        dfs.append(df_nj)
        sources_used.append("API Ninjas")

    # 4) Alpha Vantage
    df_av, av_meta = _fetch_insider_from_alpha_vantage(ticker)
    add_attempt(av_meta)
    if df_av is not None and not df_av.empty:
        dfs.append(df_av)
        sources_used.append("Alpha Vantage")

    # 5) Finnhub
    df_fh, fh_meta = _fetch_insider_from_finnhub(ticker)
    add_attempt(fh_meta)
    if df_fh is not None and not df_fh.empty:
        dfs.append(df_fh)
        sources_used.append("Finnhub")

    # 6) SEC fallback
    df_sec, sec_meta = _fetch_insider_from_sec(ticker)
    add_attempt(sec_meta)
    if df_sec is not None and not df_sec.empty:
        dfs.append(df_sec)
        sources_used.append("SEC Form 4")

    if not dfs:
        return None, meta

    # Merge + dedupe (ignore Source so the same transaction coming from multiple providers doesn't duplicate)
    merged = pd.concat([d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty], ignore_index=True, sort=False)

    # Ensure expected columns exist
    for col in ["Owner", "Position", "Code", "Security", "Shares", "Price", "Value", "Source", "FilingURL", "Transaction", "TransactionRaw", "Notes", "Date"]:
        if col not in merged.columns:
            merged[col] = None

    # Make date comparable
    try:
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.date
    except Exception:
        pass

    # Robust dedupe across providers (also aggregates Source so you keep provenance)
    merged = _dedupe_insider_df(merged)


    if len(sources_used) == 1:
        meta["chosen_source"] = sources_used[0]
    else:
        meta["chosen_source"] = "Merged: " + ", ".join(sources_used)

    return merged, meta


def fetch_insider_transactions_fmp(ticker: str) -> Optional[pd.DataFrame]:
    """
    Backwards-compatible wrapper used throughout the app.
    Stores debug info into st.session_state["insider_debug"].
    """
    df, meta = fetch_insider_transactions_multi(ticker)
    try:
        st.session_state["insider_debug"] = meta
    except Exception:
        pass
    return df



@dataclass
class Metric:
    name: str
    value: Optional[float]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    target_below: Optional[float] = None
    target_above: Optional[float] = None
    weight: float = 1.0
    source: str = "yfinance"


def extract_metrics(info: Dict[str, Any], ticker: str) -> Dict[str, Metric]:
    """Extract comprehensive metrics from Yahoo Finance info."""
    
    # Price metrics
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    
    # Valuation
    pe = safe_float(info.get("trailingPE"))
    pb = safe_float(info.get("priceToBook"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    peg = safe_float(info.get("pegRatio"))
    ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
    
    # Profitability
    roe = safe_float(info.get("returnOnEquity"))
    roa = safe_float(info.get("returnOnAssets"))
    operating_margin = safe_float(info.get("operatingMargins"))
    profit_margin = safe_float(info.get("profitMargins"))
    gross_margin = safe_float(info.get("grossMargins"))
    
    # Growth – fallback na quarterly data když annual chybí
    revenue_growth = (safe_float(info.get("revenueGrowth"))
                      or safe_float(info.get("revenueQuarterlyGrowth")))
    earnings_growth = (safe_float(info.get("earningsGrowth"))
                       or safe_float(info.get("earningsQuarterlyGrowth")))
    earnings_quarterly_growth = safe_float(info.get("earningsQuarterlyGrowth"))

    # Derived PEG fallback: P/E ÷ (earnings_growth × 100) – když provider PEG nedá
    # Pozor: earnings_growth je v yfinance typicky ve formátu 0.10 (=10%), proto ×100.
    if peg is None and pe is not None and pe > 0 and earnings_growth is not None and earnings_growth > 0.005:
        _dpeg = pe / (earnings_growth * 100.0)
        peg = round(_dpeg, 2) if 0.01 < _dpeg < 10 else None

    
    # Financial health
    current_ratio = safe_float(info.get("currentRatio"))
    quick_ratio = safe_float(info.get("quickRatio"))
    # D/E normalizace: Yahoo Finance vrací D/E v ×100 formátu (napr. "50" = skutečně 0.50).
    # Rozumný reálný D/E rozsah: 0–10. Hodnoty 10–2000 vydělíme 100.
    _raw_de = safe_float(info.get("debtToEquity"))
    if _raw_de is not None and 10 < _raw_de < 2000:
        debt_to_equity = round(_raw_de / 100.0, 4)
    else:
        debt_to_equity = _raw_de
    total_cash = safe_float(info.get("totalCash"))
    total_debt = safe_float(info.get("totalDebt"))
    
    # Cash flow
    operating_cashflow = safe_float(info.get("operatingCashflow"))
    market_cap = safe_float(info.get('marketCap'))
    fcf, _fcf_dbg = get_fcf_ttm_yfinance(ticker, market_cap)
    fcf_yield = safe_div(fcf, market_cap) if fcf and market_cap else None
    
    # Analyst targets
    target_mean = safe_float(info.get("targetMeanPrice"))
    target_median = safe_float(info.get("targetMedianPrice"))
    target_high = safe_float(info.get("targetHighPrice"))
    target_low = safe_float(info.get("targetLowPrice"))
    recommendation = info.get("recommendationKey", "")
    
    # Dividend
    dividend_yield = safe_float(info.get("dividendYield"))
    payout_ratio = safe_float(info.get("payoutRatio"))
    
    metrics = {
        "price": Metric("Current Price", price),
        "pe": Metric("P/E Ratio", pe, target_below=25, weight=1.5),
        "pb": Metric("P/B Ratio", pb, target_below=3, weight=1.0),
        "ps": Metric("P/S Ratio", ps, target_below=2, weight=1.0),
        "peg": Metric("PEG Ratio", peg, target_below=1.5, weight=1.5),
        "ev_ebitda": Metric("EV/EBITDA", ev_ebitda, target_below=15, weight=1.0),
        "roe": Metric("ROE", roe, target_above=0.15, weight=2.0),
        "roa": Metric("ROA", roa, target_above=0.05, weight=1.0),
        "operating_margin": Metric("Operating Margin", operating_margin, target_above=0.15, weight=1.5),
        "profit_margin": Metric("Profit Margin", profit_margin, target_above=0.10, weight=1.5),
        "gross_margin": Metric("Gross Margin", gross_margin, target_above=0.30, weight=1.0),
        "revenue_growth": Metric("Revenue Growth", revenue_growth, target_above=0.10, weight=2.0),
        "earnings_growth": Metric("Earnings Growth", earnings_growth, target_above=0.10, weight=2.0),
        "current_ratio": Metric("Current Ratio", current_ratio, target_above=1.5, weight=1.0),
        "quick_ratio": Metric("Quick Ratio", quick_ratio, target_above=1.0, weight=0.8),
        "debt_to_equity": Metric("Debt/Equity", debt_to_equity, target_below=1.0, weight=1.5),
        "fcf_yield": Metric("FCF Yield", fcf_yield, target_above=0.05, weight=2.0),
        "dividend_yield": Metric("Dividend Yield", dividend_yield, target_above=0.02, weight=0.5),
        "payout_ratio": Metric("Payout Ratio", payout_ratio, target_below=0.70, weight=0.5),
        "target_mean": Metric("Analyst Target (Mean)", target_mean),
        "target_median": Metric("Analyst Target (Median)", target_median),
        "target_high": Metric("Analyst Target (High)", target_high),
        "target_low": Metric("Analyst Target (Low)", target_low),
    }
    
    return metrics


# ============================================================================
# DATA ENRICHMENT ENGINE (MULTI-SOURCE)
# ============================================================================

def _maybe_pct(val: Optional[float]) -> Optional[float]:
    """Normalize percent-like values into 0-1 when providers return 0-100."""
    v = safe_float(val)
    if v is None:
        return None
    # If it looks like 15 (=15%), normalize.
    if abs(v) > 1.5 and abs(v) <= 100.0:
        return v / 100.0
    return v

def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            v = safe_float(d.get(k))
            if v is not None:
                return v
    return None

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_fmp_ratios_ttm(ticker: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """FMP stable Ratios TTM.

    Docs (stable): /stable/ratios-ttm?symbol=...
    """
    meta = {"provider": "FMP", "endpoint": "stable/ratios-ttm", "status": None, "error": None, "url": None}
    if not FMP_API_KEY:
        meta["error"] = "FMP_API_KEY není nastaven."
        return None, meta

    url = f"https://financialmodelingprep.com/stable/ratios-ttm?symbol={ticker}&apikey={FMP_API_KEY}"
    meta["url"] = _redact_apikey(url)

    status, payload, err = _http_get_json(url)
    meta["status"] = status
    meta["error"] = _extract_api_error(payload, err)

    if status != 200 or payload is None:
        return None, meta

    # Stable endpoints usually return a list (often length=1). Be flexible.
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0], meta

    if isinstance(payload, dict):
        recs = payload.get("data") or payload.get("ratios") or payload.get("results")
        if isinstance(recs, list) and recs and isinstance(recs[0], dict):
            return recs[0], meta
        # Some variants may directly return a dict of ratios
        if payload and any(isinstance(v, (int, float, str)) for v in payload.values()):
            return payload, meta

    return None, meta

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_fmp_key_metrics_ttm(ticker: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """FMP stable Key Metrics TTM.

    Docs (stable): /stable/key-metrics-ttm?symbol=...
    """
    meta = {"provider": "FMP", "endpoint": "stable/key-metrics-ttm", "status": None, "error": None, "url": None}
    if not FMP_API_KEY:
        meta["error"] = "FMP_API_KEY není nastaven."
        return None, meta

    url = f"https://financialmodelingprep.com/stable/key-metrics-ttm?symbol={ticker}&apikey={FMP_API_KEY}"
    meta["url"] = _redact_apikey(url)

    status, payload, err = _http_get_json(url)
    meta["status"] = status
    meta["error"] = _extract_api_error(payload, err)

    if status != 200 or payload is None:
        return None, meta

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0], meta

    if isinstance(payload, dict):
        recs = payload.get("data") or payload.get("metrics") or payload.get("results")
        if isinstance(recs, list) and recs and isinstance(recs[0], dict):
            return recs[0], meta
        if payload and any(isinstance(v, (int, float, str)) for v in payload.values()):
            return payload, meta

    return None, meta

@st.cache_data(show_spinner=False, ttl=21600)  # 6 hodin – rate limity AV
def _fetch_alpha_overview(ticker: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta = {"provider": "AlphaVantage", "endpoint": "query?function=OVERVIEW", "status": None, "error": None, "url": None}
    if not ALPHAVANTAGE_API_KEY:
        meta["error"] = "ALPHAVANTAGE_API_KEY není nastaven."
        return None, meta
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    meta["url"] = _redact_apikey(url)
    status, payload, err = _http_get_json(url)
    meta["status"] = status
    meta["error"] = _extract_api_error(payload, err)
    if status != 200 or not isinstance(payload, dict) or not payload:
        return None, meta
    # AlphaVantage sometimes returns empty dict or an error note
    if any(k in payload for k in ("Note", "Information", "Error Message")):
        meta["error"] = _extract_api_error(payload, meta["error"] or "")
        return None, meta
    return payload, meta

@st.cache_data(show_spinner=False, ttl=21600)  # 6 hodin – rate limity Finnhub
def _fetch_finnhub_metric(ticker: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta = {"provider": "Finnhub", "endpoint": "api/v1/stock/metric?metric=all", "status": None, "error": None, "url": None}
    if not FINNHUB_API_KEY:
        meta["error"] = "FINNHUB_API_KEY není nastaven."
        return None, meta
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_API_KEY}"
    meta["url"] = re.sub(r"(token=)[^&]+", r"\1***", url, flags=re.IGNORECASE)
    status, payload, err = _http_get_json(url)
    meta["status"] = status
    meta["error"] = _extract_api_error(payload, err)
    if status != 200 or not isinstance(payload, dict):
        return None, meta
    metric = payload.get("metric")
    if isinstance(metric, dict) and metric:
        return metric, meta
    return None, meta

def enrich_metrics_multisource(ticker: str, metrics: Dict[str, Metric], info: Dict[str, Any]) -> Tuple[Dict[str, Metric], Dict[str, Any]]:
    """Enrich core fundamental metrics with robust fallback chain.

    Fallback chain per requested spec:
      1) yfinance (already in `metrics`)
      2) FMP ratios-ttm + key-metrics-ttm
      3) Alpha Vantage OVERVIEW
      4) Finnhub metric

    Only fills missing metrics and sets Metric.source to the provider used.
    """
    debug: Dict[str, Any] = {"ticker": ticker, "fills": {}, "steps": []}

    def _set(key: str, val: Optional[float], src: str, pct: bool = False) -> None:
        if key not in metrics:
            return
        if val is None:
            return
        v = _maybe_pct(val) if pct else safe_float(val)
        if v is None:
            return
        # D/E normalizace i pro hodnoty z externích providerů (×100 formát)
        if key == "debt_to_equity" and v is not None and 10 < v < 2000:
            v = round(v / 100.0, 4)
        metrics[key].value = v
        metrics[key].source = src
        debug["fills"][key] = src

    def _is_missing(key: str) -> bool:
        m = metrics.get(key)
        if not m:
            return True
        v = safe_float(m.value)
        if v is None:
            return True
        # For valuation/ratios, treat non-positive as missing
        if key in {"pe", "peg", "pb", "ps", "ev_ebitda", "current_ratio", "quick_ratio", "debt_to_equity"} and v <= 0:
            return True
        return False

    wanted = ["pe", "peg", "pb", "ps", "ev_ebitda", "debt_to_equity", "operating_margin", "profit_margin", "gross_margin", "roe", "current_ratio", "quick_ratio", "fcf_yield", "revenue_growth", "earnings_growth"]
    if not any(_is_missing(k) for k in wanted):
        debug["steps"].append("yfinance ok (no enrichment needed)")
        return metrics, debug

    # --- Step 2: FMP TTM ---
    fmp_ratios, fmp_ratios_meta = (None, {})
    fmp_km, fmp_km_meta = (None, {})
    if FMP_API_KEY:
        fmp_ratios, fmp_ratios_meta = _fetch_fmp_ratios_ttm(ticker)
        fmp_km, fmp_km_meta = _fetch_fmp_key_metrics_ttm(ticker)
        debug["steps"].append({"FMP_ratios_ttm": fmp_ratios_meta})
        debug["steps"].append({"FMP_key_metrics_ttm": fmp_km_meta})

        merged_fmp: Dict[str, Any] = {}
        if isinstance(fmp_ratios, dict):
            merged_fmp.update(fmp_ratios)
        if isinstance(fmp_km, dict):
            merged_fmp.update(fmp_km)

        if merged_fmp:
            if _is_missing("pe"):
                _set("pe", _first_present(merged_fmp, ["peRatioTTM", "priceEarningsRatioTTM", "peTTM"]), "FMP")
            if _is_missing("peg"):
                _set("peg", _first_present(merged_fmp, ["pegRatioTTM", "pegTTM"]), "FMP")
            if _is_missing("debt_to_equity"):
                _set("debt_to_equity", _first_present(merged_fmp, ["debtEquityRatioTTM", "debtToEquityTTM", "debtToEquity"]), "FMP")
            if _is_missing("operating_margin"):
                _set("operating_margin", _first_present(merged_fmp, ["operatingProfitMarginTTM", "operatingMarginTTM", "operatingMarginsTTM"]), "FMP", pct=True)
            if _is_missing("profit_margin"):
                _set("profit_margin", _first_present(merged_fmp, ["netProfitMarginTTM", "profitMarginTTM", "profitMarginsTTM"]), "FMP", pct=True)
            if _is_missing("gross_margin"):
                _set("gross_margin", _first_present(merged_fmp, ["grossProfitMarginTTM", "grossMarginTTM", "grossMarginsTTM"]), "FMP", pct=True)
            if _is_missing("roe"):
                _set("roe", _first_present(merged_fmp, ["returnOnEquityTTM", "roeTTM", "returnOnEquity"]), "FMP", pct=True)
            if _is_missing("pb"):
                _set("pb", _first_present(merged_fmp, ["priceToBookRatioTTM", "pbRatioTTM", "pbTTM"]), "FMP")
            if _is_missing("ps"):
                _set("ps", _first_present(merged_fmp, ["priceToSalesRatioTTM", "psRatioTTM", "psTTM"]), "FMP")
            if _is_missing("ev_ebitda"):
                _set("ev_ebitda", _first_present(merged_fmp, ["enterpriseValueOverEBITDATTM", "evToEbitdaTTM", "evEbitdaTTM"]), "FMP")
            if _is_missing("current_ratio"):
                _set("current_ratio", _first_present(merged_fmp, ["currentRatioTTM", "currentRatio"]), "FMP")
            if _is_missing("quick_ratio"):
                _set("quick_ratio", _first_present(merged_fmp, ["quickRatioTTM", "quickRatio"]), "FMP")
            if _is_missing("fcf_yield"):
                _set("fcf_yield", _first_present(merged_fmp, ["freeCashFlowYieldTTM", "fcfYieldTTM", "freeCashFlowYield"]), "FMP", pct=True)

    # --- Step 3: Alpha Vantage OVERVIEW ---
    if any(_is_missing(k) for k in wanted) and ALPHAVANTAGE_API_KEY:
        av, av_meta = _fetch_alpha_overview(ticker)
        debug["steps"].append({"AlphaVantage_overview": av_meta})
        if isinstance(av, dict) and av:
            if _is_missing("pe"):
                _set("pe", _first_present(av, ["PERatio", "TrailingPE", "TrailingPERatio", "peTTM"]), "AlphaVantage")
            if _is_missing("peg"):
                _set("peg", _first_present(av, ["PEGRatio", "PegRatio", "pegTTM"]), "AlphaVantage")
            if _is_missing("operating_margin"):
                _set("operating_margin", safe_float(av.get("OperatingMarginTTM")), "AlphaVantage", pct=True)
            if _is_missing("profit_margin"):
                _set("profit_margin", safe_float(av.get("ProfitMargin")), "AlphaVantage", pct=True)
            if _is_missing("roe"):
                _set("roe", safe_float(av.get("ReturnOnEquityTTM")), "AlphaVantage", pct=True)
            if _is_missing("gross_margin"):
                gp = safe_float(av.get("GrossProfitTTM"))
                rev = safe_float(av.get("RevenueTTM"))
                if gp is not None and rev not in (None, 0):
                    _set("gross_margin", gp / rev, "AlphaVantage", pct=True)

            if _is_missing("pb"):
                _set("pb", _first_present(av, ["PriceToBookRatio", "PriceToBook"]), "AlphaVantage")
            if _is_missing("ps"):
                _set("ps", _first_present(av, ["PriceToSalesRatioTTM", "PriceToSalesRatio"]), "AlphaVantage")
            if _is_missing("ev_ebitda"):
                _set("ev_ebitda", _first_present(av, ["EVToEBITDA", "EVToEBITDAttm"]), "AlphaVantage")
            if _is_missing("current_ratio"):
                _set("current_ratio", _first_present(av, ["CurrentRatio"]), "AlphaVantage")
            if _is_missing("quick_ratio"):
                _set("quick_ratio", _first_present(av, ["QuickRatio"]), "AlphaVantage")
            if _is_missing("revenue_growth"):
                _set("revenue_growth", _first_present(av, ["QuarterlyRevenueGrowthYOY"]), "AlphaVantage", pct=True)
            if _is_missing("earnings_growth"):
                _set("earnings_growth", _first_present(av, ["QuarterlyEarningsGrowthYOY"]), "AlphaVantage", pct=True)
            if _is_missing("fcf_yield"):
                fcf = _first_present(av, ["FreeCashFlowTTM", "FCF", "freeCashFlowTTM"])
                mc = safe_float(av.get("MarketCapitalization"))
                if fcf is not None and mc not in (None, 0):
                    _set("fcf_yield", fcf / mc, "AlphaVantage", pct=True)

            if _is_missing("debt_to_equity"):
                # AlphaVantage sometimes provides DebtToEquity or TotalDebt/TotalEquity
                dte = safe_float(av.get("DebtToEquity"))
                if dte is None:
                    td = safe_float(av.get("TotalDebt"))
                    te = safe_float(av.get("TotalShareholderEquity"))
                    if td is not None and te not in (None, 0):
                        dte = td / te
                _set("debt_to_equity", dte, "AlphaVantage")

    # --- Step 4: Finnhub metric ---
    if any(_is_missing(k) for k in wanted) and FINNHUB_API_KEY:
        fh, fh_meta = _fetch_finnhub_metric(ticker)
        debug["steps"].append({"Finnhub_metric": fh_meta})
        if isinstance(fh, dict) and fh:
            if _is_missing("pe"):
                _set("pe", _first_present(fh, ["peTTM", "peAnnual", "peExclExtraTTM"]), "Finnhub")
            if _is_missing("peg"):
                _set("peg", _first_present(fh, ["pegTTM", "pegAnnual"]), "Finnhub")
            if _is_missing("roe"):
                _set("roe", _first_present(fh, ["roeTTM", "roeAnnual"]), "Finnhub", pct=True)
            if _is_missing("operating_margin"):
                _set("operating_margin", _first_present(fh, ["operatingMarginTTM", "operatingMarginAnnual"]), "Finnhub", pct=True)
            if _is_missing("profit_margin"):
                _set("profit_margin", _first_present(fh, ["netMarginTTM", "netMarginAnnual", "profitMarginTTM"]), "Finnhub", pct=True)
            if _is_missing("gross_margin"):
                _set("gross_margin", _first_present(fh, ["grossMarginTTM", "grossMarginAnnual"]), "Finnhub", pct=True)
            if _is_missing("debt_to_equity"):
                _set("debt_to_equity", _first_present(fh, ["totalDebtToEquityTTM", "totalDebt/totalEquityTTM", "totalDebt/totalEquityAnnual", "totalDebtToEquityAnnual"]), "Finnhub")
            if _is_missing("pb"):
                _set("pb", _first_present(fh, ["pbAnnual", "pbTTM", "priceToBookAnnual", "priceToBookTTM"]), "Finnhub")
            if _is_missing("ps"):
                _set("ps", _first_present(fh, ["psAnnual", "psTTM", "priceToSalesAnnual", "priceToSalesTTM"]), "Finnhub")
            if _is_missing("ev_ebitda"):
                _set("ev_ebitda", _first_present(fh, ["evToEbitdaTTM", "evToEbitdaAnnual"]), "Finnhub")
            if _is_missing("current_ratio"):
                _set("current_ratio", _first_present(fh, ["currentRatioAnnual", "currentRatioTTM"]), "Finnhub")
            if _is_missing("quick_ratio"):
                _set("quick_ratio", _first_present(fh, ["quickRatioAnnual", "quickRatioTTM"]), "Finnhub")
            if _is_missing("fcf_yield"):
                _set("fcf_yield", _first_present(fh, ["freeCashFlowYieldTTM", "freeCashFlowYieldAnnual", "fcfYieldTTM"]), "Finnhub", pct=True)
            if _is_missing("revenue_growth"):
                _set("revenue_growth", _first_present(fh, ["revenueGrowthTTM", "revenueGrowth5Y"]), "Finnhub", pct=True)
            if _is_missing("earnings_growth"):
                _set("earnings_growth", _first_present(fh, ["epsGrowthTTM", "epsGrowth5Y"]), "Finnhub", pct=True)

    # If still missing, keep as None; UI will show —
    missing_left = [k for k in wanted if _is_missing(k)]
    if missing_left:
        debug["steps"].append({"missing_after_fallbacks": missing_left})

    # ── Derived PEG jako poslední záchrana ────────────────────────────────────
    # Pokud PEG stále chybí, dopočítáme z P/E a earnings growth (i po enrichmentu)
    if _is_missing("peg"):
        _pe  = safe_float(metrics.get("pe").value if metrics.get("pe") else None)
        _eg  = safe_float(metrics.get("earnings_growth").value if metrics.get("earnings_growth") else None)
        if _pe is not None and _pe > 0 and _eg is not None and _eg > 0.005:
            _dpeg = _pe / (_eg * 100.0)
            if 0.01 < _dpeg < 10:
                metrics["peg"].value  = round(_dpeg, 2)
                metrics["peg"].source = "Derived (P/E ÷ EPS Growth%)"
                debug["fills"]["peg"] = "Derived"
                debug["steps"].append("PEG dopočítán z P/E a EPS growth")

    return metrics, debug


def calculate_metric_score(metric: Metric) -> float:
    """Calculate 0-10 score for a single metric."""
    if metric.value is None:
        return 3.0  # Chybějící data = mírně negativní (dříve 5.0 uměle nafukovalo skóre)
    
    val = metric.value
    
    # Target below (lower is better)
    if metric.target_below is not None:
        if val <= metric.target_below * 0.7:
            return 10.0
        elif val <= metric.target_below:
            return 8.0
        elif val <= metric.target_below * 1.5:
            return 5.0
        else:
            return 2.0
    
    # Target above (higher is better)
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


def build_scorecard_advanced(metrics: Dict[str, Metric], info: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Build advanced scorecard (0-100) with category breakdown.
    Returns: (total_score, category_scores, individual_metric_scores)
    """
    
    # Category definitions
    categories = {
        "Valuace": ["pe", "pb", "ps", "peg", "ev_ebitda"],
        "Kvalita": ["roe", "roa", "operating_margin", "profit_margin", "gross_margin"],
        "Růst": ["revenue_growth", "earnings_growth"],
        "Fin. zdraví": ["current_ratio", "quick_ratio", "debt_to_equity", "fcf_yield"],
    }
    
    category_scores = {}
    individual_scores = {}
    
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
            category_scores[cat_name] = (weighted_sum / total_weight) * 10  # Scale to 0-100
        else:
            category_scores[cat_name] = 50.0
    
    # Overall score (equal weight per category)
    total_score = sum(category_scores.values()) / len(category_scores)
    
    return total_score, category_scores, individual_scores


# ============================================================================
# DCF VALUATION
# ============================================================================





def reverse_dcf_implied_growth(
    current_price: float,
    fcf: float,
    terminal_growth: float = 0.03,
    wacc: float = 0.10,
    years: int = 5,
    shares_outstanding: Optional[float] = None,
    total_cash: float = 0.0,
    total_debt: float = 0.0,
    exit_multiple: Optional[float] = None,
) -> Optional[float]:
    """Reverse DCF: jaký růst FCF implikuje aktuální cena.

    Pokud je `exit_multiple` zadán, použije se stejná Exit Multiple metoda jako v hlavním DCF modelu
    (konzistentní interpretace). Pokud není, použije se Gordon Growth s `terminal_growth`.
    """
    if fcf is None or float(fcf) <= 0 or shares_outstanding is None or shares_outstanding <= 0:
        return None

    try:
        def dcf_at_growth(g: float) -> float:
            fv = calculate_dcf_fair_value(
                fcf, g, terminal_growth, wacc, years, shares_outstanding,
                total_cash, total_debt, exit_multiple=exit_multiple
            )
            return float(fv) if fv else 0.0

        low, high = -0.5, 1.0
        for _ in range(60):
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
# INSIDER TRADING ANALYSIS
# ============================================================================

def compute_insider_pro_signal(insider_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Advanced insider trading signal with role weighting + cluster detection (buy & sell).

    What we count (by default):
    - Primarily open-market transactions with Code in {P, S}.
    - If Code is missing, we fall back to the normalized `Transaction` label (Buy/Sell).

    Clusters:
    - Cluster buying: >=3 unique insiders BUY within a 30-day window.
    - Cluster selling: >=3 unique insiders SELL within a 30-day window.

    Signal:
    - Value-weighted net flow (BUY - SELL) normalized to [-100, +100].
    - Cluster buying adds a small positive adjustment.
    - Cluster selling adds a negative adjustment (worsens rating), even when base signal is bullish.
    """
    if insider_df is None or insider_df.empty:
        return {
            "signal": 0.0,
            "label": "Neutral",
            "confidence": 0.0,
            "insights": ["Žádné insider transakce k dispozici"],
            "recent_buys": 0,
            "recent_sells": 0,
            "cluster_buying": False,
            "cluster_selling": False,
        }

    role_weights = {
        "ceo": 3.0,
        "chief executive officer": 3.0,
        "cfo": 2.5,
        "chief financial officer": 2.5,
        "president": 2.0,
        "director": 1.5,
        "coo": 2.0,
        "vice president": 1.2,
        "officer": 1.0,
    }

    cutoff_date = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=180)  # naive UTC

    buy_signal = 0.0
    sell_signal = 0.0
    buy_count = 0
    sell_count = 0

    buy_dates: List[dt.datetime] = []
    buy_owners: List[str] = []
    sell_dates: List[dt.datetime] = []
    sell_owners: List[str] = []

    ignored_planned = 0

    def _norm(s: Any) -> str:
        try:
            return re.sub(r"\s+", " ", str(s or "")).strip().lower()
        except Exception:
            return ""

    def _is_open_market(code: str) -> bool:
        return code in {"P", "S"}

    def _cluster(dates: List[dt.datetime], owners: List[str], window_days: int = 30, min_unique: int = 3) -> bool:
        try:
            paired = sorted([(d, o) for d, o in zip(dates, owners) if d and o], key=lambda x: x[0])
            if len(paired) < min_unique:
                return False
            for i in range(len(paired)):
                start_d = paired[i][0]
                uniq = set()
                for j in range(i, len(paired)):
                    if (paired[j][0] - start_d).days > window_days:
                        break
                    uniq.add(paired[j][1])
                if len(uniq) >= min_unique:
                    return True
            return False
        except Exception:
            return False

    for _, row in insider_df.iterrows():
        try:
            date_raw = row.get("Date") if "Date" in row else row.get("Start Date")
            if date_raw is None or pd.isna(date_raw):
                continue

            trans_dt = pd.to_datetime(date_raw, errors="coerce")
            if pd.isna(trans_dt) or trans_dt.to_pydatetime() < cutoff_date:
                continue
            trans_dt_py = trans_dt.to_pydatetime().replace(tzinfo=None)  # normalize to naive

            code = str(row.get("Code") or "").strip().upper()
            tx_txt = _norm(row.get("Transaction"))

            # Exclude "non-informative" rows (10b5-1 plans, tax-withholding sells, option exercises, gifts...)
            tx_blob = " ".join([
                str(row.get("TransactionRaw") or ""),
                str(row.get("Notes") or ""),
                str(row.get("Transaction") or ""),
                str(row.get("Security") or ""),
                str(row.get("Source") or ""),
                str(row.get("FilingURL") or ""),
            ])
            tx_norm = _norm(tx_blob)

            noise_keys = [
                "10b5", "10b5-1", "10b51", "rule 10b5",
                "trading plan", "prearranged", "pre-arranged",
                "automatic", "non-discretionary", "nondiscretionary",
                "tax", "withhold", "withholding",
                "sell to cover", "cover taxes", "to cover taxes", "for taxes",
                "option", "exercise", "vesting", "grant", "award",
                "gift", "donation", "charity", "conversion", "distribution", "inherit",
            ]
            if any(k in tx_norm for k in noise_keys):
                ignored_planned += 1
                continue

            # Prefer explicit value; else compute from shares*price
            value = safe_float(row.get("Value"))
            if value is None:
                sh = safe_float(row.get("Shares"))
                pr = safe_float(row.get("Price"))
                value = (sh * pr) if (sh is not None and pr is not None) else 0.0

            # Role weighting
            position = _norm(row.get("Position"))
            weight = 1.0
            for role, w in role_weights.items():
                if role in position:
                    weight = max(weight, w)

            # Determine direction
            is_buy = False
            is_sell = False

            if code:
                if not _is_open_market(code):
                    # ignore non-open-market codes to avoid option exercises/grants/etc skew
                    continue
                is_buy = (code == "P")
                is_sell = (code == "S")
            else:
                # fallback to normalized label
                if tx_txt == "buy":
                    is_buy = True
                elif tx_txt == "sell":
                    is_sell = True
                else:
                    # last chance heuristic
                    if "buy" in tx_txt or "purchase" in tx_txt or "acquire" in tx_txt:
                        is_buy = True
                    elif "sell" in tx_txt or "sale" in tx_txt or "dispose" in tx_txt:
                        is_sell = True

            owner = str(row.get("Owner") or "").strip().upper()

            if is_buy:
                buy_signal += abs(float(value)) * weight  # abs() – někteří provideři vrací záporné
                buy_count += 1
                if owner:
                    buy_dates.append(trans_dt_py)
                    buy_owners.append(owner)
            elif is_sell:
                sell_signal += abs(float(value)) * weight  # abs() – konzistentní s buy
                sell_count += 1
                if owner:
                    sell_dates.append(trans_dt_py)
                    sell_owners.append(owner)

        except Exception:
            continue

    cluster_buying = _cluster(buy_dates, buy_owners)
    cluster_selling = _cluster(sell_dates, sell_owners)

    net = buy_signal - sell_signal
    denom = max(buy_signal + sell_signal, 1.0)
    signal = (net / denom) * 100.0

    # Cluster adjustments (additive so it can also soften opposite-direction signals)
    if cluster_buying:
        signal += 12.0
    if cluster_selling:
        signal -= 12.0

    signal = max(-100.0, min(100.0, float(signal)))

    if signal >= 50:
        label = "Strong Buy"
    elif signal >= 20:
        label = "Buy"
    elif signal >= -20:
        label = "Neutral"
    elif signal >= -50:
        label = "Sell"
    else:
        label = "Strong Sell"

    confidence = min(1.0, (buy_count + sell_count) / 12.0)

    insights: List[str] = []
    if buy_count > 0:
        insights.append(f"✅ {buy_count} insider nákupů v posledních 6 měsících")
    if sell_count > 0:
        insights.append(f"⚠️ {sell_count} insider prodejů v posledních 6 měsících")
    if ignored_planned > 0:
        insights.append(f"ℹ️ {ignored_planned} plánovaných/automatických transakcí (10b5-1, tax withholding, opce) ignorováno pro výpočet signálu")
    if cluster_buying:
        insights.append("🔥 Cluster buying: více insiderů nakupuje ve stejném období.")
    if cluster_selling:
        insights.append("🧊 Cluster selling: více insiderů prodává ve stejném období.")
    if signal > 30:
        insights.append(f"💪 Silný bullish signál od insiderů ({signal:.0f}/100)")
    elif signal < -30:
        insights.append(f"📉 Silný bearish signál od insiderů ({signal:.0f}/100)")

    return {
        "signal": signal,
        "label": label,
        "confidence": confidence,
        "insights": insights if insights else ["Žádné významné insider aktivity"],
        "recent_buys": buy_count,
        "recent_sells": sell_count,
        "cluster_buying": bool(cluster_buying),
        "cluster_selling": bool(cluster_selling),
    }
# ============================================================================
# PEER COMPARISON
# ============================================================================

def get_auto_peers(ticker: str, sector: str, info: Dict[str, Any]) -> List[str]:
    """
    Automaticky najde 3-5 konkurentů na základě tickeru a sektoru.
    """
    
    # 1) Check manual mapping first
    for sect, tickers_map in SECTOR_PEERS.items():
        if ticker in tickers_map:
            return tickers_map[ticker][:5]
    
    # 2) Try to find similar companies in the same sector
    # (In production, you'd use API like FMP or screen by market cap + industry)
    # For now, return placeholder
    
    return []


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_peer_comparison(ticker: str, peers: List[str]) -> pd.DataFrame:
    """
    Fetch comparison metrics for ticker and its peers.
    Používá paralelní fetching pro rychlost (ThreadPoolExecutor).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_tickers = [ticker] + peers

    def _fetch_one(t: str) -> Optional[Dict]:
        try:
            info = fetch_ticker_info(t)
            if not info:
                return None
            mc = safe_float(info.get('marketCap'))
            fcf_ttm_peer, _ = get_fcf_ttm_yfinance(t, mc)
            fcf_yield_peer = safe_div(fcf_ttm_peer, mc) if fcf_ttm_peer and mc else None
            return {
                "Ticker": t,
                "P/E": safe_float(info.get("trailingPE")),
                "Op. Margin": safe_float(info.get("operatingMargins")),
                "Rev. Growth": safe_float(info.get("revenueGrowth")),
                "FCF Yield": fcf_yield_peer,
                "Market Cap": mc,
                "ROE": safe_float(info.get("returnOnEquity")),
                "Gross Margin": safe_float(info.get("grossMargins")),
            }
        except Exception:
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in all_tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    if not rows:
        return pd.DataFrame()

    # Seřadit tak aby hlavní ticker byl první
    df = pd.DataFrame(rows)
    main = df[df["Ticker"] == ticker]
    rest = df[df["Ticker"] != ticker].sort_values("Market Cap", ascending=False)
    return pd.concat([main, rest], ignore_index=True)


# ============================================================================
# AI ANALYST (GEMINI)
# ============================================================================

def generate_ai_analyst_report_with_retry(ticker: str, company: str, info: Dict, metrics: Dict, 
                             dcf_fair_value: float, current_price: float, 
                             scorecard: float, macro_events: List[Dict], insider_signal: Any = None) -> Dict:
    """
    Wrapper s retry logikou pro Free Tier Gemini 2.5 Flash Lite.
    Zkusí max MAX_AI_RETRIES pokusů s RETRY_DELAY sekundami mezi pokusy.
    v10.0: Supabase cache (6h TTL) – pokud existuje čerstvý report, vrátí se z cache.
    """
    # --- v10.0: Check Supabase cache first ---
    try:
        cached = db.get_cached_ai_report(ticker, max_age_hours=6)
        if cached:
            cached["_from_cache"] = True
            return cached
    except Exception:
        pass  # cache miss or DB unavailable – proceed with fresh call

    for attempt in range(MAX_AI_RETRIES):
        try:
            result = generate_ai_analyst_report(ticker, company, info, metrics, 
                                              dcf_fair_value, current_price, 
                                              scorecard, macro_events, insider_signal)
            
            # Check if result indicates an error that should trigger retry
            if "Chyba AI analýzy" in result.get("market_situation", ""):
                error_msg = result["market_situation"]
                # Check for rate limit errors
                if any(keyword in error_msg.lower() for keyword in ["429", "quota", "rate limit", "too many"]):
                    if attempt < MAX_AI_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        result["market_situation"] = "⚠️ AI je přetížená (Rate Limit). Zkuste to za chvíli."
                        return result
            
            # --- v10.0: Cache successful result in Supabase ---
            if result.get("verdict") and result["verdict"] != "HOLD":
                try:
                    db.cache_ai_report(ticker, result)
                except Exception:
                    pass  # caching is best-effort

            return result
            
        except Exception as e:
            error_msg = str(e)
            # Check for rate limit errors
            if any(keyword in error_msg.lower() for keyword in ["429", "quota", "rate limit", "too many"]):
                if attempt < MAX_AI_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return {
                        "market_situation": "⚠️ AI je přetížená (Rate Limit). Zkuste to za chvíli.",
                        "bull_case": [],
                        "bear_case": [],
                        "verdict": "HOLD",
                        "wait_for_price": current_price,
                        "reasoning": "Rate limit překročen i po několika pokusech.",
                        "confidence": "LOW"
                    }
            else:
                # Non-rate-limit error - don't retry
                return {
                    "market_situation": f"Chyba AI analýzy: {error_msg}",
                    "bull_case": [],
                    "bear_case": [],
                    "verdict": "HOLD",
                    "wait_for_price": current_price,
                    "reasoning": "Selhalo spojení s Gemini API.",
                    "confidence": "LOW"
                }
    
    # Fallback (shouldn't reach here)
    return {
        "market_situation": "⚠️ AI selhala po všech pokusech.",
        "bull_case": [],
        "bear_case": [],
        "verdict": "HOLD",
        "wait_for_price": current_price,
        "reasoning": "Maximální počet pokusů vyčerpán.",
        "confidence": "LOW"
    }


def generate_ai_analyst_report(ticker: str, company: str, info: Dict, metrics: Dict, 
                               dcf_fair_value: float, current_price: float, 
                               scorecard: float, macro_events: List[Dict], insider_signal: Any = None) -> Dict:
    """
    Generuje hloubkovou asymetrickou analýzu pomocí Gemini.
    """
    if not GEMINI_API_KEY:
        return {"market_situation": "Chybí API klíč.", "verdict": "N/A"}

    # Vždy česky
    target_lang = "ČEŠTINĚ"

    # 2. PŘÍPRAVA DAT
    roic_val = calculate_roic(info) 
    regime = detect_market_regime(fetch_price_history(ticker, "6mo"))
    debt_ebitda = safe_div(info.get("totalDebt"), info.get("ebitda"))
    fcf_yield_val = metrics.get("fcf_yield").value if metrics.get("fcf_yield") else 0

    # v10.0: Live macro context from FRED + recent news headlines
    _macro_ctx = ""
    try:
        _macro_summary = spp_macro.get_macro_summary()
        if _macro_summary:
            _macro_lines = [f"  - {k}: {v}" for k, v in _macro_summary.items()]
            _macro_ctx = "\n".join(_macro_lines)
    except Exception:
        pass

    _news_ctx = ""
    try:
        _news_articles = spp_news.fetch_stock_news(ticker, max_results=5)
        if _news_articles:
            _news_ctx = "\n".join(
                [f"  - [{a.get('source','')}] {a.get('title','')}" for a in _news_articles[:5]]
            )
    except Exception:
        pass

    # 3. SESTAVENÍ PROMPTU
    _insider_ctx = ""
    if insider_signal and isinstance(insider_signal, dict):
        sig_val = insider_signal.get("signal", 0)
        sig_label = insider_signal.get("label", "")
        _insider_ctx = f"\n- Insider Signal: {sig_val} ({sig_label})"

    context = f"""
Jsi Seniorní Portfolio Manažer a Contrarian Analyst se specializací na ASYMETRICKÝ RISK/REWARD.
DŮLEŽITÉ: Celou analýzu a všechny texty v JSON výstupu napiš VÝHRADNĚ V ČEŠTINĚ.

VSTUPNÍ DATA:
- Aktiva: {company} ({ticker}) | Sektor: {info.get('sector')} / {info.get('industry')}
- Tržní cena: {fmt_money(current_price)} | Kalkulovaná Férovka (DCF): {fmt_money(dcf_fair_value)}
- Metriky: P/E: {info.get('trailingPE')}, ROIC: {fmt_pct(roic_val)}, Net Debt/EBITDA: {fmt_num(debt_ebitda)}, FCF Yield: {fmt_pct(fcf_yield_val)}
- Tržní Režim: {regime}
- Scorecard (0-100): {scorecard}{_insider_ctx}

MAKROEKONOMICKÝ KONTEXT (FRED live data):
{_macro_ctx if _macro_ctx else '  Nedostupné'}

NEDÁVNÉ ZPRÁVY O FIRMĚ:
{_news_ctx if _news_ctx else '  Nedostupné'}

TVŮJ ANALYTICKÝ RÁMEC (Chain-of-Thought):
1. FUNDAMENTÁLNÍ PODLAHA: Je cena blízko hodnotě aktiv? Jak bezpečný je dluh?
2. EMBEDDED OPTIONALITY: Má firma aktiva (data, patenty), která trh oceňuje nulou?
3. RED TEAMING: Hraj roli Short Sellera. Proč tato firma za 2 roky ztratí 50 % hodnoty?
4. ASYMETRIE: Je poměr mezi Downside a Upside alespoň 1:3?
5. MAKRO VLIV: Jak aktuální makro prostředí (úrokové sazby, inflace, VIX) ovlivňuje tuto firmu?
6. SENTIMENT Z MÉDIÍ: Zohledni nedávné zprávy a mediální sentiment.

VÝSTUP POUZE JSON:
{{
  "asymmetry_score": (číslo 0-100),
  "fundamental_floor": "Analýza bezpečnosti investice jednou větou.",
  "red_team_warning": "BRUTÁLNĚ upřímná analýza největšího rizika - proč to nekoupit.",
  "bull_case": ["Argument 1", "Argument 2"],
  "bear_case": ["Riziko 1", "Riziko 2"],
  "verdict": "STRONGBUY/BUY/HOLD/SELL/AVOID",
  "wait_for_price": {current_price * 0.85 if current_price else 0},
  "risk_reward_ratio": "Např. 1:4",
  "reasoning_synthesis": "Konečný verdikt pro investiční komisi. Proč právě teď?",
  "confidence": "HIGH/MEDIUM/LOW",
  "macro_impact": "Stručný komentář k vlivu makro prostředí na tuto investici."
}}
"""

    # 4. PARSOVÁNÍ JSONU
    def _extract_json(text: str) -> Dict[str, Any]:
        if not text: raise ValueError("Empty AI response")
        cleaned = re.sub(r"```json\n?|```", "", str(text)).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if not m: raise
            return json.loads(m.group(0))

    # 5. VOLÁNÍ API
    try:
        raw_text = ""
        try:
            from google import genai as genai_new
            client = genai_new.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=context)
            raw_text = getattr(resp, "text", None) or str(resp)
        except Exception:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=GEMINI_API_KEY)
            model = genai_legacy.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(context)
            raw_text = getattr(resp, "text", None) or str(resp)

        return _extract_json(raw_text)

    except Exception as e:
        return {
            "market_situation": f"Chyba AI: {str(e)}", 
            "bull_case": [], "bear_case": [], 
            "verdict": "HOLD", "wait_for_price": current_price
        }

def get_earnings_calendar_estimate(ticker: str, info: Dict[str, Any]) -> Optional[dt.date]:
    """
    Estimate next earnings date.

    NOTE:
    - For assets that do not report earnings (crypto, indices, ETFs, mutual funds), returns None.
    - For normal equities: tries Yahoo calendar first; if unavailable, falls back to a simple quarter-date heuristic.
    """
    try:
        qt = (info.get("quoteType") or "").upper()
        if qt in ("CRYPTOCURRENCY", "CRYPTO", "INDEX", "ETF", "MUTUALFUND"):
            return None
    except Exception:
        pass

    try:
        t = yf.Ticker(ticker)
        calendar = getattr(t, "calendar", None)
        if calendar is not None and not calendar.empty:
            # Look for "Earnings Date" row
            if "Earnings Date" in calendar.index:
                next_earnings = calendar.loc["Earnings Date"].iloc[0]
                if pd.notna(next_earnings):
                    return pd.to_datetime(next_earnings).date()
    except Exception:
        pass

    # Fallback: Estimate based on common patterns (most companies: late Jan, late Apr, late Jul, late Oct)
    today = dt.date.today()
    if today.month < 4:
        return dt.date(today.year, 4, 25)
    elif today.month < 7:
        return dt.date(today.year, 7, 25)
    elif today.month < 10:
        return dt.date(today.year, 10, 25)
    else:
        return dt.date(today.year + 1, 1, 25)


# ============================================================================
# WATCHLIST & MEMOS  (v10.0 – Supabase + local JSON fallback via spp_database)
# ============================================================================

def get_watchlist() -> Dict[str, Any]:
    """Supabase-first, local JSON fallback."""
    return db.get_watchlist()


def set_watchlist(data: Dict[str, Any]) -> None:
    """Supabase-first, local JSON fallback."""
    db.set_watchlist(data)


def get_memos() -> Dict[str, Any]:
    """Supabase-first, local JSON fallback."""
    return db.get_memos()


def set_memos(data: Dict[str, Any]) -> None:
    """Supabase-first, local JSON fallback."""
    db.set_memos(data)


# ============================================================================
# PDF EXPORT
# ============================================================================

def export_memo_pdf(ticker: str, company: str, memo: Dict[str, str], summary: Dict[str, str]) -> Optional[bytes]:
    """Export memo to PDF."""
    if not _HAS_PDF:
        return None
    
    try:
        from io import BytesIO
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, 10*inch, f"Investment Memo: {company} ({ticker})")
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, 9.5*inch, "Summary")
        c.setFont("Helvetica", 10)
        y = 9.2*inch
        for key, val in summary.items():
            c.drawString(1*inch, y, f"{key}: {val}")
            y -= 0.2*inch
        
        y -= 0.3*inch
        sections = [
            ("Thesis", memo.get("thesis", "")),
            ("Key Drivers", memo.get("drivers", "")),
            ("Risks", memo.get("risks", "")),
            ("Catalysts", memo.get("catalysts", "")),
            ("Buy Conditions", memo.get("buy_conditions", "")),
            ("Notes", memo.get("notes", ""))
        ]
        
        for title, content in sections:
            if y < 2*inch:
                c.showPage()
                y = 10*inch
            
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1*inch, y, title)
            y -= 0.2*inch
            
            c.setFont("Helvetica", 9)
            lines = content.split('\n')
            for line in lines[:10]:
                if y < 1*inch:
                    break
                c.drawString(1.2*inch, y, line[:80])
                y -= 0.15*inch
            y -= 0.2*inch
        
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    except Exception:
        return None


# ============================================================================
# VERDICT LOGIC
# ============================================================================



def _detect_value_trap_impl(info: Dict[str, Any], metrics: Dict[str, "Metric"]) -> Tuple[bool, str]:
    """
    Detekce potenciální "pasti na hodnotu".
    
    Returns:
        (is_trap, warning_message)
    """
    # OPRAVA: správně čteme metriky z dict (dříve všechno tahalo marketCap)
    pe = metrics.get("pe").value if metrics.get("pe") else None
    revenue_growth = metrics.get("revenue_growth").value if metrics.get("revenue_growth") else None
    debt_to_equity = metrics.get("debt_to_equity").value if metrics.get("debt_to_equity") else None
    eps = safe_float(info.get("trailingEps"))
    
    is_trap = False
    warnings_list = []
    
    # Podmínka 1: Nízké P/E (< 10)
    if pe and pe < 10:
        # Podmínka 2: Klesající tržby
        if revenue_growth is not None and revenue_growth < -0.05:
            is_trap = True
            warnings_list.append("Klesající tržby (YoY)")
        
        # Podmínka 3: Vysoký dluh (D/E > 2.0 po normalizaci)
        if debt_to_equity is not None and debt_to_equity > 2.0:
            is_trap = True
            warnings_list.append("Vysoká zadluženost (D/E > 2)")
        
        # Podmínka 4: Negativní EPS
        if eps is not None and eps <= 0:
            is_trap = True
            warnings_list.append("Negativní/nulové EPS")
    
    if is_trap:
        warning_msg = f"⚠️ **Potenciální Value Trap**: {', '.join(warnings_list)}. Nízká valuace může být oprávněná kvůli úpadku byznysu."
        return True, warning_msg
    
    return False, ""


def get_advanced_verdict(
    scorecard: float,
    mos_dcf: Optional[float],
    mos_analyst: Optional[float],
    insider_signal: float,
    implied_growth: Optional[float]
) -> Tuple[str, str, List[str]]:
    """
    Advanced verdict with multiple signals.
    
    Returns: (verdict, color, warnings)
    """
    
    warnings = []
    
    # Base verdict from scorecard
    if scorecard >= 85:
        base = "STRONG BUY"
        color = "#00ff88"
    elif scorecard >= 60:
        base = "BUY"
        color = "#88ff00"
    elif scorecard >= 45:
        base = "HOLD"
        color = "#ffaa00"
    elif scorecard >= 30:
        base = "CAUTION"
        color = "#ff8800"
    else:
        base = "AVOID"
        color = "#ff4444"
    
    # Adjust for MOS
    if mos_dcf is not None:
        if mos_dcf >= 0.20:
            if base in ["HOLD", "CAUTION"]:
                base = "BUY"
                color = "#88ff00"
        elif mos_dcf < -0.15:
            if base in ["STRONG BUY", "BUY"]:
                base = "HOLD"
                color = "#ffaa00"
                warnings.append("⚠️ DCF model ukazuje přeceněnost (-15% MOS)")
    
    # Check for mismatch: Analysts bullish but DCF says overvalued
    if mos_analyst is not None and mos_dcf is not None:
        if mos_analyst > 0.15 and mos_dcf < -0.10:
            warnings.append("🚨 MISMATCH WARNING: Analytici vidí upside +15%, ale DCF model ukazuje overvalued -10%!")
            warnings.append("   → Trh možná implikuje vyšší růst než je ve tvém DCF modelu konzervativní")
    
    # Insider signal adjustment
    if insider_signal > 50:
        warnings.append(f"✅ Silný insider buying signal (+{insider_signal:.0f}) podporuje BUY tezi")
    elif insider_signal < -30:
        warnings.append(f"⚠️ Negativní insider selling signal ({insider_signal:.0f})")
    
    # Implied growth check
    if implied_growth is not None:
        if implied_growth > 0.25:
            warnings.append(f"⚠️ Trh implikuje velmi agresivní růst FCF ({implied_growth*100:.0f}% ročně) - vysoká očekávání!")
        elif implied_growth < 0:
            warnings.append(f"📉 Trh implikuje pokles FCF ({implied_growth*100:.0f}%) - možná undervalued opportunity")
    
    return base, color, warnings


# End of Part 1
# ============================================================================
# MAIN APPLICATION
# ============================================================================


def render_twitter_timeline(handle: str, height: int = 600) -> None:
    """Render X/Twitter content without embeds (widgets are often blocked)."""
    handle = (handle or "").lstrip("@").strip()
    if not handle:
        st.info("Vyber guru účet.")
        return
    st.warning("⚠️ X (Twitter) často blokuje náhledy v cizích aplikacích. Použij přímý odkaz níže.")
    st.markdown(f"👉 Otevřít profil **@{handle}**: https://twitter.com/{handle}")

def analyze_social_text_with_gemini(text: str) -> str:
    """Analyze manually pasted tweet/comment using Gemini."""
    text = (text or "").strip()
    if not text:
        return "Chybí text k analýze."

    if not GEMINI_API_KEY:
        return "AI analýza není dostupná (chybí GEMINI_API_KEY)."

    prompt = f"""Jako seniorní investor analyzuj tento text z sociálních sítí týkající se financí.

1) Jaký je sentiment (Bullish/Bearish/Neutral)?
2) Jsou tam nějaká fakta nebo jen šum?
3) Verdikt pro investora.

TEXT:
{text}
"""

    try:
        # Try new google-genai SDK first
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return (response.text or "").strip()
        except ImportError:
            # Fallback to old SDK
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()
    except Exception as e:
        return f"Chyba při volání Gemini: {e}"


# -----------------------------------------------------------------------------
# Smart parameter estimation (Quality Premium)
# -----------------------------------------------------------------------------
def estimate_smart_params(info: Dict[str, Any], metrics: Dict[str, "Metric"]) -> Dict[str, Any]:
    """
    Konzervativní odhad DCF parametrů.
    Cíl: Zabránit "úletům" u Mega Caps (MSFT, AAPL) a opravit Amazon.
    """
    market_cap = safe_float(info.get('marketCap')) or 0.0
    sector = str(info.get('sector') or "").strip()
    
    # 1. DEFINICE VELIKOSTI
    is_mega_cap = market_cap > 200e9  # > 200 mld USD
    is_large_cap = market_cap > 50e9   # > 50 mld USD

    # 2. WACC (Diskontní sazba)
    # Zvedáme "podlahu" na 9.0% pro větší bezpečnost
    beta = safe_float(info.get("beta"))
    if beta is None or beta <= 0:
        base_wacc = 0.10
    else:
        # RiskFree (4.2%) + Beta * ERP (5.0%)
        base_wacc = 0.042 + (beta * 0.05)
    
    # Omezení WACC: Min 9%, Max 15%
    wacc = max(0.09, min(0.15, base_wacc))
    
    # Size Premium: Malé firmy jsou rizikovější -> přidáme 1.5%
    if market_cap < 10e9 and market_cap > 0:
        wacc += 0.015

    # 3. RŮST (Weighted Growth)
    # Vážíme tržby (70%) a zisky (30%), protože tržby jsou stabilnější
    rev_g = None
    earn_g = None
    try:
        if metrics.get("revenue_growth") and metrics["revenue_growth"].value is not None:
            rev_g = float(metrics["revenue_growth"].value)
    except:
        pass
    
    try:
        if metrics.get("earnings_growth") and metrics["earnings_growth"].value is not None:
            earn_g = float(metrics["earnings_growth"].value)
    except:
        pass

    # Výpočet váženého růstu
    if rev_g is not None and earn_g is not None:
        raw_growth = (0.7 * rev_g) + (0.3 * earn_g)
    elif rev_g is not None:
        raw_growth = rev_g
    elif earn_g is not None:
        raw_growth = earn_g
    else:
        raw_growth = 0.10  # Fallback

    # 4. STROP RŮSTU (Growth Cap) - Tady se krotí ty "brutální" čísla
    if is_mega_cap:
        # Giganti nemohou růst o 20% věčně -> Cap 12%
        growth_cap = 0.08
    elif is_large_cap:
        growth_cap = 0.12
    else:
        # Malé dravé firmy mohou růst rychleji
        growth_cap = 0.20
        
    growth = max(0.03, min(growth_cap, raw_growth))

    # 5. EXIT MULTIPLE (Konzervativní)
    # Základ podle sektoru
    sector_l = sector.lower()
    
    if "technology" in sector_l:
        base_multiple = 20.0
    elif "communication" in sector_l:  # Google, Meta
        base_multiple = 18.0
    elif "consumer cyclical" in sector_l:  # Amazon, Tesla
        base_multiple = 20.0
    elif "financial" in sector_l or "energy" in sector_l:
        base_multiple = 12.0
    elif "healthcare" in sector_l:
        base_multiple = 18.0
    else:
        base_multiple = 15.0
        
    # === PLYNULÝ QUALITY PREMIUM - BODOVÝ SYSTÉM ===
    quality_score = 0
    
    # ROE > 15% → +2 body, > 10% → +1 bod
    roe = safe_float(metrics.get("roe").value) if metrics.get("roe") else None
    if roe is not None:
        if roe > 0.15:
            quality_score += 2
        elif roe > 0.10:
            quality_score += 1

    # Net Margin > 20% → +2 body, > 10% → +1 bod
    pm = safe_float(metrics.get("profit_margin").value) if metrics.get("profit_margin") else None
    if pm is not None:
        if pm > 0.20:
            quality_score += 2
        elif pm > 0.10:
            quality_score += 1

    # ROIC (aproximace pomocí ROA) > 15% → +2 body, > 10% → +1 bod
    roa = safe_float(metrics.get("roa").value) if metrics.get("roa") else None
    if roa is not None:
        if roa > 0.15:
            quality_score += 2
        elif roa > 0.10:
            quality_score += 1
    
    # Debt/Equity < 0.5 (po normalizaci) → +1 bod
    debt_eq = safe_float(metrics.get("debt_to_equity").value) if metrics.get("debt_to_equity") else None
    if debt_eq is not None and debt_eq < 0.5:
        quality_score += 1
    
    # Konverze bodů na Exit Multiple: Base + score, max 25x
    exit_multiple = base_multiple + quality_score
    exit_multiple = min(25.0, exit_multiple)

    return {
        "wacc": float(wacc),
        "growth": float(growth),
        "exit_multiple": float(exit_multiple),
        "is_mega_cap": bool(is_mega_cap),
        "market_cap": float(market_cap),
        "sector": sector
    }


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

        # 1. FCF Payout Ratio
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

        # 2. Zadluženost
        de = safe_float(info.get("debtToEquity"))
        if de is not None:
            de_n = de / 100.0 if de > 10 else de
            if de_n < 0.5:
                score += 1
                details.append("✅ Nízká zadluženost (D/E < 0.5)")
            elif de_n < 1.5:
                details.append("👍 Střední zadluženost")
            else:
                details.append("⚠️ Vysoká zadluženost")

        # 3. Provozní CF kladný
        ocf = safe_float(info.get("operatingCashflow"))
        if ocf and ocf > 0:
            score += 1
            details.append("✅ Kladné provozní CF")
        elif ocf is not None:
            details.append("🚨 Záporné provozní CF!")

        # 4. 5Y dividend yield (firma vyplácí dlouho)
        five_y = safe_float(info.get("fiveYearAvgDividendYield"))
        if five_y and five_y > 0:
            details.append("✅ Dividenda vyplácena min. 5 let")

        labels = {
            5: "Velmi bezpečná 🟢", 4: "Bezpečná 🟢",
            3: "Přiměřená 🟡", 2: "Opatrnost ⚠️",
            1: "Riziková 🔴", 0: "Velmi riziková 🚨"
        }
        label = labels.get(score, f"Skóre {score}/5")
        return score, label + "\n" + "\n".join(details)
    except Exception:
        return 0, "Data nedostupná"


def build_radar_data(
    category_scores: Dict[str, float],
    piotroski_score: int,
    insider_signal: float,
    tech_signals: Dict[str, Any],
) -> Dict[str, float]:
    """
    Připraví data pro Spider/Radar Chart (7 os, hodnoty 0–100).
    """
    tech_score = 50.0
    try:
        rsi = tech_signals.get("rsi")
        macd_label = tech_signals.get("macd_label", "")
        pct_ma200 = tech_signals.get("pct_from_ma200")
        components = []
        if rsi is not None:
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
            import numpy as _np
            tech_score = float(_np.mean(components))
    except Exception:
        pass

    insider_norm = max(0.0, min(100.0, (insider_signal + 100) / 2.0))
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


# Pouze čeština – překladový systém odstraněn

def main():
    # Session state initialization
    if "force_tab_label" not in st.session_state:
        st.session_state.force_tab_label = None
    if "ai_report_data" not in st.session_state:
        st.session_state.ai_report_data = None
    if "ai_report_ticker" not in st.session_state:
        st.session_state.ai_report_ticker = None
    if "active_tab_index" not in st.session_state:
        st.session_state.active_tab_index = 0
    if "ai_report_cache" not in st.session_state:
        st.session_state.ai_report_cache = {}
    
    """Main application entry point."""

    # --- UI mode state (picker vs results) ---
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = "PICKER"
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = ""

    if "close_sidebar_js" not in st.session_state:
        st.session_state.close_sidebar_js = False

    # Optional: hide sidebar overlay on mobile after analyze (keeps results visible)
    if st.session_state.get("sidebar_hidden"):
        st.markdown("""
        <style>
        @media (max-width: 900px) {
          section[data-testid="stSidebar"], [data-testid="stSidebar"] {
            transform: translateX(-120%) !important;
            opacity: 0 !important;
            pointer-events: none !important;
          }
        }
        </style>
        """, unsafe_allow_html=True)

    if "sidebar_hidden" not in st.session_state:
        st.session_state.sidebar_hidden = False


    # If requested (e.g., after clicking Analyze), inject JS in MAIN area to force-close the sidebar on mobile.
    if st.session_state.get("close_sidebar_js"):
        components.html(js_close_sidebar(), height=0, width=0)
        st.session_state.close_sidebar_js = False



    # All CSS is loaded via load_custom_css() – no duplicate inline styles
    
    # ========================================================================
    # SIDEBAR - Settings & Controls
    # ========================================================================

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 8px 0 4px 0;">
            <span style="font-size: 2rem;">📈</span><br>
            <span style="font-size: 1.1rem; font-weight: 800; background: linear-gradient(135deg, #00e68a, #00b4d8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Stock Picker Pro</span><br>
            <span style="font-size: 0.72rem; color: #555d70; letter-spacing: 1.5px; text-transform: uppercase;">v10.0 · Quantitative Analysis</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Ticker input (Form -> Enter submits)
        with st.form("analyze_form", clear_on_submit=False):
            default_ticker = st.session_state.get("last_ticker") or "AAPL"
            _raw_ticker = st.text_input(
                "Ticker Symbol",
                value=str(default_ticker),
                help="Zadej ticker (např. AAPL, MSFT, GOOGL) a potvrď Enterem",
                max_chars=10,
                key="ticker_input",
            )
            ticker_input = (_raw_ticker or "").upper().strip()
            analyze_btn = st.form_submit_button("🔍 Analyzovat", type="primary", use_container_width=True)

        if analyze_btn:
            st.session_state.close_sidebar_js = True
            st.session_state.sidebar_hidden = True
            st.session_state.ui_mode = "RESULTS"
            st.session_state.selected_ticker = ticker_input
            st.session_state["last_ticker"] = ticker_input
            st.rerun()
        # Debug & Data settings
        debug_mode = st.checkbox("🧪 Debug mode", value=st.session_state.get("debug_mode", False),
                                 help="Zobrazí panel Diagnostika & Zdroje (fallbacky, gating, provenance).", key="debug_mode")
        auto_adjust_prices = st.checkbox("📉 Auto-adjust ceny (splity/dividendy)", value=st.session_state.get("auto_adjust_prices", False),
                                         help="Použije adjustované historické ceny. Vhodné pro dlouhé grafy a simulace.", key="auto_adjust_prices")
        if st.button("♻️ Refresh cache", use_container_width=True, help="Vyčistí cache (data se znovu stáhnou)."):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            diag_log("Cache cleared by user", "INFO")
            st.rerun()

        # Theme toggle (dark/light)
        render_theme_toggle()
        # Global keyboard shortcuts (Ctrl+K = search)
        inject_keyboard_shortcuts()

        render_diagnostics_panel()
        st.markdown("---")
        
        # DCF Settings
        with st.expander("⚙️ DCF Parametry", expanded=False):
            smart_dcf = st.checkbox("⚡ Smart DCF (Automaticky)", value=True, key="smart_dcf")
            dcf_growth = st.slider(
                "Růst FCF (roční)",
                0.0, 0.50, 0.10, 0.01,
                help="Očekávaný roční růst FCF. Historicky S&P 500 ≈ 8–10 %, tech 15–25 %, utility 3–5 %. Smart DCF odhaduje automaticky.",
                disabled=smart_dcf
            )
            dcf_terminal = st.slider(
                "Terminální růst",
                0.0, 0.10, 0.03, 0.01,
                help="Terminální (věčný) růst po skončení projekce. Typicky 2–3 % ≈ inflace. NIKDY nezadávej > WACC – model by se rozpadl. Malá změna = velký dopad na fair value!"
            )
            dcf_wacc = st.slider(
                "WACC (diskont)",
                0.05, 0.20, 0.10, 0.01,
                help="WACC: diskontní sazba DCF. Pro US large cap typicky 8–12 %. Vyšší WACC = nižší fair value. Smart DCF odhaduje z beta a sektoru.",
                disabled=smart_dcf
            )
            dcf_years = st.slider(
                "Projektované roky",
                3, 10, 5, 1,
                help="Na kolik let dopředu modelujeme FCF. Standardně 5 let. Pro stabilní firmy 5–7 let, pro cyklické 3–5 let."
            )
            dcf_exit_multiple = st.slider(
                "Exit Multiple (FCF)",
                10.0, 50.0, 25.0, 1.0,
                help="Exit Multiple: kolikrát FCF hodnotíme v posledním roce. Tech firmy 20–35×, utility 10–15×, industrials 15–20×.",
                disabled=smart_dcf
            )
        
        st.markdown("---")
        
        # AI Settings
        with st.expander("🤖 AI Nastavení", expanded=False):
            use_ai = st.checkbox(
                "Povolit AI analýzu",
                value=bool(GEMINI_API_KEY),
                help="Vyžaduje Gemini API klíč",
                disabled=not GEMINI_API_KEY
            )
            if not GEMINI_API_KEY:
                st.warning("⚠️ Nastav GEMINI_API_KEY v kódu")
        
        st.markdown("---")
        
        # Quick links
        st.markdown("### 🔗 Odkazy")
        if ticker_input:
            st.markdown(f"- [Yahoo Finance](https://finance.yahoo.com/quote/{ticker_input})")
            st.markdown(f"- [SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=&type=&dateb=&owner=exclude&count=40&search_text={ticker_input})")
            st.markdown(f"- [Finviz](https://finviz.com/quote.ashx?t={ticker_input})")

        # Quick Score badge (zobrazí se po analýze)
        _qs = st.session_state.get("quick_score_cache")
        if _qs:
            st.markdown("---")
            st.markdown("### 📊 Poslední výsledek")
            _qc = _qs.get("color", "#aaa")
            st.markdown(f"""
            <div style="border: 2px solid {_qc}; border-radius: 8px; padding: 10px; text-align: center; font-size: 0.85rem;">
                <b>{_qs.get('ticker', '—')}</b><br>
                <span style="font-size: 1.4rem; font-weight: 900; color: {_qc};">{_qs.get('scorecard', '—'):.0f}<span style="font-size: 0.9rem; opacity: 0.6;">/100</span></span><br>
                <span style="color: {_qc}; font-weight: 700;">{_qs.get('verdict', '—')}</span><br>
                <span style="opacity: 0.6; font-size: 0.75rem;">{_qs.get('price', '—')} | MOS: {_qs.get('mos', '—')}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Welcome screen if no analysis yet
    if st.session_state.get("ui_mode") == "PICKER" and (not analyze_btn) and ("last_ticker" not in st.session_state):
        display_welcome_screen()
        st.stop()
    
        # Pokud jsme ve výsledcích, nabídni rychlý návrat na výběr (hlavně pro mobil)
    if st.session_state.get("ui_mode") == "RESULTS":
        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("☰ Menu", use_container_width=True):
                st.session_state.sidebar_hidden = False
                st.rerun()
        with colB:
            st.empty()
        if st.button("⬅️ Zpět na výběr", use_container_width=True):
            st.session_state.ui_mode = "PICKER"
            st.session_state.sidebar_hidden = False
            st.session_state.selected_ticker = ""
            st.session_state.pop("last_ticker", None)
            st.rerun()

    # Process ticker
    _raw_sel = ticker_input if analyze_btn else st.session_state.get("last_ticker", "AAPL")
    ticker = normalize_ticker(_raw_sel)
    if ticker != (_raw_sel or ""):
        diag_log(f"Ticker normalizován: {_raw_sel} → {ticker}", "INFO")
    st.session_state["last_ticker"] = ticker
    
    # Fetch data
    with st.spinner(f"📊 Načítám data pro {ticker}..."):
        info = fetch_ticker_info(ticker)
        
        if not info:
            st.error(f"❌ Nepodařilo se načíst data pro {ticker}. Zkontroluj ticker.")
            st.stop()
        
        # === CRYPTO DETEKCE (BTC-USD, ETH-USD, ...) ===
        _quote_type = (info.get("quoteType") or "").upper()
        _is_crypto_asset = _quote_type in ("CRYPTOCURRENCY", "CRYPTO")
        # Asset gating (DCF/Earnings/Insiders/Peers jen pro EQUITY)
        asset_flags = detect_asset_flags(ticker, info)
        supports_dcf = bool(asset_flags.get("supports_dcf"))
        supports_earnings = bool(asset_flags.get("supports_earnings"))
        supports_insiders = bool(asset_flags.get("supports_insiders"))
        supports_peers = bool(asset_flags.get("supports_peers"))
        diag_set_source("asset", asset_flags)
        
        # Pro crypto: price může být v regularMarketPrice (ne currentPrice)
        if _is_crypto_asset and not info.get("currentPrice"):
            info["currentPrice"] = info.get("regularMarketPrice")
        # Pro crypto: shares = circulatingSupply (pro výpočty)
        if _is_crypto_asset and not info.get("sharesOutstanding"):
            info["sharesOutstanding"] = info.get("circulatingSupply")

        company = info.get("longName") or info.get("shortName") or ticker
        metrics = extract_metrics(info, ticker)
        # Multi-source enrichment for core fundamentals (fills missing values + tracks sources)
        metrics, metrics_enrich_dbg = enrich_metrics_multisource(ticker, metrics, info)
        st.session_state["metrics_enrich_debug"] = metrics_enrich_dbg

        price_history = fetch_price_history(ticker, period="1y")
        income, balance, cashflow = fetch_financials(ticker)
        
        # Advanced data
        ath = get_all_time_high(ticker)
        # Crypto nemá insider data – přeskočit
        if _is_crypto_asset:
            insider_df = None
            insider_signal = {
                "signal": 0.0, "label": "N/A (krypto)",
                "confidence": 0.0, "insights": ["Insider data nejsou dostupná pro kryptoměny"],
                "recent_buys": 0, "recent_sells": 0,
                "cluster_buying": False, "cluster_selling": False,
            }
        else:
            insider_df = fetch_insider_transactions_fmp(ticker)
            insider_signal = compute_insider_pro_signal(insider_df)
        
        # DCF calculations
        market_cap_for_fcf = safe_float(info.get('marketCap'))
        fcf, fcf_dbg = get_fcf_ttm_yfinance(ticker, market_cap_for_fcf)
        # FCF debug suppressed in UI
        shares = safe_float(info.get("sharesOutstanding"))
        current_price = metrics.get("price").value if metrics.get("price") else None
        diag_set_source("price", {"value": current_price, "source": "metrics.price" if metrics.get("price") else "none"})

        # Cash/Debt bridge (používá se v DCF, Reverse DCF i Monte Carlo)
        total_cash = safe_float(info.get("totalCash")) or 0
        total_debt = safe_float(info.get("totalDebt")) or 0
        # Fallback: sharesOutstanding někdy chybí (ADR/ETF...). Odhadneme z market cap / ceny.
        if (not shares or shares <= 0) and current_price and market_cap_for_fcf:
            try:
                shares = float(market_cap_for_fcf) / float(current_price)
                shares_estimated = True
                diag_log("sharesOutstanding missing -> shares estimated as marketCap/currentPrice", "WARN")
                diag_set_source("shares", {"value": shares, "source": "derived:marketCap/currentPrice", "marketCap": market_cap_for_fcf, "price": current_price})
            except Exception as e:
                diag_log(f"shares estimation failed: {e}", "WARN")


        # Decide DCF inputs (Smart vs Manual)
        used_dcf_growth = float(dcf_growth)
        used_dcf_wacc = float(dcf_wacc)
        used_exit_multiple = float(dcf_exit_multiple)
        used_mode_label = "Manual"

        if st.session_state.get("smart_dcf", True) and supports_dcf and (not _is_crypto_asset):
            smart = estimate_smart_params(info, metrics)
            used_dcf_growth = float(smart["growth"])
            used_dcf_wacc = float(smart["wacc"])
            used_exit_multiple = float(smart["exit_multiple"])
            used_mode_label = "Smart"

        
        # --- Amazon-style reinvestment heavy adjustment (Adjusted FCF) ---
        # If FCF is unusually low relative to Operating Cash Flow, treat it as heavy reinvestment and
        # use an adjusted cash-flow proxy for DCF (maintenance earnings proxy).
        dcf_fcf_used = fcf if supports_dcf else None
        if not supports_dcf:
            diag_log(f"DCF gated: asset_class={asset_flags.get('asset_class')} quoteType={asset_flags.get('quoteType')}", "INFO")
        try:
            operating_cashflow = safe_float(info.get("operatingCashflow"))
        except Exception:
            operating_cashflow = None

        # DCF needs a positive cash-flow base. Some tickers have missing/negative FCF, and some
        # (Amazon-style) show very low FCF due to heavy reinvestment. In both cases we use a
        # conservative "maintenance earnings" proxy derived from Operating Cash Flow (OCF).
        if operating_cashflow and operating_cashflow > 0:
            # If FCF is missing/non-positive, use OCF proxy directly.
            if not dcf_fcf_used or dcf_fcf_used <= 0:
                dcf_fcf_used = operating_cashflow * 0.6
                diag_log("FCF missing/non-positive -> using 60% OCF proxy for DCF", "WARN")
                diag_set_source("dcf_fcf_base", {"mode":"OCF_PROXY_60PCT","reason":"FCF missing/non-positive","operatingCashflow": operating_cashflow})
            # If FCF looks unrealistically low vs OCF, treat it as heavy reinvestment and use the proxy.
            elif dcf_fcf_used < (0.3 * operating_cashflow):
                dcf_fcf_used = operating_cashflow * 0.6
                diag_log("FCF very low vs OCF -> using 60% OCF proxy for DCF", "WARN")
                diag_set_source("dcf_fcf_base", {"mode":"OCF_PROXY_60PCT","reason":"FCF < 30% OCF (reinvestment heavy)","operatingCashflow": operating_cashflow})

        
        # Provenance: cashflow base for DCF
        # Provenance: cashflow base for DCF
        if "dcf_fcf_base" not in st.session_state.get("_diag_sources", {}):
            diag_set_source("dcf_fcf_base", {"mode":"FCF_TTM_RAW","fcf_raw": fcf})

        # Cash/Debt bridge (používá se i pro reverse/MC; definujeme vždy)
        total_cash = safe_float(info.get("totalCash")) or 0
        total_debt = safe_float(info.get("totalDebt")) or 0
        cash_missing = (info.get("totalCash") is None)
        debt_missing = (info.get("totalDebt") is None)
        diag_set_source("cash_debt", {"totalCash": total_cash, "totalDebt": total_debt, "cash_missing": cash_missing, "debt_missing": debt_missing})
        fair_value_dcf = None
        mos_dcf = None
        implied_growth = None
        
        if dcf_fcf_used and shares and dcf_fcf_used > 0:
            # --- NOVÝ VÝPOČET DCF (Exit Multiple Metoda) ---
            # 1. Spočítáme budoucí FCF pro každý rok
            future_fcf = []
            current_fcf = dcf_fcf_used
            
            # Diskontní faktor
            discount_factors = [(1 + used_dcf_wacc) ** i for i in range(1, dcf_years + 1)]
            
            for i in range(dcf_years):
                current_fcf = current_fcf * (1 + used_dcf_growth)
                future_fcf.append(current_fcf)
            
            # 2. Terminal Value (Hodnota na konci 5. roku)
            # Použijeme Exit Multiple (pro Big Tech standardně 25x, ne konzervativní Gordon)
            exit_multiple = float(used_exit_multiple)
            terminal_value = future_fcf[-1] * exit_multiple
            
            # 3. Diskontování na dnešní hodnotu (PV)
            pv_cash_flows = sum([f / d for f, d in zip(future_fcf, discount_factors)])
            pv_terminal_value = terminal_value / ((1 + used_dcf_wacc) ** dcf_years)
            
            enterprise_value = pv_cash_flows + pv_terminal_value
            
            # 4. Equity Value (EV + Cash - Debt)
            # total_cash/total_debt defined above
            equity_value = enterprise_value + total_cash - total_debt
            
            fair_value_dcf = equity_value / shares
            
            # Přepočet MOS a Implied Growth
            if current_price:
                mos_dcf = (fair_value_dcf / current_price) - 1.0
                implied_growth = reverse_exit_multiple_implied_growth(
                    price=current_price,
                    fcf0=float(dcf_fcf_used),
                    wacc=float(used_dcf_wacc),
                    years=int(dcf_years),
                    shares=float(shares),
                    cash=float(total_cash),
                    debt=float(total_debt),
                    exit_multiple=float(used_exit_multiple),
                )
        
        # Data Quality & Confidence (pro DCF UI + diagnostiku)
        one_off_flag = False  # best-effort; plná heuristika by vyžadovala další zdroje
        dq = build_data_quality(
            ticker=ticker,
            info=info,
            asset_flags=asset_flags,
            fcf_raw=fcf,
            fcf_used=dcf_fcf_used,
            used_ocf_proxy=bool(operating_cashflow and dcf_fcf_used and fcf and fcf > 0 and (dcf_fcf_used != fcf)),
            shares_estimated=bool(locals().get("shares_estimated", False)),
            cash_missing=bool(cash_missing),
            debt_missing=bool(debt_missing),
            one_off_flag=bool(one_off_flag),
        )
        dcf_conf, dcf_conf_reasons = compute_dcf_confidence(dq)
        st.session_state["dcf_confidence"] = dcf_conf
        st.session_state["dcf_confidence_reasons"] = dcf_conf_reasons
        diag_set_source("data_quality", dq)
        diag_set_source("dcf_confidence", {"score": dcf_conf, "reasons": dcf_conf_reasons})
        # Analyst fair value
        analyst_target = metrics.get("target_mean").value if metrics.get("target_mean") else None
        mos_analyst = None
        if analyst_target and current_price:
            mos_analyst = (analyst_target / current_price) - 1.0
        
        # Scorecard
        scorecard, category_scores, individual_scores = build_scorecard_advanced(metrics, info)
        
        # Verdict
        verdict, verdict_color, verdict_warnings = get_advanced_verdict(
            scorecard, mos_dcf, mos_analyst, insider_signal.get("signal", 0), implied_growth
        )

        # Quick Score cache pro sidebar badge
        st.session_state["quick_score_cache"] = {
            "ticker": ticker,
            "scorecard": scorecard,
            "verdict": verdict,
            "color": verdict_color,
            "price": fmt_money(current_price) if current_price else "—",
            "mos": f"{mos_dcf*100:+.1f}%" if mos_dcf is not None else "—",
        }
        
        # Peers
        sector = info.get("sector", "")
        auto_peers = get_auto_peers(ticker, sector, info)

        # === NOVÉ ANALYTICKÉ VÝPOČTY v6.0 ===
        # Technické indikátory
        price_history_1y = fetch_price_history(ticker, "1y", auto_adjust=st.session_state.get("auto_adjust_prices", False))
        diag_set_source("price_history", {"period":"1y","auto_adjust": bool(st.session_state.get("auto_adjust_prices", False))})
        tech_signals = calculate_technical_signals(price_history_1y)

        # Piotroski F-Score
        piotroski_score, piotroski_breakdown = calculate_piotroski_fscore(info, income, balance, cashflow)

        # Altman Z-Score
        altman_z, altman_zone = calculate_altman_zscore(info, income=income, balance=balance)

        # Graham Number
        graham_number = calculate_graham_number(info)

        # Earnings Quality
        earnings_quality_ratio, earnings_quality_label = calculate_earnings_quality(info)

        # Short Interest
        short_interest = get_short_interest(info)

        # Monte Carlo DCF
        mc_dcf = {}
        if dcf_fcf_used and shares and dcf_fcf_used > 0 and shares > 0:
            mc_dcf = monte_carlo_dcf(dcf_fcf_used, used_dcf_growth, dcf_terminal, used_dcf_wacc, dcf_years, shares, total_cash, total_debt, exit_multiple=used_exit_multiple)

        # Value Trap detection (nyní funguje správně)
        is_value_trap, value_trap_msg = detect_value_trap(info, metrics)

        # Earnings countdown (jen pro EQUITY)
        next_earnings = None
        earnings_countdown = None
        if supports_earnings:
            next_earnings = get_earnings_calendar_estimate(ticker, info)
            if next_earnings:
                earnings_countdown = (next_earnings - dt.date.today()).days
        else:
            diag_log(f"Earnings gated: asset_class={asset_flags.get('asset_class')}", "INFO")

        # === NOVÉ v9.0 ===
        # Crypto detekce
        _is_crypto = (info.get("quoteType", "").upper() in ("CRYPTOCURRENCY", "CRYPTO"))

        # Net Debt / EBITDA
        _total_debt_val = safe_float(info.get("totalDebt")) or 0
        _total_cash_val = safe_float(info.get("totalCash")) or 0
        _ebitda_val = safe_float(info.get("ebitda"))
        net_debt_ebitda = None
        if _ebitda_val and _ebitda_val > 0:
            net_debt_ebitda = round((_total_debt_val - _total_cash_val) / _ebitda_val, 2)

        # Rule of 40 (jen pro tech/SaaS)
        rule_of_40 = None
        _rev_g = metrics.get("revenue_growth").value if metrics.get("revenue_growth") else None
        _op_m = metrics.get("operating_margin").value if metrics.get("operating_margin") else None
        if _rev_g is not None and _op_m is not None:
            rule_of_40 = round((_rev_g * 100) + (_op_m * 100), 1)

        # FCF Margin = FCF / Revenue
        fcf_margin = None
        _revenue = safe_float(info.get("totalRevenue"))
        if fcf and _revenue and _revenue > 0:
            fcf_margin = fcf / _revenue

        # Dividend Safety Score
        div_safety_score, div_safety_label = calculate_dividend_safety_score(info, fcf)

        # Insider Ownership %
        insider_ownership = safe_float(info.get("heldPercentInsiders"))

        # Radar chart data
        radar_data = build_radar_data(
            category_scores,
            piotroski_score,
            float(insider_signal.get("signal", 0)),
            tech_signals,
        )

        # v10.0: Auto-save snapshot to Supabase (once per ticker per session)
        _snap_key = f"_snapshot_saved_{ticker}"
        if not st.session_state.get(_snap_key):
            try:
                _pe_val = safe_float(info.get("trailingPE"))
                db.save_snapshot(
                    ticker=ticker,
                    current_price=current_price,
                    fair_value_dcf=fair_value_dcf,
                    scorecard=scorecard.get("total", 0) if isinstance(scorecard, dict) else 0,
                    mos_dcf=round((1 - current_price / fair_value_dcf) * 100, 1)
                        if fair_value_dcf and current_price and fair_value_dcf > 0 else None,
                    verdict=verdict,
                    piotroski=piotroski_score,
                    altman_z=altman_z,
                    pe=_pe_val,
                    insider_signal=float(insider_signal.get("signal", 0)),
                )
                st.session_state[_snap_key] = True
            except Exception:
                pass  # silent — snapshot is best-effort
    
    # ========================================================================
    # SMART HEADER (6 karet)
    # ========================================================================
    
    st.title(f"{company} ({ticker})")
    st.caption(f"📊 {sector} | Market Cap: {fmt_money(info.get('marketCap'), 0) if info.get('marketCap') else '—'}")

    # Crypto badge
    if _is_crypto:
        st.info("₿ **Kryptoměna** – DCF a fundamentální analýza nejsou relevantní. Zobrazena technická analýza a price history.", icon="💡")

    # Value Trap warning (nyní funkční)
    if is_value_trap:
        st.markdown(f'<div class="warning-box">{value_trap_msg}</div>', unsafe_allow_html=True)

    # Header cards row
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    
    with h1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Aktuální cena</div>
            <div class="metric-value">{fmt_money(current_price)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h2:
        analyst_price = analyst_target if analyst_target else None
        analyst_delta = f"+{((analyst_price/current_price - 1)*100):.1f}%" if analyst_price and current_price else "—"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Férovka (Analytici)</div>
            <div class="metric-value">{fmt_money(analyst_price)}</div>
            <div class="metric-delta" style="color: #00ff88;">{analyst_delta if analyst_price else ""}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h3:
        dcf_mos_str = f"{mos_dcf*100:+.1f}% MOS" if mos_dcf is not None else "—"
        dcf_color = "#00ff88" if mos_dcf and mos_dcf > 0 else "#ff4444"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Férovka (DCF)</div>
            <div class="metric-value">{fmt_money(fair_value_dcf)}</div>
            <div class="metric-delta" style="color: {dcf_color};">{dcf_mos_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h4:
        if ath and current_price:
            pct_from_ath = ((current_price / ath) - 1) * 100
            ath_str = f"{pct_from_ath:+.1f}%"
        else:
            ath_str = "—"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">ATH</div>
            <div class="metric-value">{fmt_money(ath)}</div>
            <div class="metric-delta">{ath_str} od vrcholu</div>
        </div>
        """, unsafe_allow_html=True)

    with h5:
        # Earnings countdown
        if earnings_countdown is not None and earnings_countdown >= 0:
            earn_color = "#ff8800" if earnings_countdown <= 14 else "#aaaaaa"
            earn_label = f"Za {earnings_countdown} dní" if earnings_countdown > 0 else "🔔 Dnes!"
        else:
            earn_color = "#aaaaaa"
            earn_label = "—"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">📅 Earnings</div>
            <div class="metric-value" style="font-size:1.1rem;">{next_earnings.strftime('%d.%m.%Y') if next_earnings else '—'}</div>
            <div class="metric-delta" style="color: {earn_color};">{earn_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h6:
        st.markdown(f"""
        <div class="glass-card" style="border: 2px solid {verdict_color};">
            <div class="metric-label">Sektor</div>
            <div class="metric-value" style="font-size: 1.1rem;">{sector[:18]}</div>
            <div class="metric-delta" style="color: {verdict_color}; font-weight: 700;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tabs = st.tabs([
        "📊 Overview",
        "🗓️ Market Watch",
        "🤖 AI Analyst",
        "🏢 Peer Comparison",
        "📋 Scorecard Pro",
        "💰 Valuace (DCF)",
        "📐 Tech. Analýza",
        "📝 Memo & Watchlist",
        "🐦 Social & Guru",
        "🔍 Screener",
        "💼 Portfolio"
    ])

    # Persistent tab tracking: always restore last active tab across re-renders
    _force_label = st.session_state.get("force_tab_label") or ""
    components.html(js_persist_tab(_force_label), height=0, width=0)
    if _force_label:
        st.session_state.force_tab_label = None

    
    # ------------------------------------------------------------------------
    # TAB 1: Overview
    # ------------------------------------------------------------------------
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Rychlý přehled</div>', unsafe_allow_html=True)
        
        # Two columns
        left, right = st.columns([1, 1])
        
        with left:
            st.markdown("#### 📌 Základní info")
            st.write(f"**Společnost:** {company}")
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Sektor:** {sector}")
            st.write(f"**Odvětví:** {info.get('industry', '—')}")
            st.write(f"**Země:** {info.get('country', '—')}")
            
            summary = info.get("longBusinessSummary", "")
            if summary:
                st.markdown("#### 📝 O společnosti")
                with st.expander("Zobrazit popis", expanded=False):
                    st.write(summary)
        
        with right:
            st.markdown("#### 💎 Klíčové metriky")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("P/E", fmt_num(metrics.get("pe").value if metrics.get("pe") else None), help=metric_help("P/E"))
                st.metric("ROE", fmt_pct(metrics.get("roe").value if metrics.get("roe") else None), help=metric_help("ROE"))
                st.metric("Op. Margin", fmt_pct(metrics.get("operating_margin").value if metrics.get("operating_margin") else None), help=metric_help("Op. Margin"))
            
            with m2:
                st.metric("FCF Yield", fmt_pct(metrics.get("fcf_yield").value if metrics.get("fcf_yield") else None), help=metric_help("FCF Yield"))
                st.metric("Debt/Equity", fmt_num(metrics.get("debt_to_equity").value if metrics.get("debt_to_equity") else None), help=metric_help("Debt/Equity"))
                st.metric("Rev. Growth", fmt_pct(metrics.get("revenue_growth").value if metrics.get("revenue_growth") else None), help=metric_help("Rev. Growth"))

            # Nové advanced metriky
            st.markdown("---")
            st.markdown("#### 🔬 Advanced Metriky")
            adv1, adv2, adv3, adv4 = st.columns(4)
            with adv1:
                pf_color = "normal" if piotroski_score >= 6 else ("inverse" if piotroski_score <= 3 else "off")
                st.metric("Piotroski F-Score", f"{piotroski_score}/9", help=metric_help("Piotroski"),
                          delta="Silná" if piotroski_score >= 6 else ("Slabá" if piotroski_score <= 3 else "Průměrná"),
                          delta_color=pf_color)
            with adv2:
                z_color = "normal" if altman_z and altman_z > 2.99 else ("inverse" if altman_z and altman_z < 1.81 else "off")
                st.metric("Altman Z-Score", fmt_num(altman_z), delta=(altman_zone.split(" ", 1)[-1] if (altman_zone and altman_zone[0] in "✅⚠️🚨ℹ️") else altman_zone) if altman_zone else None, delta_color=z_color, help=metric_help("Altman Z"))
            with adv3:
                st.metric("Graham Number", fmt_money(graham_number), help=metric_help("Graham Number"),
                          delta=f"{((current_price/graham_number-1)*100):+.1f}% vs cena" if graham_number and current_price else None,
                          delta_color="inverse" if graham_number and current_price and current_price > graham_number else "normal")
            with adv4:
                si_pct = short_interest * 100 if short_interest else None
                si_color = "inverse" if si_pct and si_pct > 10 else "normal"
                st.metric("Short Interest", f"{si_pct:.1f}%" if si_pct else "—", help=metric_help("Short Int."),
                          delta="Vysoký!" if si_pct and si_pct > 10 else None, delta_color=si_color)

            st.markdown("---")
            eq_col1, eq_col2 = st.columns(2)
            with eq_col1:
                st.metric("Earnings Quality (CFO/NI)", fmt_num(earnings_quality_ratio), help=metric_help("Earnings Q."))
                st.caption(earnings_quality_label)
            with eq_col2:
                roic_val_display = calculate_roic(info)
                st.metric("ROIC (approx.)", fmt_pct(roic_val_display), help=metric_help("ROIC"))

            # === NOVÉ v9.0 METRIKY ===
            st.markdown("---")
            st.markdown("#### 📐 Nové metriky v9.0")
            new_col1, new_col2, new_col3, new_col4 = st.columns(4)
            with new_col1:
                nd_color = "normal" if net_debt_ebitda is not None and net_debt_ebitda < 2 else "inverse"
                st.metric(
                    "Net Debt/EBITDA",
                    fmt_num(net_debt_ebitda) + "×" if net_debt_ebitda is not None else "—",
                    delta="Zdravé" if net_debt_ebitda is not None and net_debt_ebitda < 2 else ("Vysoké" if net_debt_ebitda is not None else None),
                    delta_color=nd_color,
                    help=metric_help("Net Debt/EBITDA")
                )
            with new_col2:
                if rule_of_40 is not None:
                    ro40_color = "normal" if rule_of_40 >= 40 else "inverse"
                    st.metric(
                        "Rule of 40",
                        f"{rule_of_40:.1f}%",
                        delta="✅ Nad 40" if rule_of_40 >= 40 else "⚠️ Pod 40",
                        delta_color=ro40_color,
                        help=metric_help("Rule of 40")
                    )
                else:
                    st.metric("Rule of 40", "—", help=metric_help("Rule of 40"))
            with new_col3:
                st.metric(
                    "FCF Margin",
                    fmt_pct(fcf_margin),
                    help=metric_help("FCF Margin")
                )
            with new_col4:
                if insider_ownership is not None:
                    st.metric(
                        "Insider Ownership",
                        f"{insider_ownership*100:.1f}%",
                        help=metric_help("Insider Ownership")
                    )
                else:
                    st.metric("Insider Ownership", "—", help=metric_help("Insider Ownership"))

            # Dividend Safety (pokud firma vyplácí dividendu)
            if div_safety_score > 0 or safe_float(info.get("dividendYield")):
                st.markdown("---")
                div_col1, div_col2 = st.columns(2)
                with div_col1:
                    div_colors = {5: "#00ff88", 4: "#00ff88", 3: "#ffaa00", 2: "#ff8800", 1: "#ff4444", 0: "#ff4444"}
                    st.metric(
                        "Dividend Safety Score",
                        f"{div_safety_score}/5",
                        help=metric_help("Dividend Safety")
                    )
                with div_col2:
                    st.caption(div_safety_label.split("\n")[0])

            with st.expander("🔧 Metrics debug", expanded=False):
                mdbg = st.session_state.get("metrics_enrich_debug", None)
                if isinstance(mdbg, dict):
                    fills = mdbg.get("fills") or {}
                    if fills:
                        st.caption("Filled metrics (key → source):")
                        st.json(fills)
                    steps = mdbg.get("steps")
                    if steps:
                        st.caption("Fetch steps / provider statuses:")
                        st.json(steps)

        
        # Price chart
        st.markdown("---")
        st.markdown("#### 📈 Cenový vývoj (1 rok)")
        if not price_history.empty:
            chart_data = price_history[["Close"]].copy()
            chart_data.columns = ["Cena"]
            st.line_chart(chart_data, use_container_width=True, height=400)
        else:
            st.info("Graf není k dispozici")
        
        # Insider signal
        st.markdown("---")
        st.markdown("#### 🔐 Insider Trading Signal")
        
        ins1, ins2, ins3 = st.columns(3)
        with ins1:
            st.metric(
                "Signal",
                f"{insider_signal.get('signal', 0):.0f}/100",
                delta=insider_signal.get('label', 'N/A')
            )
        with ins2:
            st.metric("Nákupy (6M)", insider_signal.get('recent_buys', 0), help="Počet open-market nákupů insiderů za posledních 6 měsíců. Nákupy insiderů jsou silnější signál než prodeje (insideři prodávají z mnoha důvodů, ale kupují jen když věří v růst).")
        with ins3:
            st.metric("Prodeje (6M)", insider_signal.get('recent_sells', 0), help="Počet open-market prodejů insiderů za posledních 6 měsíců. Samotné prodeje nejsou nutně negativní – insideři prodávají z daňových, osobních nebo diverzifikačních důvodů.")
        
        if insider_signal.get("cluster_buying"):
            st.markdown(
                '<div class="success-box">🔥 <b>Cluster Buying Detected</b> Více insiderů nakupuje ve stejném období.</div>',
                unsafe_allow_html=True
            )

        if insider_signal.get("cluster_selling"):
            st.markdown(
                '<div class="warning-box">🧊 <b>Cluster Selling Detected</b> Více insiderů prodává ve stejném období (negativní signál).</div>',
                unsafe_allow_html=True
            )
        
        for insight in insider_signal.get('insights', []):
            st.write(f"• {insight}")

        with st.expander("🔧 Insider debug", expanded=False):
            dbg = st.session_state.get("insider_debug", None)
            if dbg:
                # Compact table view first (easier to read than raw JSON)
                try:
                    attempts = dbg.get("attempts") if isinstance(dbg, dict) else None
                    if isinstance(attempts, list) and attempts:
                        rows = []
                        for a in attempts:
                            if not isinstance(a, dict):
                                continue
                            rows.append({
                                "Provider": a.get("provider") or a.get("provider_name") or "—",
                                "Endpoint": a.get("endpoint") or "—",
                                "Status": a.get("status_code") if a.get("status_code") is not None else a.get("status"),
                                "Items": a.get("items"),
                                "Error/Note": a.get("error") or a.get("note"),
                            })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    # Useful headline
                    if isinstance(dbg, dict):
                        cs = dbg.get("chosen_source")
                        if cs:
                            st.caption(f"Chosen source: **{cs}**")
                except Exception:
                    pass

                # Raw JSON for full details
                st.json(dbg)
            else:
                st.info("Debug info není k dispozici.")
    
    # ------------------------------------------------------------------------
    # TAB 2: Market Watch (Makro & Earnings Calendar)
    # ------------------------------------------------------------------------
    with tabs[1]:
        st.markdown('<div class="section-header">🗓️ Market Watch - Upcoming Events</div>', unsafe_allow_html=True)

        # ── Live Macro Dashboard (FRED API) ──────────────────────────────
        st.markdown("### 📡 Živý makro dashboard")
        live_macro = spp_macro.fetch_live_macro_indicators()
        if live_macro:
            # Market Regime
            regime = spp_macro.get_market_regime_macro()
            st.markdown(f"**Aktuální makro režim:** {regime}")

            # Indicators table
            macro_live_df = pd.DataFrame(live_macro)
            display_cols = ["Indikátor", "Hodnota", "Důležitost", "Popis"]
            st.dataframe(macro_live_df[display_cols], use_container_width=True, hide_index=True)

            # Yield Curve
            yc_data = spp_macro.get_yield_curve_data()
            yc_fig = spp_charts.build_yield_curve_chart(yc_data)
            if yc_fig:
                st.plotly_chart(yc_fig, use_container_width=True)
        else:
            st.info("💡 Pro živá makro data z FRED nastav `FRED_API_KEY` v secrets.toml")

        st.markdown("---")

        # ── Static Macro Calendar (fallback / upcoming events) ──────────
        st.markdown("### 🌍 Nadcházející makro události")
        macro_calendar = spp_macro.get_macro_calendar()
        macro_df = pd.DataFrame(macro_calendar)
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_df = macro_df[macro_df['date'] >= dt.datetime.now()]
        macro_df = macro_df.sort_values('date')

        if not macro_df.empty:
            def color_importance(val):
                if val == "Critical":
                    return 'background-color: #ff4444; color: white; font-weight: bold;'
                elif val == "High":
                    return 'background-color: #ff8800; color: white;'
                else:
                    return 'background-color: #ffaa00;'
            styled_df = macro_df.style.map(color_importance, subset=['importance'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 📊 Earnings Calendar")
        
        # next_earnings je nyní vypočítáno výše (earnings_countdown)
        if next_earnings:
            if earnings_countdown is not None and earnings_countdown == 0:
                st.error(f"🔔 **{ticker} DNES earnings!** {next_earnings.strftime('%d.%m.%Y')}")
            elif earnings_countdown is not None and earnings_countdown <= 7:
                st.warning(f"⏰ **{ticker} earnings za {earnings_countdown} dní:** {next_earnings.strftime('%d.%m.%Y')}")
            else:
                st.success(f"📅 **{ticker} očekávané earnings:** {next_earnings.strftime('%d.%m.%Y')} (za {earnings_countdown} dní)")
        
        # Show peer earnings too
        if auto_peers:
            st.markdown("#### Earnings konkurence")
            peer_earnings = []
            for peer in auto_peers[:3]:
                peer_info = fetch_ticker_info(peer)
                peer_date = get_earnings_calendar_estimate(peer, peer_info)
                if peer_date:
                    peer_earnings.append({
                        "Ticker": peer,
                        "Earnings Date": peer_date.strftime('%d.%m.%Y')
                    })
            
            if peer_earnings:
                st.dataframe(pd.DataFrame(peer_earnings), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.info("💡 **Tip:** Sleduj tyto události pro včasné rozhodnutí o entry/exit pointech!")
    
    # ------------------------------------------------------------------------
    # TAB 3: AI Analyst Report (Asymetrická verze)
    # ------------------------------------------------------------------------
    with tabs[2]:
        st.markdown('<div class="section-header">🤖 AI Analytik & Asymetrie</div>', unsafe_allow_html=True)
        
        # --- EDUKATIVNÍ LEGENDA ---
        with st.expander("ℹ️ Co znamenají tyto metriky?", expanded=False):
            st.markdown("""
            ### ⚖️ Asymmetry Score
            Měří tzv. **konvexitu** investice. Cílem je najít situace, kde je distribuce pravděpodobnosti "nakloněna" ve váš prospěch.
            * **Vysoké skóre (70+):** Downside je omezen (např. vysokou hotovostí, aktivy), zatímco upside je otevřený.
            * **Nízké skóre (0-30):** Riskujete 50 %, abyste vydělali 10 %. To je asymetrie, které se chceme vyhnout.

            ### 🥊 Red Team Attack
            Technika eliminace **konfirmačního zkreslení** (tendence hledat jen důkazy pro svůj názor). 
            AI v tomto modulu simuluje roli *Short Sellera* nebo agresivního oponenta. Pokud vaše investiční teze přežije 
            tento "útok" a rizika jsou akceptovatelná, je vaše rozhodnutí mnohem robustnější.
            """)
            
        if not GEMINI_API_KEY:
            st.warning("⚠️ **AI analýza není dostupná**")
            st.info("Nastav GEMINI_API_KEY v secrets pro aktivaci AI analytika.")
        else:
            # OPRAVENÉ TLAČÍTKO: Teď už skutečně volá funkci
            if st.button("🚀 Vygenerovat Asymetrický Report", use_container_width=True, type="primary"):
                st.session_state.force_tab_label = "🤖 AI Analyst"
                st.session_state.ai_report_ticker = None
                
                with st.spinner("🧠 Seniorní manažer analyzuje asymetrii trhu..."):
                    # Volání tvé retry funkce
                    ai_report = generate_ai_analyst_report_with_retry(
                        ticker=ticker,
                        company=company,
                        metrics=metrics,
                        info=info,
                        dcf_fair_value=fair_value_dcf,
                        current_price=current_price,
                        scorecard=scorecard,
                        macro_events=MACRO_CALENDAR,
                        insider_signal=insider_signal
                    )
                    
                    # Uložení výsledku do session_state
                    st.session_state['ai_report'] = ai_report
                    st.session_state.ai_report_ticker = ticker
                    st.session_state.force_tab_label = "🤖 AI Analyst"
                    st.rerun() # Refresh pro zobrazení výsledků

            # --- ZOBRAZENÍ VÝSLEDKŮ ---
            if 'ai_report' in st.session_state and st.session_state.ai_report_ticker == ticker:
                report = st.session_state['ai_report']

                # v10.0: Cache indicator
                if report.get("_from_cache"):
                    st.caption("⚡ Report načten z cache (Supabase, max 6h starý)")
                
                # 1. Gauge Chart (Ukazatel asymetrie)
                import plotly.graph_objects as go
                score = report.get("asymmetry_score", 50)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    title = {'text': "Asymmetry Score", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#00ff88" if score > 70 else "#ffaa00"},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(255, 68, 68, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(255, 170, 0, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(0, 255, 136, 0.2)"}
                        ],
                        'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': score}
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

                # 2. RED TEAM WARNING BOX
                st.markdown(f"""
                    <div style="background-color: rgba(255, 68, 68, 0.1); border: 2px solid #ff4444; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
                        <h3 style="color: #ff4444; margin-top: 0; font-size: 1.2rem;">🚨 RED TEAM ATTACK</h3>
                        <p style="font-style: italic; color: #ffcccc; margin-bottom: 0;">{report.get('red_team_warning', 'N/A')}</p>
                    </div>
                """, unsafe_allow_html=True)

                # 3. Bull & Bear Case Sloupce
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🐂 Bull Case (Upside)")
                    for item in report.get('bull_case', []):
                        st.write(f"✅ {item}")
                
                with col2:
                    st.markdown("### 🐻 Bear Case (Downside)")
                    for item in report.get('bear_case', []):
                        st.write(f"⚠️ {item}")

                # 4. Syntéza a detaily
                st.markdown("---")
                st.markdown(f"**🛡️ Fundamentální podlaha:** {report.get('fundamental_floor', 'N/A')}")
                st.info(f"**🎯 Strategická syntéza:** {report.get('reasoning_synthesis', 'N/A')}")

                # v10.0: Macro impact
                _macro_impact = report.get("macro_impact")
                if _macro_impact:
                    st.warning(f"🌍 **Makro vliv:** {_macro_impact}")
                # Spodní řada metrik
                v_col1, v_col2, v_col3 = st.columns(3)
                with v_col1:
                    verdict = report.get('verdict', 'N/A')
                    st.metric("Finální verdikt", verdict)
                with v_col2:
                    st.metric("Risk/Reward Ratio", report.get('risk_reward_ratio', 'N/A'))
                with v_col3:
                    st.metric("Confidence", report.get('confidence', 'N/A'))
    # ------------------------------------------------------------------------
    # TAB 4: Peer Comparison
    # ------------------------------------------------------------------------
    with tabs[3]:
        st.markdown('<div class="section-header">🏢 Srovnání s konkurencí</div>', unsafe_allow_html=True)
        
        if not auto_peers:
            st.info(f"📊 **{ticker}** - Aktuálně bez přímé srovnatelné konkurence v databázi.")
            st.caption("Přidej manuálně do SECTOR_PEERS slovníku v kódu pro zobrazení peer analýzy.")
        else:
            st.success(f"🔍 Nalezeno {len(auto_peers)} konkurentů: {', '.join(auto_peers)}")
            
            with st.spinner("Načítám data konkurence..."):
                peer_df = fetch_peer_comparison(ticker, auto_peers)
            
            if not peer_df.empty:
                # Format for display
                display_df = peer_df.copy()
                display_df['P/E'] = display_df['P/E'].apply(lambda x: fmt_num(x))
                display_df['Op. Margin'] = display_df['Op. Margin'].apply(lambda x: fmt_pct(x))
                display_df['Rev. Growth'] = display_df['Rev. Growth'].apply(lambda x: fmt_pct(x))
                display_df['FCF Yield'] = display_df['FCF Yield'].apply(lambda x: fmt_pct(x))
                display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: fmt_money(x, 0, "$") if x else "—")
                # OPRAVA: formátovat ROE a Gross Margin jako procenta
                if 'ROE' in display_df.columns:
                    display_df['ROE'] = display_df['ROE'].apply(lambda x: fmt_pct(x))
                if 'Gross Margin' in display_df.columns:
                    display_df['Gross Margin'] = display_df['Gross Margin'].apply(lambda x: fmt_pct(x))
                
                # Highlight main ticker
                def highlight_ticker(row):
                    if row['Ticker'] == ticker:
                        return ['background-color: #00ff8820'] * len(row)
                    return [''] * len(row)
                
                styled = display_df.style.apply(highlight_ticker, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)
                
                # Insights
                st.markdown("#### 📊 Relativní pozice")
                
                # Calculate percentiles
                if len(peer_df) > 1:
                    main_row = peer_df[peer_df['Ticker'] == ticker].iloc[0] if ticker in peer_df['Ticker'].values else None
                    
                    if main_row is not None:
                        insights = []
                        
                        # P/E comparison
                        pe_val = main_row['P/E']
                        if pd.notna(pe_val):
                            pe_rank = (peer_df['P/E'] < pe_val).sum() + 1
                            total = peer_df['P/E'].notna().sum()
                            if pe_rank <= total * 0.33:
                                insights.append(f"✅ P/E je v dolní třetině (levnější valuace než většina konkurence)")
                            elif pe_rank >= total * 0.67:
                                insights.append(f"⚠️ P/E je v horní třetině (dražší valuace)")
                        
                        # Revenue growth
                        rg_val = main_row['Rev. Growth']
                        if pd.notna(rg_val):
                            rg_rank = (peer_df['Rev. Growth'] > rg_val).sum() + 1
                            total = peer_df['Rev. Growth'].notna().sum()
                            if rg_rank <= total * 0.33:
                                insights.append(f"🚀 Revenue growth v TOP třetině (roste rychleji než konkurence)")
                        
                        for insight in insights:
                            st.write(f"• {insight}")
            else:
                st.warning("Nepodařilo se načíst data konkurence")
    
    # ------------------------------------------------------------------------
    # TAB 5: Scorecard Pro
    # ------------------------------------------------------------------------
    with tabs[4]:
        st.markdown('<div class="section-header">📋 Investiční Scorecard Pro</div>', unsafe_allow_html=True)
        
        # Overall score
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; border: 3px solid {verdict_color}; border-radius: 15px; background: rgba(255,255,255,0.03);">
            <div style="font-size: 1rem; opacity: 0.8;">CELKOVÉ SKÓRE</div>
            <div style="font-size: 4rem; font-weight: 900; color: {verdict_color};">{scorecard:.0f}<span style="font-size: 2rem; opacity: 0.6;">/100</span></div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-top: 10px;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category breakdown
        st.markdown("### 📊 Rozpad podle kategorií")
        
        cat_cols = st.columns(len(category_scores))
        for idx, (cat_name, cat_score) in enumerate(category_scores.items()):
            with cat_cols[idx]:
                cat_color = "#00ff88" if cat_score >= 70 else ("#ffaa00" if cat_score >= 50 else "#ff4444")
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {cat_color}; border-radius: 10px;">
                    <div style="font-size: 0.9rem; opacity: 0.8;">{cat_name}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: {cat_color};">{cat_score:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Warnings from verdict
        if verdict_warnings:
            st.markdown("### ⚠️ Důležitá upozornění")
            for warning in verdict_warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
        
        # === RADAR CHART (nové v9.0) ===
        st.markdown("---")
        st.markdown("### 🕸️ Radar Chart – 7 dimenzí analýzy")
        try:
            import plotly.graph_objects as go
            radar_categories = list(radar_data.keys())
            radar_values = list(radar_data.values())
            radar_categories_closed = radar_categories + [radar_categories[0]]
            radar_values_closed = radar_values + [radar_values[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values_closed,
                theta=radar_categories_closed,
                fill="toself",
                fillcolor="rgba(0, 255, 136, 0.15)",
                line=dict(color="#00ff88", width=2),
                name="Analýza"
            ))
            ref_vals = [50] * len(radar_categories_closed)
            fig_radar.add_trace(go.Scatterpolar(
                r=ref_vals,
                theta=radar_categories_closed,
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                name="Průměr trhu (50)",
                showlegend=True
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="white", size=9)),
                    angularaxis=dict(tickfont=dict(color="white", size=11)),
                    bgcolor="rgba(0,0,0,0)"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=True,
                height=420,
                margin=dict(l=60, r=60, t=40, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("📌 Každá osa: 0 = nejhorší, 100 = nejlepší. Referenční linie = průměrná hodnota (50).")
        except Exception as e:
            st.info(f"Radar chart není dostupný: {e}")

        st.markdown("---")
        
        # Individual metrics
        st.markdown("### 🔍 Detailní metriky")
        
        if individual_scores:
            # Klíč → hezký název → tooltip
            _metric_tooltip_map = {
                "pe": ("P/E", metric_help("P/E")),
                "pb": ("P/B", metric_help("P/B")),
                "ps": ("P/S", metric_help("P/S")),
                "peg": ("PEG", metric_help("PEG")),
                "ev_ebitda": ("EV/EBITDA", metric_help("EV/EBITDA")),
                "roe": ("ROE", metric_help("ROE")),
                "roa": ("ROA", metric_help("ROA")),
                "operating_margin": ("Op. Marže", metric_help("Op. Margin")),
                "profit_margin": ("Čistá Marže", metric_help("Profit Margin")),
                "gross_margin": ("Hrubá Marže", metric_help("Gross Margin")),
                "revenue_growth": ("Růst Tržeb", metric_help("Rev. Growth")),
                "earnings_growth": ("Růst EPS", metric_help("EPS Growth")),
                "current_ratio": ("Current Ratio", metric_help("Current Ratio")),
                "quick_ratio": ("Quick Ratio", metric_help("Quick Ratio")),
                "debt_to_equity": ("Dluh/Vlastní kap.", metric_help("D/E")),
                "fcf_yield": ("FCF Yield", metric_help("FCF Yield")),
            }
            metric_rows = []
            for key, metric in metrics.items():
                for name, score in individual_scores.items():
                    if metric.name == name:
                        nice_name, tip = _metric_tooltip_map.get(key, (name, None))
                        if key in ["pe", "pb", "ps", "peg", "current_ratio", "quick_ratio", "debt_to_equity"]:
                            val_str = fmt_num(metric.value)
                        elif key in ["roe", "roa", "operating_margin", "profit_margin", "gross_margin",
                                     "revenue_growth", "earnings_growth", "fcf_yield"]:
                            val_str = fmt_pct(metric.value)
                        else:
                            val_str = fmt_num(metric.value)
                        score_bar = "█" * int(score) + "░" * (10 - int(score))
                        metric_rows.append({
                            "Metrika": nice_name,
                            "Hodnota": val_str,
                            "Skóre": f"{score:.1f}/10",
                            "Vizuál": score_bar,
                            "Zdroj": metric.source or "yfinance",
                            "ℹ️ Popis": tip[:80] + "…" if tip and len(tip) > 80 else (tip or ""),
                        })
            metric_df = pd.DataFrame(metric_rows)
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

            # === CSV EXPORT (nové v9.0) ===
            try:
                _export_df = metric_df[["Metrika", "Hodnota", "Skóre", "Zdroj"]].copy()
                _export_df.insert(0, "Ticker", ticker)
                _export_df.insert(1, "Datum", dt.datetime.now().strftime("%Y-%m-%d"))
                _csv = _export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Exportovat metriky (CSV)",
                    data=_csv,
                    file_name=f"{ticker}_metriky_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception:
                pass

        # Piotroski F-Score breakdown
        st.markdown("---")
        st.markdown("### 🔬 Piotroski F-Score breakdown")
        if piotroski_breakdown:
            pf_col1, pf_col2 = st.columns([1, 1])
            with pf_col1:
                pf_color_main = "#00ff88" if piotroski_score >= 6 else ("#ffaa00" if piotroski_score >= 4 else "#ff4444")
                pf_label = "Silné fundamenty" if piotroski_score >= 6 else ("Průměrné" if piotroski_score >= 4 else "Slabé fundamenty")
                st.markdown(f"""
                <div style="text-align:center; padding:20px; border: 2px solid {pf_color_main}; border-radius:10px;">
                    <div style="font-size:0.9rem; opacity:0.8;">Piotroski F-Score</div>
                    <div style="font-size:3rem; font-weight:900; color:{pf_color_main};">{piotroski_score}<span style="font-size:1.5rem; opacity:0.6;">/9</span></div>
                    <div style="color:{pf_color_main};">{pf_label}</div>
                </div>
                """, unsafe_allow_html=True)
            with pf_col2:
                for criterion, val in piotroski_breakdown.items():
                    icon = "✅" if val == 1 else "❌"
                    st.write(f"{icon} {criterion}")

        # Altman Z-Score
        st.markdown("---")
        st.markdown("### 🏥 Altman Z-Score (Bankruptcy Risk)")
        if altman_z is not None:
            az_color = "#00ff88" if altman_z > 2.99 else ("#ffaa00" if altman_z > 1.81 else "#ff4444")
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid {az_color};">
                <div class="metric-label">Z-Score</div>
                <div class="metric-value" style="color: {az_color};">{altman_z}</div>
                <div class="metric-delta">{altman_zone}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #444;">
                <div class="metric-label">Z-Score</div>
                <div class="metric-value" style="color: #aaa;">—</div>
                <div class="metric-delta">{altman_zone or "Data nedostupná"}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Z > 2.99 = Bezpečná | 1.81 – 2.99 = Šedá zóna | Z < 1.81 = Riziko bankrotu")
# ------------------------------------------------------------------------
    # TAB 6: DCF Valuation
    # ------------------------------------------------------------------------
    with tabs[5]:
        st.markdown('<div class="section-header">💰 DCF Valuace & Reverse DCF</div>', unsafe_allow_html=True)
        # Gating: DCF jen pro EQUITY (ETF/CRYPTO/INDEX/FUND/FX -> N/A)
        if not supports_dcf:
            na_box("DCF valuace", f"tento ticker je {asset_flags.get('asset_class')} ({asset_flags.get('quoteType')})",
                   "DCF se dává smysl jen u akcií (EQUITY). Pro ETF/Index/Krypto použij technickou analýzu a price history.")
        else:
            # Confidence & Data Quality
            _conf = st.session_state.get("dcf_confidence")
            _reasons = st.session_state.get("dcf_confidence_reasons", [])
            if _conf is not None:
                st.markdown(f"**DCF Confidence:** {_conf}/100 {qmark('Skóre kvality vstupních dat. Penalizace: odhad shares, OCF proxy, cash/debt missing, currency mismatch, outlier flagy.')}", unsafe_allow_html=True)
                with st.expander("📌 Data Quality & Confidence – detaily", expanded=False):
                    st.json(st.session_state.get("_diag_sources", {}).get("data_quality", {}))
                    if _reasons:
                        st.markdown("**Důvody penalizace:**")
                        for r in _reasons:
                            st.write(f"- {r}")
            st.markdown("---")

            # Scenario Builder (Bull / Base / Bear)
            section_title("🎛️ Scenario Builder (Bull / Base / Bear)", "Rychlé porovnání tří scénářů bez ručního ladění sliderů. Používá stejný cash/debt bridge a Exit Multiple jako hlavní DCF.", level=3)
            if dcf_fcf_used and shares and dcf_fcf_used > 0 and shares > 0 and current_price:
                base = {"growth": float(used_dcf_growth), "wacc": float(used_dcf_wacc), "exit": float(used_exit_multiple)}
                bull = {"growth": min(0.50, base["growth"] * 1.25), "wacc": max(0.05, base["wacc"] * 0.9), "exit": min(60.0, base["exit"] * 1.15)}
                bear = {"growth": max(-0.10, base["growth"] * 0.75), "wacc": min(0.25, base["wacc"] * 1.1), "exit": max(5.0, base["exit"] * 0.85)}

                def _fv(g, w, ex):
                    pv_sum = 0.0
                    cf = float(dcf_fcf_used)
                    for year in range(1, int(dcf_years) + 1):
                        cf *= (1 + g)
                        pv_sum += cf / ((1 + w) ** year)
                    tv = cf * ex
                    pv_tv = tv / ((1 + w) ** int(dcf_years))
                    ev = pv_sum + pv_tv
                    return (ev + total_cash - total_debt) / float(shares)

                rows = []
                for name, s in [("Bear", bear), ("Base", base), ("Bull", bull)]:
                    fv = _fv(s["growth"], s["wacc"], s["exit"])
                    mos = (fv / current_price) - 1.0 if fv and current_price else None
                    ig = reverse_exit_multiple_implied_growth(current_price, float(dcf_fcf_used), float(s["wacc"]), int(dcf_years), float(shares), float(total_cash), float(total_debt), float(s["exit"]))
                    rows.append({
                        "Scénář": name,
                        "Growth": f"{s['growth']*100:.1f}%",
                        "WACC": f"{s['wacc']*100:.1f}%",
                        "Exit": f"{s['exit']:.1f}×",
                        "Fair Value": fmt_money(fv),
                        "MOS": f"{mos*100:+.1f}%" if mos is not None else "—",
                        "Implied Growth": f"{ig*100:.1f}%" if ig is not None else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                na_box("Scenario Builder", "chybí kladný FCF/OCF proxy, počet akcií nebo aktuální cena",
                       "Pro scénáře potřebujeme FCF/OCF proxy > 0, shares > 0 a cenu.")
            st.markdown("---")


        st.markdown(f"**Použité parametry pro DCF:** Růst {used_dcf_growth*100:.1f}% ({used_mode_label}) | WACC {used_dcf_wacc*100:.1f}% ({used_mode_label}) | Exit multiple {used_exit_multiple:.1f}× ({used_mode_label}) {qmark('Smart = automatický odhad z metrik; Manual = hodnoty z panelu „DCF Parametry“ v levém menu. Exit multiple je terminální metoda používaná v hlavním DCF. DCF používá FCF TTM; když FCF chybí nebo je nelogické, použije se konzervativní proxy 60 % Operating Cash Flow (OCF).')}", unsafe_allow_html=True)
        
        if dcf_fcf_used and shares and dcf_fcf_used > 0:
            # Main DCF results
            dcf_col1, dcf_col2, dcf_col3, dcf_col4 = st.columns(4)
            
            with dcf_col1:
                st.metric("Férová hodnota (DCF)", fmt_money(fair_value_dcf), help=metric_help("DCF"))
            with dcf_col2:
                st.metric("Aktuální cena", fmt_money(current_price))
            with dcf_col3:
                mos_str = f"{mos_dcf*100:+.1f}%" if mos_dcf is not None else "—"
                mos_color_delta = mos_str if mos_dcf else None
                st.metric("Margin of Safety", mos_str, delta=mos_color_delta, help=metric_help("MOS"))
            with dcf_col4:
                if implied_growth is not None:
                    st.metric("Implied Growth (Reverse DCF)", f"{implied_growth*100:.1f}%", help=metric_help("Implied Growth"))
                else:
                    st.metric("Implied Growth", "—")
            
            st.markdown("---")
            
            # Sensitivity analysis
            section_title("📊 Sensitivity Analysis", "Jak moc se férová cena změní, když upravíš 1 parametr (růst nebo WACC) a ostatní necháš stejné. Počítá se se stejným cash/debt bridge a Exit Multiple jako hlavní DCF.", level=3)
            st.caption("Výpočet zahrnuje stejnou úpravu o cash/dluh jako hlavní DCF model.")
            
            sens_col1, sens_col2 = st.columns(2)
            
            with sens_col1:
                st.markdown("**🔼 Růst FCF Impact**")
                growth_rates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
                sens_data = []
                for g in growth_rates:
                    # OPRAVA: přidáme cash/debt stejně jako v hlavním DCF výpočtu
                    pv_sum = 0.0
                    cf = float(dcf_fcf_used)
                    for year in range(1, dcf_years + 1):
                        cf *= (1 + g)
                        pv_sum += cf / ((1 + used_dcf_wacc) ** year)
                    tv = cf * used_exit_multiple
                    pv_tv = tv / ((1 + used_dcf_wacc) ** dcf_years)
                    ev = pv_sum + pv_tv
                    fv = (ev + total_cash - total_debt) / shares if shares and shares > 0 else None
                    upside = ((fv / current_price) - 1) * 100 if fv and current_price else None
                    sens_data.append({
                        "Růst": f"{g*100:.0f}%",
                        "Fair Value": fmt_money(fv),
                        "Upside": f"{upside:+.1f}%" if upside is not None else "—"
                    })
                st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)
            
            with sens_col2:
                st.markdown("**💹 WACC Impact**")
                wacc_rates = [0.08, 0.09, 0.10, 0.11, 0.12, 0.15]
                wacc_data = []
                for w in wacc_rates:
                    # OPRAVA: exit multiple metoda + cash/debt bridge (konzistentní s hlavním DCF)
                    pv_sum = 0.0
                    cf = float(dcf_fcf_used)
                    for year in range(1, dcf_years + 1):
                        cf *= (1 + used_dcf_growth)
                        pv_sum += cf / ((1 + w) ** year)
                    tv = cf * used_exit_multiple
                    pv_tv = tv / ((1 + w) ** dcf_years)
                    ev = pv_sum + pv_tv
                    fv = (ev + total_cash - total_debt) / shares if shares and shares > 0 else None
                    upside = ((fv / current_price) - 1) * 100 if fv and current_price else None
                    wacc_data.append({
                        "WACC": f"{w*100:.0f}%",
                        "Fair Value": fmt_money(fv),
                        "Upside": f"{upside:+.1f}%" if upside is not None else "—"
                    })
                st.dataframe(pd.DataFrame(wacc_data), use_container_width=True, hide_index=True)
            
            # Interpretation
            

            st.markdown("---")
            section_title(
                "🧠 Interpretace",
                "Shrnutí toho, co DCF říká vs. aktuální cena. Vychází z MOS a z Reverse DCF (Implied Growth). Pozor: DCF je citlivé na vstupy – ber to jako interval, ne jako přesné číslo.",
                level=3,
            )

            # 1) MOS (fair value vs cena)
            if fair_value_dcf is None or current_price is None or mos_dcf is None:
                st.markdown(
                    f"• Interpretace je omezená – chybí DCF férovka / aktuální cena / MOS. {qmark('MOS = (férovka / cena) - 1. Kladné číslo znamená, že je akcie pod férovkou (polštář).')}",
                    unsafe_allow_html=True,
                )
            else:
                if mos_dcf > 0.25:
                    st.success(f"✅ **Podhodnocené podle DCF:** MOS {mos_dcf*100:+.1f}% (velký polštář).")
                elif mos_dcf > 0.05:
                    st.info(f"🟢 **Lehce pod férovkou:** MOS {mos_dcf*100:+.1f}% (menší polštář).")
                elif mos_dcf >= -0.05:
                    st.info(f"⚖️ **Zhruba férové:** MOS {mos_dcf*100:+.1f}% (v rámci šumu modelu).")
                else:
                    st.warning(f"⚠️ **Nadhodnocené podle DCF:** MOS {mos_dcf*100:+.1f}% (trh platí prémii).")

            # 2) Poznámka k použitému cash-flow
            try:
                if fcf and dcf_fcf_used and fcf > 0 and abs(dcf_fcf_used - fcf) / max(abs(fcf), 1.0) > 0.25:
                    st.markdown(
                        f"• Pro DCF byl použit **upravený cash-flow proxy** (OCF-based), protože reportované FCF je velmi nízké vůči Operating Cash Flow. {qmark('Typicky „Amazon style“: firma silně reinvestuje. V takovém případě může být DCF z čistého FCF zavádějící.')}",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

            # 3) Reverse DCF (Implied Growth)
            if implied_growth is None:
                st.markdown(
                    f"• **Implied Growth (Reverse DCF)** není dostupný. {qmark('Počítá se růst FCF, při kterém by DCF (se stejným WACC a exit multiple) vyšel přesně na aktuální cenu. Potřebuje kladný FCF/OCF proxy, počet akcií a cenu.')}",
                    unsafe_allow_html=True,
                )
            else:
                diff = implied_growth - used_dcf_growth

                # klasifikace očekávání trhu
                if implied_growth < 0:
                    st.warning(f"📉 Trh implikuje **pokles FCF** ({implied_growth*100:.1f}% ročně).")
                elif implied_growth < 0.05:
                    st.info(f"📊 Trh implikuje **nízký růst** ({implied_growth*100:.1f}% ročně).")
                elif implied_growth < 0.15:
                    st.success(f"✅ Trh implikuje **zdravý růst** ({implied_growth*100:.1f}% ročně).")
                else:
                    st.warning(f"🚀 Trh implikuje **agresivní růst** ({implied_growth*100:.1f}% ročně) – riziko zklamání.")

                # porovnání s tvým modelem
                if abs(diff) >= 0.05:
                    direction = "vyšší" if diff > 0 else "nižší"
                    st.markdown(
                        f"• Oproti tvému modelu je implied growth o **{abs(diff)*100:.1f} p.b. {direction}** (model: {used_dcf_growth*100:.1f}% / trh: {implied_growth*100:.1f}%). {qmark('Když trh implikuje výrazně vyšší růst než model, akcie bývá „priced for perfection“. Naopak výrazně nižší implied growth může znamenat příležitost – nebo reálné problémy.')}",
                        unsafe_allow_html=True,
                    )



            # Monte Carlo DCF
            st.markdown("---")
            section_title("🎲 Monte Carlo DCF Simulace (1 000 scénářů)", "Simulace generuje 1 000 scénářů a náhodně rozptyluje růst (±30 %), WACC (±15 %) a exit multiple (±20 %). Výsledek je distribuce férové ceny (percentily).", level=3)
            if mc_dcf:
                import plotly.graph_objects as go
                mc_col1, mc_col2, mc_col3 = st.columns(3)
                with mc_col1:
                    st.metric("P10 (pesimistický)", fmt_money(mc_dcf.get("p10")), help=metric_help("P10/P90"))
                    st.metric("Medián", fmt_money(mc_dcf.get("median")))
                with mc_col2:
                    st.metric("Průměr", fmt_money(mc_dcf.get("mean")))
                    st.metric("P90 (optimistický)", fmt_money(mc_dcf.get("p90")), help=metric_help("P10/P90"))
                with mc_col3:
                    prob_upside = None
                    if current_price and mc_dcf.get("mean"):
                        # Crude estimate of probability of being undervalued
                        mean_fv = mc_dcf["mean"]
                        std_fv = mc_dcf.get("std", mean_fv * 0.3)
                        if std_fv > 0:
                            try:
                                from scipy import stats as scipy_stats
                                prob_upside = float(scipy_stats.norm.sf(current_price, mean_fv, std_fv)) * 100
                            except Exception:
                                # fallback (bez SciPy): hrubý odhad z normalizované vzdálenosti od mean
                                prob_upside = 100 * max(0, min(1, (mean_fv - current_price) / (2 * std_fv) + 0.5))
                    st.metric("Pravděp. undervalued", f"{prob_upside:.0f}%" if prob_upside is not None else "—")
                    st.metric("Simulací", mc_dcf.get("n", 0))

                st.caption("💡 Monte Carlo rozptyluje growth (±30 %), WACC (±15 %) a exit multiple (±20 %). P10/P90 = 10./90. percentil všech scénářů. Model používá stejný cash/debt bridge jako hlavní DCF.")
            else:
                st.info("Monte Carlo není dostupné. Důvody: chybí kladný FCF/OCF proxy nebo počet akcií, případně se nevygeneroval žádný validní scénář (zkus upravit parametry).")

            # Investment Simulator
            st.markdown("---")
            section_title("💰 Co kdybych investoval X Kč?", "Historická simulace: kdybys investoval před N lety a držel do dneška. Není to predikce budoucnosti.", level=3)
            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                sim_amount = st.number_input("Investovaná částka (Kč)", min_value=1000, max_value=10_000_000,
                                              value=100_000, step=10_000, key="sim_amount")
                sim_years = st.selectbox("Investiční horizont", [1, 2, 3, 5, 10], index=2, key="sim_years")
                if st.button("▶️ Spustit simulaci", use_container_width=True, key="btn_sim"):
                    with st.spinner("Načítám historická data..."):
                        sim_result = simulate_investment(ticker, sim_amount, sim_years)
                    st.session_state["sim_result"] = sim_result

            with sim_col2:
                sim_result = st.session_state.get("sim_result")
                if sim_result:
                    profit = sim_result["final"] - sim_result["initial"]
                    ret_color = "#00ff88" if profit >= 0 else "#ff4444"
                    st.markdown(f"""
                    <div class="metric-card" style="border: 2px solid {ret_color};">
                        <div class="metric-label">Výsledek za {sim_result['years']} let ({sim_result['start_date']} → {sim_result['end_date']})</div>
                        <div class="metric-value" style="color: {ret_color};">{sim_result['final']:,.0f} Kč</div>
                        <div class="metric-delta">Zisk/Ztráta: {profit:+,.0f} Kč ({sim_result['stock_return']*100:+.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if sim_result.get("spy_return") is not None:
                        spy_final = sim_result["initial"] * (1 + sim_result["spy_return"])
                        spy_profit = spy_final - sim_result["initial"]
                        spy_color = "#00ff88" if spy_profit >= 0 else "#ff4444"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">📊 vs. S&P 500 (SPY)</div>
                            <div class="metric-value" style="color: {spy_color};">{spy_final:,.0f} Kč</div>
                            <div class="metric-delta">SPY: {sim_result['spy_return']*100:+.1f}% | Outperformance: {(sim_result['stock_return']-sim_result['spy_return'])*100:+.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Zadej částku a klikni na 'Spustit simulaci'")
        
        else:
            na_box("DCF valuace", "chybí kladný FCF/OCF proxy nebo počet akcií", "DCF potřebuje FCF/OCF proxy > 0 a shares > 0.")
    
    # ------------------------------------------------------------------------
    # TAB 7: Technická Analýza
    # ------------------------------------------------------------------------
    with tabs[6]:
        st.markdown('<div class="section-header">📐 Technická Analýza</div>', unsafe_allow_html=True)

        if not tech_signals:
            st.warning("Nedostatek cenových dat pro technickou analýzu.")
        else:
            import plotly.graph_objects as go

            cp = tech_signals.get("current_price", 0)

            # --- RSI ---
            ta1, ta2, ta3 = st.columns(3)
            rsi_val = tech_signals.get("rsi")
            with ta1:
                if rsi_val is not None:
                    rsi_color = "#ff4444" if rsi_val > 70 else ("#00ff88" if rsi_val < 30 else "#ffaa00")
                    rsi_label = "🔴 Překoupeno" if rsi_val > 70 else ("🟢 Přeprodáno" if rsi_val < 30 else "🟡 Neutrální")
                    st.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_label, help=metric_help("RSI"))
                else:
                    st.metric("RSI (14)", "—", help=metric_help("RSI"))

            # --- MACD ---
            with ta2:
                st.metric("MACD signal", tech_signals.get("macd_label", "—"), help=metric_help("MACD"))

            # --- MA200 ---
            with ta3:
                pct_ma200 = tech_signals.get("pct_from_ma200")
                if pct_ma200 is not None:
                    ma200_color = "normal" if pct_ma200 > 0 else "inverse"
                    st.metric("vs. MA200", f"{pct_ma200*100:+.1f}%", delta_color=ma200_color, help=metric_help("MA50/MA200"))
                else:
                    st.metric("vs. MA200", "—")

            st.markdown("---")

            # --- 52W Range + Bollinger Bands ---
            st.markdown("### 📊 Cenový kontext")
            ta4, ta5 = st.columns(2)
            with ta4:
                high_52w = tech_signals.get("high_52w")
                low_52w = tech_signals.get("low_52w")
                if high_52w and low_52w and cp:
                    st.markdown(f"**52W High:** {fmt_money(high_52w)} ({((cp/high_52w-1)*100):+.1f}%)")
                    st.markdown(f"**52W Low:** {fmt_money(low_52w)} ({((cp/low_52w-1)*100):+.1f}%)")

                    # Range bar visualization
                    if high_52w > low_52w:
                        pos = (cp - low_52w) / (high_52w - low_52w)
                        fig_range = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=pos * 100,
                            title={"text": "Pozice v 52W rozsahu"},
                            number={"suffix": "%"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#00ff88" if pos < 0.4 else ("#ffaa00" if pos < 0.7 else "#ff4444")},
                                "steps": [
                                    {"range": [0, 30], "color": "rgba(0,255,136,0.15)"},
                                    {"range": [30, 70], "color": "rgba(255,170,0,0.1)"},
                                    {"range": [70, 100], "color": "rgba(255,68,68,0.15)"},
                                ]
                            }
                        ))
                        fig_range.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10),
                                                paper_bgcolor='rgba(0,0,0,0)', font={"color": "white"})
                        st.plotly_chart(fig_range, use_container_width=True)

            with ta5:
                bb_upper = tech_signals.get("bb_upper")
                bb_lower = tech_signals.get("bb_lower")
                bb_mid = tech_signals.get("bb_mid")
                if bb_upper and bb_lower and cp:
                    bb_pos = "nad horním pásmem 🔴" if cp > bb_upper else ("pod dolním pásmem 🟢" if cp < bb_lower else "uvnitř pásem 🟡")
                    st.markdown("**Bollinger Bands (20d)**")
                    st.markdown(f"Horní: {fmt_money(bb_upper)} | Střed: {fmt_money(bb_mid)} | Dolní: {fmt_money(bb_lower)}")
                    st.markdown(f"Cena je: **{bb_pos}**")

                ma50 = tech_signals.get("ma50")
                ma200 = tech_signals.get("ma200")
                st.markdown("**Moving Averages**")
                if ma50:
                    col = "#00ff88" if cp >= ma50 else "#ff4444"
                    st.markdown(f"MA50: {fmt_money(ma50)} {'✅ nad' if cp >= ma50 else '❌ pod'}")
                if ma200:
                    st.markdown(f"MA200: {fmt_money(ma200)} {'✅ nad' if cp >= ma200 else '❌ pod'}")
                if ma50 and ma200:
                    if ma50 > ma200:
                        st.success("📈 Golden Cross aktivní (MA50 > MA200)")
                    else:
                        st.error("📉 Death Cross aktivní (MA50 < MA200)")

            st.markdown("---")

            # --- Cenový chart s MA50/MA200 ---
            st.markdown("### 📈 Cenový vývoj s indikátory")
            if not price_history_1y.empty and "Close" in price_history_1y.columns:
                close_s = price_history_1y["Close"].dropna()
                fig_ta = go.Figure()
                fig_ta.add_trace(go.Scatter(x=price_history_1y.index, y=close_s, name="Cena", line=dict(color="#4fc3f7", width=2)))

                if len(close_s) >= 50:
                    ma50_s = close_s.rolling(50).mean()
                    fig_ta.add_trace(go.Scatter(x=price_history_1y.index, y=ma50_s, name="MA50",
                                                line=dict(color="#ffaa00", width=1, dash="dash")))
                if len(close_s) >= 200:
                    ma200_s = close_s.rolling(200).mean()
                    fig_ta.add_trace(go.Scatter(x=price_history_1y.index, y=ma200_s, name="MA200",
                                                line=dict(color="#ff4444", width=1, dash="dot")))
                if bb_upper and bb_lower:
                    ma20_s = close_s.rolling(20).mean()
                    std20_s = close_s.rolling(20).std()
                    fig_ta.add_trace(go.Scatter(x=price_history_1y.index, y=(ma20_s + 2*std20_s),
                                                name="BB Upper", line=dict(color="rgba(150,150,255,0.5)", width=1), fill=None))
                    fig_ta.add_trace(go.Scatter(x=price_history_1y.index, y=(ma20_s - 2*std20_s),
                                                name="BB Lower", line=dict(color="rgba(150,150,255,0.5)", width=1),
                                                fill="tonexty", fillcolor="rgba(150,150,255,0.05)"))

                fig_ta.update_layout(
                    height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={"color": "white"}, legend=dict(orientation="h"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
                )
                st.plotly_chart(fig_ta, use_container_width=True)

            # --- Volume trend ---
            vol_trend = tech_signals.get("vol_trend")
            if vol_trend is not None:
                vt_label = f"Objem (20d vs 60d průměr): {vol_trend*100:+.1f}%"
                if vol_trend > 0.2:
                    st.success(f"📶 Zvýšený zájem: {vt_label}")
                elif vol_trend < -0.2:
                    st.warning(f"📉 Snížený zájem: {vt_label}")
                else:
                    st.info(f"📊 Objem normální: {vt_label}")

            # --- Interactive Candlestick Chart (v10.0) ---
            st.markdown("---")
            st.markdown("### 🕯️ Interaktivní svíčkový graf")
            st.caption("Kompletní OHLCV chart s indikátory – Volume, RSI, MACD, Bollinger Bands, MA50/MA200 a DCF fair value linie.")

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                _cc_vol = st.checkbox("Volume", value=True, key="cc_vol")
            with cc2:
                _cc_rsi = st.checkbox("RSI", value=True, key="cc_rsi")
            with cc3:
                _cc_macd = st.checkbox("MACD", value=True, key="cc_macd")
            with cc4:
                _cc_bb = st.checkbox("Bollinger Bands", value=True, key="cc_bb")

            try:
                fig_candle = spp_charts.build_candlestick_chart(
                    price_history=price_history_1y,
                    ticker=ticker,
                    show_ma50=True,
                    show_ma200=True,
                    show_bollinger=_cc_bb,
                    show_volume=_cc_vol,
                    show_rsi=_cc_rsi,
                    show_macd=_cc_macd,
                    fair_value_dcf=locals().get("fair_value_dcf"),
                    height=700,
                )
                st.plotly_chart(fig_candle, use_container_width=True)
            except Exception as e:
                st.warning(f"Svíčkový graf nelze vykreslit: {e}")

    # ------------------------------------------------------------------------
    # TAB 8: Memo & Watchlist  (formerly 7)
    # ------------------------------------------------------------------------
    with tabs[7]:
        st.markdown('<div class="section-header">📝 Investment Memo & Watchlist</div>', unsafe_allow_html=True)
        
        # Load existing
        memos = get_memos()
        watch = get_watchlist()
        
        memo = memos.get("memos", {}).get(ticker, {})
        wl = watch.get("items", {}).get(ticker, {})
        
        # Auto-generate snippets
        auto_thesis = (
            f"{company} ({ticker}) - Investment Thesis\n\n"
            f"• Sektor: {sector}\n"
            f"• Cena: {fmt_money(current_price)} | Verdikt: {verdict}\n"
            f"• DCF Fair Value: {fmt_money(fair_value_dcf)} (MOS: {fmt_pct(mos_dcf)})\n"
            f"• Scorecard: {scorecard:.0f}/100\n"
            f"• Insider Signal: {insider_signal.get('label', '—')} ({float(insider_signal.get('signal', 0)):.0f}/100)"
        )
        
        # Memo form
        st.markdown("### 📄 Investment Memo")
        
        thesis = st.text_area(
            "Investiční teze",
            value=memo.get("thesis") or auto_thesis,
            height=120
        )
        
        drivers = st.text_area(
            "Klíčové faktory úspěchu",
            value=memo.get("drivers") or "- Růst tržeb\n- Zlepšení marží\n- Inovace",
            height=100
        )
        
        risks = st.text_area(
            "Rizika",
            value=memo.get("risks") or "- Konkurence\n- Regulace\n- Makro",
            height=100
        )
        
        catalysts = st.text_area(
            "Katalyzátory",
            value=memo.get("catalysts") or "",
            height=80
        )
        
        buy_conditions = st.text_area(
            "Buy podmínky",
            value=memo.get("buy_conditions") or f"- Entry < {fmt_money(fair_value_dcf * 0.95) if fair_value_dcf else '—'}",
            height=80
        )
        
        notes = st.text_area(
            "Poznámky",
            value=memo.get("notes") or "",
            height=80
        )
        
        # Save/Export buttons
        memo_col1, memo_col2 = st.columns(2)
        
        with memo_col1:
            if st.button("💾 Uložit Memo", use_container_width=True):
                memos.setdefault("memos", {})[ticker] = {
                    "thesis": thesis,
                    "drivers": drivers,
                    "risks": risks,
                    "catalysts": catalysts,
                    "buy_conditions": buy_conditions,
                    "notes": notes,
                    "updated_at": dt.datetime.now().isoformat(),
                }
                set_memos(memos)
                st.success("✅ Memo uloženo!")
        
        with memo_col2:
            if _HAS_PDF and st.button("📄 Export PDF", use_container_width=True):
                summary = {
                    "Price": fmt_money(current_price),
                    "DCF Fair": fmt_money(fair_value_dcf),
                    "Score": f"{scorecard:.0f}/100",
                    "Verdict": verdict
                }
                pdf_bytes = export_memo_pdf(ticker, company, {
                    "thesis": thesis,
                    "drivers": drivers,
                    "risks": risks,
                    "catalysts": catalysts,
                    "buy_conditions": buy_conditions,
                    "notes": notes
                }, summary)
                
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Stáhnout PDF",
                        data=pdf_bytes,
                        file_name=f"memo_{ticker}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        # Watchlist
        st.markdown("---")
        st.markdown("### ⭐ Watchlist")
        
        target_buy = st.number_input(
            "Cílová nákupní cena",
            value=float(wl.get("target_buy", 0.0)) if wl else 0.0,
            step=1.0
        )
        
        wl_col1, wl_col2 = st.columns(2)
        
        with wl_col1:
            if st.button("⭐ Přidat/Aktualizovat", use_container_width=True):
                # Vytvoř snapshot aktuálních dat pro historické sledování
                _snapshot = {
                    "date": dt.datetime.now().strftime("%Y-%m-%d"),
                    "price": current_price,
                    "scorecard": round(scorecard, 1),
                    "dcf_fair": fair_value_dcf,
                    "mos": mos_dcf,
                    "verdict": verdict,
                }
                _existing = watch.get("items", {}).get(ticker, {})
                _snapshots = _existing.get("snapshots", [])
                # Přidat snapshot jen pokud ještě dnes nebyl přidán
                _today = dt.datetime.now().strftime("%Y-%m-%d")
                if not _snapshots or _snapshots[-1].get("date") != _today:
                    _snapshots.append(_snapshot)
                # Uchovat max 52 snapshotů (1 rok týdně)
                _snapshots = _snapshots[-52:]
                watch.setdefault("items", {})[ticker] = {
                    "target_buy": target_buy,
                    "added_at": _existing.get("added_at") or dt.datetime.now().isoformat(),
                    "updated_at": dt.datetime.now().isoformat(),
                    "snapshots": _snapshots,
                }
                set_watchlist(watch)
                st.success("✅ Watchlist aktualizován + snapshot uložen!")
        
        with wl_col2:
            if st.button("🗑️ Odebrat", use_container_width=True):
                if ticker in watch.get("items", {}):
                    watch["items"].pop(ticker, None)
                    set_watchlist(watch)
                    st.success("✅ Odebráno!")
        
        # Show watchlist
        st.markdown("#### 📋 Moje Watchlist")
        items = watch.get("items", {})
        
        if items:
            rows = []
            for tkr, item in items.items():
                inf = fetch_ticker_info(tkr)
                # OPRAVA: správně čteme currentPrice, ne marketCap
                price_now = safe_float(inf.get("currentPrice") or inf.get("regularMarketPrice"))
                tgt = safe_float(item.get("target_buy"))  # OPRAVA: čteme target_buy, ne marketCap
                
                if price_now is not None and tgt is not None and tgt > 0:
                    diff_pct = (price_now / tgt - 1) * 100
                    if price_now <= tgt:
                        status = "🟢 BUY!"
                    elif diff_pct < 5:
                        status = f"🟡 Blízko ({diff_pct:+.1f}%)"
                    else:
                        status = f"⏳ Wait ({diff_pct:+.1f}%)"
                else:
                    status = "⏳ Wait"
                    diff_pct = None

                rows.append({
                    "Ticker": tkr,
                    "Aktuální cena": fmt_money(price_now),
                    "Cílová cena": fmt_money(tgt),
                    "Status": status,
                    "Aktualizováno": item.get("updated_at", "")[:10]
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # === SNAPSHOT HISTORY (nové v9.0) ===
            current_item = items.get(ticker, {})
            snapshots = current_item.get("snapshots", [])
            if len(snapshots) >= 2:
                st.markdown("---")
                st.markdown(f"#### 📈 Historie skóre a ceny – {ticker}")
                snap_df = pd.DataFrame(snapshots)
                snap_df["date"] = pd.to_datetime(snap_df["date"])
                snap_df = snap_df.sort_values("date")

                import plotly.graph_objects as go
                fig_snap = go.Figure()
                if "scorecard" in snap_df.columns:
                    fig_snap.add_trace(go.Scatter(
                        x=snap_df["date"], y=snap_df["scorecard"],
                        name="Scorecard", line=dict(color="#00ff88", width=2),
                        yaxis="y1"
                    ))
                if "price" in snap_df.columns and snap_df["price"].notna().any():
                    fig_snap.add_trace(go.Scatter(
                        x=snap_df["date"], y=snap_df["price"],
                        name="Cena ($)", line=dict(color="#4fc3f7", width=2, dash="dash"),
                        yaxis="y2"
                    ))
                fig_snap.update_layout(
                    height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    yaxis=dict(title="Scorecard (0–100)", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis2=dict(title="Cena ($)", overlaying="y", side="right", gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(orientation="h"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)")
                )
                st.plotly_chart(fig_snap, use_container_width=True)
                st.caption(f"💡 Graf zobrazuje {len(snapshots)} uložených snapshotů. Snapshot se ukládá vždy při kliknutí na '⭐ Přidat/Aktualizovat'.")
        else:
            st.info("Watchlist je prázdný")
    

    # ------------------------------------------------------------------------
    # TAB 9: Social & Guru  (formerly 8)
    # ------------------------------------------------------------------------
    with tabs[8]:
        st.markdown('<div class="section-header">🐦 Social & Guru</div>', unsafe_allow_html=True)

        # ── Automated News Feed (v10.0) ──────────────────────────────────
        st.markdown("### 📰 Automatické zprávy")
        news_articles = spp_news.fetch_stock_news(ticker, info.get("longName", ticker) if info else ticker)
        if news_articles:
            for art in news_articles[:8]:
                src = art.get("source", "")
                title = art.get("title", "")
                url = art.get("url", "#")
                pub = art.get("published", "")[:10] if art.get("published") else ""
                st.markdown(
                    f"- [{title}]({url}) — *{src}* {pub}"
                )
            # AI Sentiment Summary
            if st.button("🤖 AI Sentiment Summary", key="btn_news_sentiment"):
                with st.spinner("Analyzuji sentiment zpráv..."):
                    sentiment_result = spp_news.analyze_news_sentiment_ai(
                        news_articles[:6],
                        gemini_api_key=GEMINI_API_KEY
                    )
                if sentiment_result:
                    sc_col1, sc_col2 = st.columns(2)
                    with sc_col1:
                        st.metric("Sentiment", sentiment_result.get("sentiment", "N/A"))
                        st.metric("Skóre", sentiment_result.get("score", "N/A"))
                    with sc_col2:
                        st.markdown(f"**Souhrn:** {sentiment_result.get('summary', '—')}")
                        themes = sentiment_result.get("themes", [])
                        if themes:
                            st.markdown(f"**Témata:** {', '.join(themes)}")
                        risks = sentiment_result.get("risks", [])
                        if risks:
                            st.markdown(f"**Rizika:** {', '.join(risks)}")
                else:
                    st.info("Sentiment analysis vyžaduje GEMINI_API_KEY.")
        else:
            st.info("💡 Pro automatické zprávy nastav NEWSAPI_KEY nebo POLYGON_API_KEY.")

        st.markdown("---")
        st.markdown("### 🐦 Social Media & Guru Tracking")
        options = []
        option_map = {}
        for cat, people in GURUS.items():
            for name, handle in people.items():
                label = f"{cat} | {name}"
                options.append(label)
                option_map[label] = (cat, name, handle)

        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown("### 👤 Výběr Guru")
            sel = st.selectbox(
                "Vyber guru účet",
                options=options,
                index=0 if options else None,
                key="guru_selectbox"
            )
            cat, name, handle = option_map.get(sel, ("", "", ""))
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Kategorie</div>'
                f'<div class="metric-value" style="font-size:1.1rem;">{cat or "—"}</div>'
                f'<div class="metric-delta" style="opacity:0.8;">@{handle}</div></div>',
                unsafe_allow_html=True
            )
            st.caption("Tip: Text tweetu pro AI analýzu vlož ručně níže (bez Twitter API).")

        with right:
            st.markdown(f"### 🐦 Timeline: {name or '—'}")
            guru_handle = handle
            st.markdown("### 📡 Přímý přenos")
            st.warning("⚠️ X (Twitter) blokuje náhledy v cizích aplikacích. Použij přímý odkaz níže.")
            st.markdown(f"""
            <div style="
                padding: 20px; 
                border-radius: 12px; 
                border: 1px solid rgba(255,255,255,0.1); 
                background: linear-gradient(135deg, rgba(29,161,242,0.1) 0%, rgba(0,0,0,0) 100%);
                text-align: center;
            ">
                <div style="font-size: 50px; margin-bottom: 10px;">🐦</div>
                <h3>@{guru_handle}</h3>
                <p>Klikni pro zobrazení nejnovějších analýz a komentářů přímo na X.</p>
                <a href="https://twitter.com/{guru_handle}" target="_blank" style="text-decoration: none;">
                    <button style="background-color: #1DA1F2; color: white; border: none; padding: 10px 20px; border-radius: 20px; font-weight: bold; cursor: pointer;">
                        Otevřít profil @{guru_handle} ↗
                    </button>
                </a>
                <br><br>
                <div style="text-align: left; font-size: 0.8em; opacity: 0.7;">
                    <strong>Tip:</strong> Otevři profil, najdi zajímavý tweet, zkopíruj text a vlož ho vlevo do AI analýzy.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(f"#### 🔎 Hledat **${ticker}** na X")
            st.markdown(f"""
                <a href="https://twitter.com/search?q=%24{ticker}&src=typed_query&f=top" target="_blank">
                    <button style="background: transparent; border: 1px solid #1DA1F2; color: #1DA1F2; padding: 5px 15px; border-radius: 15px; cursor: pointer;">
                        Nejlepší tweety o ${ticker} ↗
                    </button>
                </a>
            """, unsafe_allow_html=True)

            social_text = st.text_area(
                "Vlož text tweetu nebo komentáře k analýze",
                height=140,
                key="social_text_area"
            )

            analyze_col1, analyze_col2 = st.columns([1, 3])
            with analyze_col1:
                do_analyze = st.button("Analyzovat Sentiment", use_container_width=True, key="btn_analyze_social")
            with analyze_col2:
                st.caption("Použije Gemini (pokud je nastaven GEMINI_API_KEY).")

            if do_analyze:
                if not social_text.strip():
                    st.warning("Vlož prosím text tweetu/komentáře k analýze.")
                else:
                    with st.spinner("Analyzuji…"):
                        result = analyze_social_text_with_gemini(social_text)

                    st.markdown(
                        '<div class="metric-card"><div class="metric-label">Výstup AI</div></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(result)


    # ------------------------------------------------------------------------
    # TAB 10: Screener
    # ------------------------------------------------------------------------
    with tabs[9]:
        st.markdown('<div class="section-header">🔍 Mini Stock Screener</div>', unsafe_allow_html=True)
        st.caption("Prohledej 50 nejpopulárnějších US akcií s vlastními filtry.")

        with st.expander("⚙️ Filtry", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                scr_max_pe = st.number_input("Max P/E", min_value=0.0, value=0.0, step=1.0, key="scr_pe")
                scr_min_roe = st.number_input("Min ROE %", min_value=0.0, value=0.0, step=1.0, key="scr_roe")
            with fc2:
                scr_max_pb = st.number_input("Max P/B", min_value=0.0, value=0.0, step=0.5, key="scr_pb")
                scr_min_div = st.number_input("Min Div Yield %", min_value=0.0, value=0.0, step=0.5, key="scr_div")
            with fc3:
                scr_min_upside = st.number_input("Min Upside %", min_value=0.0, value=0.0, step=5.0, key="scr_upside")
                scr_sector = st.text_input("Sektor (volitelné)", key="scr_sector")

        if st.button("🚀 Spustit Screener", use_container_width=True, type="primary", key="btn_screener"):
            with st.spinner("Načítám data pro ~50 akcií (cca 30–60s)..."):
                scr_df = spp_screener.run_screener(
                    max_pe=scr_max_pe if scr_max_pe > 0 else None,
                    max_pb=scr_max_pb if scr_max_pb > 0 else None,
                    min_div_yield=scr_min_div if scr_min_div > 0 else None,
                    min_roe=scr_min_roe if scr_min_roe > 0 else None,
                    min_upside=scr_min_upside if scr_min_upside > 0 else None,
                    sector_filter=scr_sector if scr_sector.strip() else None,
                )
                st.session_state["screener_result"] = scr_df

        if "screener_result" in st.session_state:
            scr_df = st.session_state["screener_result"]
            if scr_df is not None and not scr_df.empty:
                st.success(f"✅ Nalezeno {len(scr_df)} akcií odpovídajících filtrům")
                st.dataframe(scr_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Žádné akcie neodpovídají zadaným filtrům.")

    # ------------------------------------------------------------------------
    # TAB 11: Portfolio Tracker
    # ------------------------------------------------------------------------
    with tabs[10]:
        st.markdown('<div class="section-header">💼 Portfolio Tracker</div>', unsafe_allow_html=True)

        # Add holding form
        with st.expander("➕ Přidat pozici", expanded=False):
            pf_cols = st.columns(4)
            with pf_cols[0]:
                pf_ticker = st.text_input("Ticker", key="pf_ticker").upper()
            with pf_cols[1]:
                pf_shares = st.number_input("Počet akcií", min_value=0.0, step=1.0, key="pf_shares")
            with pf_cols[2]:
                pf_buy_price = st.number_input("Nákupní cena ($)", min_value=0.0, step=1.0, key="pf_buy_price")
            with pf_cols[3]:
                pf_buy_date = st.date_input("Datum nákupu", key="pf_buy_date")

            if st.button("Uložit do portfolia", key="btn_pf_add", type="primary"):
                if pf_ticker and pf_shares > 0 and pf_buy_price > 0:
                    ok = spp_screener.add_portfolio_holding(
                        ticker=pf_ticker, shares=pf_shares,
                        buy_price=pf_buy_price, buy_date=str(pf_buy_date)
                    )
                    if ok:
                        st.success(f"✅ {pf_ticker} přidán do portfolia!")
                    else:
                        st.error("❌ Supabase není dostupný. Nastav SUPABASE_URL a SUPABASE_KEY.")
                else:
                    st.warning("Vyplň ticker, počet akcií a cenu.")

        # Display portfolio
        if st.button("🔄 Načíst portfolio", key="btn_pf_load", use_container_width=True):
            with st.spinner("Načítám portfolio a aktuální ceny..."):
                pf_summary = spp_screener.get_portfolio_summary()
                st.session_state["pf_summary"] = pf_summary

        if "pf_summary" in st.session_state:
            pf = st.session_state["pf_summary"]
            holdings = pf.get("holdings", [])

            if holdings:
                # Top metrics
                pm1, pm2, pm3 = st.columns(3)
                with pm1:
                    st.metric("Celková hodnota", f"${pf['total_value']:,.2f}")
                with pm2:
                    pnl_delta = f"{pf['total_pnl']:+,.2f}$"
                    st.metric("P&L", pnl_delta, delta=f"{pf['total_pnl_pct']:+.1f}%")
                with pm3:
                    st.metric("Náklad", f"${pf['total_cost']:,.2f}")

                # Holdings table
                import pandas as _pd
                df_h = _pd.DataFrame(holdings)
                display_cols = ["ticker", "shares", "buy_price", "current_price",
                                "cost", "value", "pnl", "pnl_pct", "sector"]
                existing_cols = [c for c in display_cols if c in df_h.columns]
                st.dataframe(df_h[existing_cols], use_container_width=True, hide_index=True)

                # Sector breakdown pie chart
                sectors = pf.get("sector_breakdown", {})
                if sectors:
                    import plotly.graph_objects as go
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(sectors.keys()),
                        values=list(sectors.values()),
                        hole=0.4,
                    )])
                    fig_pie.update_layout(
                        title="Sektorová diverzifikace",
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Zatím nemáš žádné pozice v portfoliu. Přidej je výše ☝️")
                st.caption("Portfolio vyžaduje Supabase tabulku 'portfolio' (user_id TEXT, ticker TEXT, shares FLOAT, buy_price FLOAT, buy_date DATE).")

    # Footer
    st.markdown("---")
    st.caption(f"📊 Data: Yahoo Finance | {APP_NAME} v10.0 | Toto není investiční doporučení")


def display_welcome_screen():
    """Display premium welcome screen when no ticker is selected."""
    # Hero section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Stock Picker Pro</div>
        <div class="hero-subtitle">
            Kvantitativní analýza akcií nové generace.<br>
            DCF, Monte Carlo, AI Analyst, technické indikátory a mnohem víc – vše na jednom místě.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown('<div class="section-header">🆕 Klíčové funkce v10.0</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    features = [
        (fc1, "🤖", "AI Analyst", "Gemini-powered hloubkové reporty s makro kontextem a news sentimentem"),
        (fc2, "📊", "DCF & Monte Carlo", "Pokročilé oceňovací modely s 1 000 simulacemi"),
        (fc3, "🔍", "Stock Screener", "Prohledej 50+ akcií s filtry P/E, ROE, dividenda a víc"),
        (fc4, "📰", "Live Macro & News", "FRED makro data, NewsAPI feed s AI sentiment analýzou"),
    ]
    for col, icon, title, desc in features:
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    fc5, fc6, fc7, fc8 = st.columns(4)
    features2 = [
        (fc5, "💼", "Portfolio Tracker", "Spravuj pozice s P&L, sektorová diverzifikace"),
        (fc6, "🕵️", "Insider Signals", "Multi-source insider data s cluster detekcí"),
        (fc7, "📈", "Technická analýza", "RSI, MACD, Bollinger Bands, MA50/200 interaktivně"),
        (fc8, "☁️", "Cloud Sync", "Supabase watchlist, memos, snapshoty a AI cache"),
    ]
    for col, icon, title, desc in features2:
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Quick start guide
    st.markdown("")
    st.markdown('<div class="section-header">🚀 Jak začít</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px;">
        <div style="flex: 1; min-width: 200px;">
            <div style="font-size: 2rem; margin-bottom: 6px;">①</div>
            <div style="font-weight: 700; margin-bottom: 4px;">Zadej ticker</div>
            <div style="color: #8b92a5; font-size: 0.85rem;">Napiš symbol do levého panelu (AAPL, NVDA, BTC-USD...)</div>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <div style="font-size: 2rem; margin-bottom: 6px;">②</div>
            <div style="font-weight: 700; margin-bottom: 4px;">Analyzuj</div>
            <div style="color: #8b92a5; font-size: 0.85rem;">Klikni "Analyzovat" nebo stiskni Enter</div>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <div style="font-size: 2rem; margin-bottom: 6px;">③</div>
            <div style="font-weight: 700; margin-bottom: 4px;">Prozkoumej</div>
            <div style="color: #8b92a5; font-size: 0.85rem;">11 tabů: Overview, DCF, AI, Macro, Screener a víc</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Popular tickers
    st.markdown('<div class="section-header">💡 Populární tickery</div>', unsafe_allow_html=True)
    cols = st.columns(8)
    samples = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BTC-USD"]
    for i, ticker in enumerate(samples):
        with cols[i]:
            if st.button(ticker, use_container_width=True, key=f"sample_{ticker}"):
                st.session_state["last_ticker"] = ticker
                st.rerun()

    # AI tip
    st.markdown("")
    st.markdown("""
    <div class="info-box">
        💡 <strong>Pro AI analýzu</strong> nastav <code>GEMINI_API_KEY</code> v <code>.streamlit/secrets.toml</code> a získej hloubkové AI reporty!
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
