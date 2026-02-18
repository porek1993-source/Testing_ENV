"""
Stock Picker Pro – News & Sentiment Module
===========================================
Provides automated news feeds from NewsAPI and Polygon.io,
with optional AI-powered sentiment analysis via Gemini.
"""

import os
import datetime as dt
from typing import Any, Dict, List, Optional

import streamlit as st
import requests


# ---------------------------------------------------------------------------
# API Key helpers
# ---------------------------------------------------------------------------

def _get_newsapi_key() -> str:
    try:
        return st.secrets.get("NEWSAPI_KEY", "") or os.environ.get("NEWSAPI_KEY", "")
    except Exception:
        return os.environ.get("NEWSAPI_KEY", "")


def _get_polygon_key() -> str:
    try:
        return st.secrets.get("POLYGON_API_KEY", "") or os.environ.get("POLYGON_API_KEY", "")
    except Exception:
        return os.environ.get("POLYGON_API_KEY", "")


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_newsapi(query: str, page_size: int = 10, language: str = "en") -> List[Dict]:
    """
    Fetch news articles from NewsAPI.org.
    Free tier: 100 requests/day, recent articles only.
    """
    key = _get_newsapi_key()
    if not key:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 20),
            "apiKey": key,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "image_url": article.get("urlToImage", ""),
            })
        return articles
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Polygon.io News
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_polygon(ticker: str, limit: int = 10) -> List[Dict]:
    """
    Fetch news from Polygon.io for a specific ticker.
    Free tier: 5 calls/min.
    """
    key = _get_polygon_key()
    if not key:
        return []

    try:
        url = f"https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": ticker.upper(),
            "limit": min(limit, 20),
            "sort": "published_utc",
            "order": "desc",
            "apiKey": key,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        articles = []
        for article in data.get("results", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("publisher", {}).get("name", ""),
                "url": article.get("article_url", ""),
                "published_at": article.get("published_utc", ""),
                "image_url": article.get("image_url", ""),
                "tickers": article.get("tickers", []),
            })
        return articles
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Polygon.io – Ticker Details (company info fallback)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ticker_details_polygon(ticker: str) -> Optional[Dict]:
    """Fetch detailed company info from Polygon.io as fallback data source."""
    key = _get_polygon_key()
    if not key:
        return None

    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}"
        params = {"apiKey": key}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        result = data.get("results", {})
        if not result:
            return None

        return {
            "name": result.get("name", ""),
            "description": result.get("description", ""),
            "market_cap": result.get("market_cap"),
            "sic_description": result.get("sic_description", ""),
            "homepage_url": result.get("homepage_url", ""),
            "total_employees": result.get("total_employees"),
            "list_date": result.get("list_date", ""),
            "locale": result.get("locale", ""),
            "primary_exchange": result.get("primary_exchange", ""),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Combined News Feed (multi-source)
# ---------------------------------------------------------------------------

def fetch_stock_news(ticker: str, company_name: str = "", max_articles: int = 10) -> List[Dict]:
    """
    Fetch news from multiple sources, merge and deduplicate.
    Priority: Polygon (ticker-specific) > NewsAPI (broader search).
    """
    all_articles = []

    # Source 1: Polygon.io (ticker-specific, more relevant)
    polygon_news = fetch_news_polygon(ticker, limit=max_articles)
    for article in polygon_news:
        article["_source_api"] = "Polygon.io"
        all_articles.append(article)

    # Source 2: NewsAPI (broader search including company name)
    remaining = max(0, max_articles - len(all_articles))
    if remaining > 0:
        query = f'"{ticker}" OR "{company_name}"' if company_name else ticker
        newsapi_news = fetch_news_newsapi(query, page_size=remaining)
        for article in newsapi_news:
            article["_source_api"] = "NewsAPI"
            all_articles.append(article)

    # Simple deduplication by title similarity
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_key = article.get("title", "").lower().strip()[:60]
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)

    return unique_articles[:max_articles]


# ---------------------------------------------------------------------------
# AI Sentiment Analysis (via Gemini)
# ---------------------------------------------------------------------------

def analyze_news_sentiment_ai(
    articles: List[Dict],
    gemini_api_key: str = "",
    model: str = "gemini-2.5-flash-lite",
) -> Optional[Dict]:
    """
    Use Gemini to analyze the overall sentiment of news articles.
    Returns: {sentiment, score, summary, key_themes}
    """
    if not gemini_api_key or not articles:
        return None

    # Prepare text from top 5 articles
    texts = []
    for article in articles[:5]:
        title = article.get("title", "")
        desc = article.get("description", "")
        source = article.get("source", "")
        texts.append(f"[{source}] {title}. {desc}")

    combined = "\n\n".join(texts)

    prompt = f"""Analyzuj sentiment těchto finančních zpráv a odpověz POUZE ve formátu JSON (česky):

ZPRÁVY:
{combined}

VÝSTUP JSON:
{{
  "sentiment": "BULLISH/BEARISH/NEUTRAL",
  "score": <číslo 0-100, kde 100=extrémně bullish>,
  "summary": "Jednovětné shrnutí celkového sentimentu zpráv.",
  "key_themes": ["Téma 1", "Téma 2", "Téma 3"],
  "risk_flags": ["Riziko zmíněné ve zprávách, pokud existuje"]
}}"""

    try:
        import re
        import json

        try:
            from google import genai as genai_new
            client = genai_new.Client(api_key=gemini_api_key)
            resp = client.models.generate_content(model=model, contents=prompt)
            raw_text = getattr(resp, "text", None) or str(resp)
        except Exception:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=gemini_api_key)
            m = genai_legacy.GenerativeModel(model)
            resp = m.generate_content(prompt)
            raw_text = getattr(resp, "text", None) or str(resp)

        # Parse
        cleaned = re.sub(r"```json\n?|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                return json.loads(match.group(0))
            return None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenFIGI Ticker Resolution
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def resolve_ticker_openfigi(ticker: str, exchange: str = "") -> Optional[Dict]:
    """
    Resolve a ticker to its full instrument details via OpenFIGI.
    No API key required for basic usage (rate limited).
    """
    try:
        url = "https://api.openfigi.com/v3/mapping"
        headers = {"Content-Type": "application/json"}
        payload = [{"idType": "TICKER", "idValue": ticker.upper()}]
        if exchange:
            payload[0]["exchCode"] = exchange

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data or not isinstance(data, list) or not data[0].get("data"):
            return None

        result = data[0]["data"][0]
        return {
            "figi": result.get("figi", ""),
            "name": result.get("name", ""),
            "ticker": result.get("ticker", ""),
            "exchange": result.get("exchCode", ""),
            "market_sector": result.get("marketSector", ""),
            "security_type": result.get("securityType", ""),
        }
    except Exception:
        return None
