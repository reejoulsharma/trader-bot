"""
tests/test_normaliser.py — Unit tests for the normaliser.

Run with:  python -m pytest tests/ -v
"""

from datetime import datetime, timezone

import pytest
from ingestion.normaliser import (
    normalise_angel_tick,
    normalise_google_news_rss,
    normalise_screener_fundamentals,
)


# ── Angel One tick ──────────────────────────────────────────────────────────

def test_normalise_valid_angel_tick():
    raw = {
        "last_traded_price": 250050,  # paisa -> 2500.50 rupees
        "volume_traded_today": 12345,
        "exchange_timestamp": 1705329000000,  # epoch ms
    }
    tick = normalise_angel_tick("reliance", raw)
    assert tick is not None
    assert tick.symbol == "RELIANCE"  # normaliser uppercases via the model
    assert tick.price == pytest.approx(2500.50)
    assert tick.volume == 12345
    assert tick.source == "angel_one"
    assert tick.timestamp.tzinfo is not None  # must be tz-aware


def test_normalise_angel_tick_missing_timestamp_defaults_to_now():
    raw = {"last_traded_price": 100000, "volume_traded_today": 10}
    before = datetime.now(timezone.utc)
    tick = normalise_angel_tick("TCS", raw)
    after = datetime.now(timezone.utc)
    assert tick is not None
    assert before <= tick.timestamp <= after


def test_normalise_angel_tick_zero_price_returns_none():
    raw = {"last_traded_price": 0, "volume_traded_today": 10}
    tick = normalise_angel_tick("INFY", raw)
    assert tick is None


def test_normalise_angel_tick_invalid_price_returns_none():
    raw = {"last_traded_price": "not_a_number", "volume_traded_today": 10}
    tick = normalise_angel_tick("INFY", raw)
    assert tick is None


# ── Google News RSS ──────────────────────────────────────────────────────────

WATCHLIST = ["RELIANCE", "TCS", "INFY"]


def test_normalise_news_article_tags_symbols():
    raw = {
        "title": "RELIANCE hits record high amid refining margins",
        "description": "Reliance Industries shares surged today.",
        "link": "https://example.com/article1",
        "pubDate": "Mon, 15 Jan 2024 10:00:00 GMT",
    }
    article = normalise_google_news_rss(raw, WATCHLIST)
    assert article is not None
    assert "RELIANCE" in article.symbols
    assert article.source_name == "Google News"


def test_normalise_news_article_no_symbol_match():
    raw = {
        "title": "RBI holds interest rates steady",
        "description": "The Reserve Bank voted unanimously to hold rates.",
        "link": "https://example.com/article2",
        "pubDate": "Mon, 15 Jan 2024 10:00:00 GMT",
    }
    article = normalise_google_news_rss(raw, WATCHLIST)
    assert article is not None
    assert article.symbols == []


def test_normalise_news_missing_pubdate_defaults_to_now():
    raw = {
        "title": "TCS wins large deal",
        "description": "TCS signed a multi-year contract.",
        "link": "https://example.com/article3",
    }
    article = normalise_google_news_rss(raw, WATCHLIST)
    assert article is not None
    assert article.published_at.tzinfo is not None


def test_normalise_news_unparseable_pubdate_defaults_to_now():
    raw = {
        "title": "INFY announces buyback",
        "description": "Infosys board approves share buyback.",
        "link": "https://example.com/article4",
        "pubDate": "not a real date",
    }
    article = normalise_google_news_rss(raw, WATCHLIST)
    assert article is not None
    assert article.published_at.tzinfo is not None


def test_normalise_news_missing_link_defaults_to_empty_url():
    raw = {
        "title": "TCS earnings beat expectations",
        "pubDate": "Mon, 15 Jan 2024 10:00:00 GMT",
        # link is missing
    }
    article = normalise_google_news_rss(raw, WATCHLIST)
    assert article is not None
    assert article.url == ""


# ── Screener.in fundamentals ─────────────────────────────────────────────────

def test_normalise_screener_fundamentals():
    raw = {
        "Stock P/E": "28.5",
        "Market Cap": "19,28,450",
        "High / Low": "3,024 / 2,220",
    }
    f = normalise_screener_fundamentals("RELIANCE", raw)
    assert f is not None
    assert f.symbol == "RELIANCE"
    assert f.pe_ratio == 28.5
    assert f.market_cap == 1928450
    assert f.fifty_two_week_high == 3024
    assert f.fifty_two_week_low == 2220
    assert f.source == "screener"


def test_normalise_screener_fundamentals_missing_high_low():
    raw = {"Stock P/E": "22.1", "Market Cap": "500000"}
    f = normalise_screener_fundamentals("TCS", raw)
    assert f is not None
    assert f.fifty_two_week_high is None
    assert f.fifty_two_week_low is None


def test_normalise_screener_fundamentals_handles_unparseable_values():
    raw = {"Stock P/E": "None", "Market Cap": "1,000"}
    f = normalise_screener_fundamentals("INFY", raw)
    assert f is not None
    assert f.pe_ratio is None  # gracefully handled
    assert f.market_cap == 1000
