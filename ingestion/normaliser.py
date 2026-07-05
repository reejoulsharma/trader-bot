"""
ingestion/normaliser.py — Raw API payload → canonical model.

Each API returns data in a different shape. This module converts
every raw payload into a Tick, NewsArticle, or Fundamental.

If validation fails, it logs the error and returns None — the
caller decides whether to drop or retry the record.
"""

from datetime import datetime, timezone
import dateutil.parser
from typing import Optional
from loguru import logger
from ingestion.models import Tick, NewsArticle, Fundamental


def normalise_angel_tick(symbol: str, raw: dict) -> Optional[Tick]:
    """
    Converts an Angel One SmartAPI tick event to a Tick.
    Price is divided by 100 as it comes in paisa.
    """
    try:
        price = float(raw.get("last_traded_price", 0)) / 100
        if price <= 0:
            return None

        # Try to parse timestamp from exchange_timestamp if available
        # exchange_timestamp is epoch in milliseconds
        ts = raw.get("exchange_timestamp")
        if ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        return Tick(
            symbol=symbol,
            timestamp=dt,
            price=price,
            volume=int(raw.get("volume_traded_today", 0)),
            source="angel_one",
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to normalise Angel One tick: {e} | raw={raw}")
        return None


def normalise_google_news_rss(raw: dict, watchlist: list[str]) -> Optional[NewsArticle]:
    """
    Converts a parsed Google News RSS item to a NewsArticle.
    """
    try:
        title = raw.get("title", "") or ""
        description = raw.get("description", "") or ""
        combined_text = f"{title} {description}".upper()

        # Tag which symbols from our watchlist appear in the title/description
        matched_symbols = [s for s in watchlist if s in combined_text]

        # Parse the published string
        pub_str = raw.get("pubDate")
        if pub_str:
            try:
                published_at = dateutil.parser.parse(pub_str)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except Exception:
                published_at = datetime.now(timezone.utc)
        else:
            published_at = datetime.now(timezone.utc)

        return NewsArticle(
            title=title,
            description=description,
            url=raw.get("link", ""),
            published_at=published_at,
            source_name="Google News",
            symbols=matched_symbols,
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to normalise Google News article: {e} | raw={raw}")
        return None


def normalise_screener_fundamentals(symbol: str, raw: dict) -> Optional[Fundamental]:
    """
    Converts a scraped dictionary from Screener.in to a Fundamental.
    """
    def safe_float(value: str) -> Optional[float]:
        if not value:
            return None
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    try:
        high_low = str(raw.get("High / Low", ""))
        high = None
        low = None
        if "/" in high_low:
            parts = high_low.split("/")
            high = safe_float(parts[0])
            low = safe_float(parts[1])

        return Fundamental(
            symbol=symbol,
            fetched_at=datetime.now(timezone.utc),
            pe_ratio=safe_float(raw.get("Stock P/E")),
            eps=None, # Screener usually doesn't show raw EPS in the top bar
            revenue=None,
            market_cap=safe_float(raw.get("Market Cap")),
            fifty_two_week_high=high,
            fifty_two_week_low=low,
            source="screener",
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to normalise Screener fundamentals for {symbol}: {e}")
        return None
