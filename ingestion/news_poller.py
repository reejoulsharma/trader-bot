"""
ingestion/news_poller.py — Periodic news ingestion from Google News RSS.

Polls Google News RSS every NEWS_POLL_INTERVAL seconds, deduplicates articles
using Redis, and pushes new articles into the Redis stream.
"""

import asyncio
import hashlib
import httpx
from bs4 import BeautifulSoup
from loguru import logger
import redis.asyncio as aioredis

from config import settings
from ingestion.normaliser import normalise_google_news_rss


class NewsPoller:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.symbols = settings.WATCHLIST
        # Build a query string like "RELIANCE OR TCS OR INFY"
        self.query = " OR ".join(self.symbols)
        self.url = f"https://news.google.com/rss/search?q={self.query}&hl=en-IN&gl=IN&ceid=IN:en"

    def _article_id(self, url: str) -> str:
        """Hash the URL to a short unique ID for deduplication."""
        return hashlib.md5(url.encode()).hexdigest()

    async def _already_seen(self, article_id: str) -> bool:
        """Check Redis if we've already processed this article."""
        return await self.redis.sismember("seen_articles", article_id)

    async def _mark_seen(self, article_id: str):
        """Mark an article as seen. Expires after 24 hours."""
        await self.redis.sadd("seen_articles", article_id)
        await self.redis.expire("seen_articles", 86400)

    async def _fetch_headlines(self) -> list[dict]:
        """Call Google News RSS and return a list of raw article dicts."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")

        articles = []
        for item in items:
            title = item.title.text if item.title else ""
            link = item.link.text if item.link else ""
            description = item.description.text if item.description else ""
            pubDate = item.pubDate.text if item.pubDate else ""

            articles.append({
                "title": title,
                "link": link,
                "description": description,
                "pubDate": pubDate
            })

        return articles

    async def _poll_once(self):
        """Fetch headlines, deduplicate, normalise, and push to Redis."""
        try:
            raw_articles = await self._fetch_headlines()
        except httpx.HTTPError as e:
            logger.warning(f"Google News HTTP error: {e}")
            return
        except Exception as e:
            logger.warning(f"Unexpected error fetching news: {e}")
            return

        new_count = 0
        for raw in raw_articles:
            url = raw.get("link", "")
            if not url:
                continue

            article_id = self._article_id(url)

            if await self._already_seen(article_id):
                continue  # already processed this one

            article = normalise_google_news_rss(raw, self.symbols)
            if article is None:
                continue  # normalisation failed

            # Only care about articles that mention at least one watchlist symbol
            if not article.symbols:
                await self._mark_seen(article_id)
                continue

            payload = article.model_dump_json()
            await self.redis.xadd(
                settings.STREAM_NEWS,
                {"payload": payload},
                maxlen=settings.STREAM_MAXLEN_NEWS,
            )
            await self._mark_seen(article_id)

            logger.info(
                f"News → [{', '.join(article.symbols)}] {article.title[:60]}..."
            )
            new_count += 1

        if new_count:
            logger.info(f"News poll complete — {new_count} new articles ingested")

    async def run(self):
        """
        Run the news poller indefinitely.
        Polls every NEWS_POLL_INTERVAL seconds.
        """
        logger.info(
            f"News poller started | query='{self.query}' "
            f"| interval={settings.NEWS_POLL_INTERVAL}s"
        )
        while True:
            await self._poll_once()
            await asyncio.sleep(settings.NEWS_POLL_INTERVAL)
