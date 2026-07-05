"""
ingestion/fundamentals_poller.py — Daily fundamental data from Screener.in.

Fetches company overview (P/E, market cap etc.) once per day
for each symbol in your watchlist by scraping Screener.in and pushes it into the Redis stream.
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
from loguru import logger
import redis.asyncio as aioredis

from config import settings
from ingestion.normaliser import normalise_screener_fundamentals


class FundamentalsPoller:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.symbols = settings.WATCHLIST

    async def _fetch_overview(self, symbol: str) -> dict:
        """Scrape company overview from Screener.in."""
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            
            # If consolidated is not available, fallback to standalone
            if response.status_code == 404:
                url = f"https://www.screener.in/company/{symbol}/"
                response = await client.get(url)
            
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            company_ratios = soup.find("div", {"class": "company-ratios"})
            if not company_ratios:
                return {}

            data = {}
            for li in company_ratios.find_all("li"):
                name_span = li.find("span", {"class": "name"})
                value_span = li.find("span", {"class": "number"})
                if name_span and value_span:
                    name = name_span.text.strip()
                    value = value_span.text.strip()
                    data[name] = value
            return data

    async def _fetch_and_push(self, symbol: str):
        """Fetch fundamentals for one symbol and push to Redis."""
        try:
            raw = await self._fetch_overview(symbol)
        except httpx.HTTPError as e:
            logger.warning(f"Screener HTTP error for {symbol}: {e}")
            return
        except Exception as e:
            logger.warning(f"Unexpected error scraping {symbol}: {e}")
            return

        if not raw:
            logger.warning(f"Could not parse fundamentals for {symbol} from Screener.in.")
            return

        fundamental = normalise_screener_fundamentals(symbol, raw)
        if fundamental is None:
            return

        payload = fundamental.model_dump_json()
        await self.redis.xadd(
            settings.STREAM_FUNDAMENTALS,
            {"payload": payload},
            maxlen=settings.STREAM_MAXLEN_FUNDAMENTALS,
        )

        # Cache the latest fundamentals in a Redis hash for instant access
        await self.redis.hset(
            f"fundamentals:{symbol}",
            mapping={
                "pe_ratio": str(fundamental.pe_ratio or ""),
                "eps": str(fundamental.eps or ""),
                "market_cap": str(fundamental.market_cap or ""),
            },
        )

        logger.info(
            f"Fundamentals → {symbol} | P/E={fundamental.pe_ratio} "
            f"MarketCap={fundamental.market_cap}"
        )

    async def _poll_once(self):
        """Fetch fundamentals for all symbols sequentially."""
        logger.info(f"Fetching fundamentals for {self.symbols}")
        for symbol in self.symbols:
            await self._fetch_and_push(symbol)
            # Be polite to screener.in
            await asyncio.sleep(5)

    async def run(self):
        """
        Run the fundamentals poller indefinitely.
        Fetches once at startup, then every 24 hours.
        """
        logger.info("Fundamentals poller started")
        while True:
            await self._poll_once()
            logger.info(
                f"Fundamentals update complete. "
                f"Next update in {settings.FUNDAMENTALS_POLL_INTERVAL // 3600}h"
            )
            await asyncio.sleep(settings.FUNDAMENTALS_POLL_INTERVAL)
