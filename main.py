"""
main.py — Entry point for the trading bot Phase 1 pipeline.

Starts all ingestion components as concurrent async tasks:
  - Price stream (WebSocket, real-time)
  - News poller (REST, every 30s)
  - Fundamentals poller (REST, daily)
  - Stream consumer (Redis → TimescaleDB writer)

Run with:  python main.py
Stop with: Ctrl+C
"""

import asyncio
import sys
from loguru import logger

from config import settings
from storage.redis_client import get_redis
from storage.timescale import get_db_pool, run_migrations
from ingestion.price_stream import PriceStream
from ingestion.news_poller import NewsPoller
from ingestion.fundamentals_poller import FundamentalsPoller
from ingestion.consumer import StreamConsumer
from engine.signal_engine import SignalEngine


def configure_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/tradingbot.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
    )


async def healthcheck(redis, db_pool):
    """Verify all connections are alive before starting producers."""
    logger.info("Running pre-flight checks...")

    # Redis
    await redis.ping()
    logger.info("  ✓ Redis reachable")

    # TimescaleDB
    async with db_pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    logger.info("  ✓ TimescaleDB reachable")

    # Warn if API keys look like placeholders
    if not settings.ANGEL_API_KEY or settings.ANGEL_API_KEY.startswith("your_"):
        logger.warning(
            "  ⚠ ANGEL_API_KEY looks like a placeholder or is missing — "
            "price stream will fail. Update your .env file."
        )
    else:
        logger.info("  ✓ Angel One API key set")


async def main():
    configure_logging()
    logger.info("=" * 50)
    logger.info("  Trading Bot — Phase 1: Data Ingestion")
    logger.info(f"  Watchlist: {settings.WATCHLIST}")
    logger.info("=" * 50)

    # ── Initialise infrastructure ──────────────────────────────────────────────
    redis = await get_redis()
    db_pool = await get_db_pool()

    await healthcheck(redis, db_pool)
    await run_migrations(db_pool)

    # ── Wire up components ─────────────────────────────────────────────────────
    price_stream = PriceStream(redis_client=redis)
    news_poller = NewsPoller(redis_client=redis)
    fundamentals_poller = FundamentalsPoller(redis_client=redis)
    consumer = StreamConsumer(redis_client=redis, db_pool=db_pool)
    signal_engine = SignalEngine(redis_client=redis)

    # ── Launch all tasks concurrently ──────────────────────────────────────────
    tasks = [
        asyncio.create_task(price_stream.run(), name="price_stream"),
        asyncio.create_task(news_poller.run(), name="news_poller"),
        asyncio.create_task(fundamentals_poller.run(), name="fundamentals_poller"),
        asyncio.create_task(consumer.run(), name="consumer"),
        asyncio.create_task(signal_engine.run(), name="signal_engine"),
    ]

    logger.info("All ingestion tasks started. Press Ctrl+C to stop.\n")

    try:
        # Wait for all tasks — if any crash, the exception propagates here
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}")
        raise
    finally:
        # Clean up gracefully
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await redis.aclose()
        await db_pool.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
