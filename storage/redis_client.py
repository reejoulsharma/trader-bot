"""
storage/redis_client.py — Redis connection and helper utilities.

Provides a single shared async Redis client and convenience
functions used across the ingestion pipeline.
"""

import redis.asyncio as aioredis
from loguru import logger
from config import settings


async def get_redis() -> aioredis.Redis:
    """Create and verify a Redis connection."""
    client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    try:
        await client.ping()
        logger.info(f"Redis connected at {settings.REDIS_URL}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise
    return client


async def get_latest_price(redis: aioredis.Redis, symbol: str) -> float | None:
    """
    Get the most recent price for a symbol from Redis.
    Returns None if the symbol hasn't been seen yet.
    Used by the signal analysis phase (Phase 2) for instant lookups.
    """
    value = await redis.hget("latest_prices", symbol.upper())
    return float(value) if value else None


async def get_latest_prices(redis: aioredis.Redis) -> dict[str, float]:
    """Get the latest price for all symbols currently tracked."""
    raw = await redis.hgetall("latest_prices")
    return {symbol: float(price) for symbol, price in raw.items()}


async def get_fundamentals(redis: aioredis.Redis, symbol: str) -> dict:
    """Retrieve cached fundamentals for a symbol."""
    return await redis.hgetall(f"fundamentals:{symbol.upper()}")
