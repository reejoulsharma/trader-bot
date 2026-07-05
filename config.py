"""
config.py — Central configuration loader.
Reads from .env and exposes typed settings to the rest of the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Angel One
    ANGEL_API_KEY: str = os.getenv("ANGEL_API_KEY", "")
    ANGEL_CLIENT_CODE: str = os.getenv("ANGEL_CLIENT_CODE", "")
    ANGEL_PASSWORD: str = os.getenv("ANGEL_PASSWORD", "")
    ANGEL_TOTP_SECRET: str = os.getenv("ANGEL_TOTP_SECRET", "")

    # Infrastructure
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradingbot")

    # Bot settings
    WATCHLIST: list[str] = os.getenv("WATCHLIST", "RELIANCE,TCS,INFY").split(",")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Redis stream names
    STREAM_TICKS: str = "market:ticks"
    STREAM_NEWS: str = "market:news"
    STREAM_FUNDAMENTALS: str = "market:fundamentals"
    STREAM_SIGNALS: str = "market:signals"

    # Max length each Redis stream is trimmed to (approximate), so streams
    # don't grow unbounded in memory. Ticks are highest-frequency so get
    # the largest cap.
    STREAM_MAXLEN_TICKS: int = 50_000
    STREAM_MAXLEN_NEWS: int = 5_000
    STREAM_MAXLEN_FUNDAMENTALS: int = 2_000
    STREAM_MAXLEN_SIGNALS: int = 10_000

    # Polling intervals (seconds)
    NEWS_POLL_INTERVAL: int = 30
    FUNDAMENTALS_POLL_INTERVAL: int = 86400  # once per day
    SIGNAL_ENGINE_INTERVAL: int = 60


    # Reconnection backoff
    MAX_RECONNECT_DELAY: int = 60


settings = Config()
