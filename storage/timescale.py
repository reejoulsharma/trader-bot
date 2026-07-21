"""
storage/timescale.py — TimescaleDB connection and write operations.

Handles:
  - Creating tables and hypertables on first run (idempotent)
  - Writing ticks, news articles, and fundamentals to the database
  - Querying recent data (used by the signal analysis phase)
"""

import asyncpg
from loguru import logger
from config import settings
from ingestion.models import Tick, NewsArticle, Fundamental, Signal


# ── Schema SQL ────────────────────────────────────────────────────────────────

CREATE_TICKS_TABLE = """
CREATE TABLE IF NOT EXISTS ticks (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    volume      BIGINT,
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    source      TEXT
);
"""

CREATE_TICKS_HYPERTABLE = """
SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);
"""

CREATE_TICKS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON ticks (symbol, time DESC);
"""

CREATE_NEWS_TABLE = """
CREATE TABLE IF NOT EXISTS news_articles (
    id          SERIAL PRIMARY KEY,
    published_at TIMESTAMPTZ NOT NULL,
    title       TEXT NOT NULL,
    description TEXT,
    url         TEXT UNIQUE NOT NULL,
    source_name TEXT,
    symbols     TEXT[]
);
"""

CREATE_FUNDAMENTALS_TABLE = """
CREATE TABLE IF NOT EXISTS fundamentals (
    id          SERIAL PRIMARY KEY,
    fetched_at  TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    pe_ratio    DOUBLE PRECISION,
    eps         DOUBLE PRECISION,
    revenue     DOUBLE PRECISION,
    market_cap  DOUBLE PRECISION,
    week52_high DOUBLE PRECISION,
    week52_low  DOUBLE PRECISION,
    source      TEXT
);
"""

CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    time             TIMESTAMPTZ NOT NULL,
    symbol           TEXT        NOT NULL,
    signal_type      TEXT        NOT NULL,
    value            TEXT        NOT NULL,
    confidence_score DOUBLE PRECISION,
    source           TEXT
);
"""

CREATE_SIGNALS_HYPERTABLE = """
SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);
"""

CREATE_SIGNALS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, time DESC);
"""


# ── Connection ────────────────────────────────────────────────────────────────

async def get_db_pool() -> asyncpg.Pool:
    """Create a connection pool to TimescaleDB."""
    pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=10)
    logger.info("TimescaleDB connection pool created")
    return pool


async def run_migrations(pool: asyncpg.Pool):
    """
    Create all tables and hypertables if they don't exist.
    Safe to run on every startup — all statements are idempotent.
    """
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TICKS_TABLE)
        await conn.execute(CREATE_TICKS_HYPERTABLE)
        await conn.execute(CREATE_TICKS_INDEX)
        await conn.execute(CREATE_NEWS_TABLE)
        await conn.execute(CREATE_FUNDAMENTALS_TABLE)
        await conn.execute(CREATE_SIGNALS_TABLE)
        await conn.execute(CREATE_SIGNALS_HYPERTABLE)
        await conn.execute(CREATE_SIGNALS_INDEX)
    logger.info("Database migrations complete")


# ── Writers ───────────────────────────────────────────────────────────────────

async def insert_tick(pool: asyncpg.Pool, tick: Tick):
    """Insert a single tick into the ticks hypertable."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ticks (time, symbol, price, volume, bid, ask, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            tick.timestamp, tick.symbol, tick.price,
            tick.volume, tick.bid, tick.ask, tick.source,
        )


async def insert_news_article(pool: asyncpg.Pool, article: NewsArticle):
    """Insert a news article. Silently ignores duplicates (ON CONFLICT DO NOTHING)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO news_articles
                (published_at, title, description, url, source_name, symbols)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (url) DO NOTHING
            """,
            article.published_at, article.title, article.description,
            article.url, article.source_name, article.symbols,
        )


async def insert_fundamental(pool: asyncpg.Pool, f: Fundamental):
    """Insert a fundamentals snapshot."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO fundamentals
                (fetched_at, symbol, pe_ratio, eps, revenue,
                 market_cap, week52_high, week52_low, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            f.fetched_at, f.symbol, f.pe_ratio, f.eps, f.revenue,
            f.market_cap, f.fifty_two_week_high, f.fifty_two_week_low, f.source,
        )


async def insert_signal(pool: asyncpg.Pool, signal: Signal):
    """Insert a generated trading signal into the signals hypertable."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO signals (time, symbol, signal_type, value, confidence_score, source)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            signal.timestamp, signal.symbol, signal.signal_type,
            signal.value, signal.confidence_score, signal.source,
        )


# ── Readers (used by Phase 2 — Signal Analysis) ───────────────────────────────

async def get_recent_ticks(
    pool: asyncpg.Pool,
    symbol: str,
    minutes: int = 15,
) -> list[dict]:
    """
    Fetch all ticks for a symbol in the last N minutes.
    Used by the signal analysis phase to compute indicators.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT time, price, volume, bid, ask
            FROM ticks
            WHERE symbol = $1
              AND time >= NOW() - ($2 || ' minutes')::INTERVAL
            ORDER BY time ASC
            """,
            symbol.upper(), str(minutes),
        )
    return [dict(row) for row in rows]


async def get_recent_signals(
    pool: asyncpg.Pool,
    symbol: str,
    minutes: int = 30,
) -> list[dict]:
    """
    Fetch all signals for a symbol in the last N minutes.
    Used by the decision engine to score a symbol's current outlook.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT time, signal_type, value, confidence_score
            FROM signals
            WHERE symbol = $1
              AND time >= NOW() - ($2 || ' minutes')::INTERVAL
            ORDER BY time ASC
            """,
            symbol.upper(), str(minutes),
        )
    return [dict(row) for row in rows]


async def get_signal_forward_prices(pool: asyncpg.Pool, horizon_minutes: int) -> list[dict]:
    """
    For every signal, find the entry price (last tick at/before the signal)
    and the exit price (first tick at/after signal_time + horizon_minutes).

    Used by the signal evaluation script to check whether signals actually
    predict favorable forward price moves, before any decision layer is
    built on top of them.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.symbol,
                s.signal_type,
                s.value,
                s.confidence_score,
                entry.price AS entry_price,
                exitp.price AS exit_price
            FROM signals s
            JOIN LATERAL (
                SELECT price FROM ticks t
                WHERE t.symbol = s.symbol AND t.time <= s.time
                ORDER BY t.time DESC
                LIMIT 1
            ) entry ON true
            JOIN LATERAL (
                SELECT price FROM ticks t
                WHERE t.symbol = s.symbol AND t.time >= s.time + ($1 || ' minutes')::INTERVAL
                ORDER BY t.time ASC
                LIMIT 1
            ) exitp ON true
            """,
            str(horizon_minutes),
        )
    return [dict(row) for row in rows]
