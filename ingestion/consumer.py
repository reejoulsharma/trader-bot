"""
ingestion/consumer.py — Redis stream consumer.

Reads from all Redis streams (ticks, news, fundamentals) and writes
validated records to TimescaleDB. This decouples data ingestion from
storage — producers never wait for a DB write to complete.

Uses Redis XREAD with blocking mode so it sleeps when the stream is
empty instead of spinning in a busy loop.
"""

import asyncio
import json
from loguru import logger
import redis.asyncio as aioredis
import asyncpg

from config import settings
from ingestion.models import Tick, NewsArticle, Fundamental, Signal
from storage.timescale import (
    insert_tick,
    insert_news_article,
    insert_fundamental,
    insert_signal,
)


class StreamConsumer:
    def __init__(self, redis_client: aioredis.Redis, db_pool: asyncpg.Pool):
        self.redis = redis_client
        self.db = db_pool
        # Placeholder keys only — prepare() MUST be awaited before run() (and
        # before any producer task starts) to resolve these to real IDs.
        self.cursors = {
            settings.STREAM_TICKS: "$",
            settings.STREAM_NEWS: "$",
            settings.STREAM_FUNDAMENTALS: "$",
            settings.STREAM_SIGNALS: "$",
        }

    async def prepare(self):
        """
        Resolve each stream's current last ID as our starting cursor.

        The symbolic "$" cursor only means "the stream's last ID" at the
        moment XREAD actually executes inside run()'s loop — and that can
        run arbitrarily late relative to when producer tasks start
        publishing, since asyncio doesn't guarantee scheduling order.
        Anything published in that gap is silently lost forever (confirmed
        live: fundamentals for 2 of 3 watchlist symbols vanished this way
        on startup). Resolving concrete IDs here, and awaiting this before
        any producer task is created, makes that gap impossible: producers
        can't publish anything before this method has already fixed the
        cutoff.
        """
        for stream in self.cursors:
            last = await self.redis.xrevrange(stream, count=1)
            self.cursors[stream] = last[0][0] if last else "0"
        logger.info(f"Stream consumer cursors initialized: {self.cursors}")

    async def _handle_tick(self, payload: str):
        tick = Tick.model_validate_json(payload)
        await insert_tick(self.db, tick)
        logger.debug(f"DB write → tick {tick.symbol} @ {tick.price}")

    async def _handle_news(self, payload: str):
        article = NewsArticle.model_validate_json(payload)
        await insert_news_article(self.db, article)
        logger.debug(f"DB write → news '{article.title[:50]}...'")

    async def _handle_fundamental(self, payload: str):
        fundamental = Fundamental.model_validate_json(payload)
        await insert_fundamental(self.db, fundamental)
        logger.debug(f"DB write → fundamentals {fundamental.symbol}")

    async def _handle_signal(self, payload: str):
        signal = Signal.model_validate_json(payload)
        await insert_signal(self.db, signal)
        logger.debug(f"DB write → signal {signal.symbol} | {signal.signal_type} | {signal.value}")

    # Map each stream name to its handler
    _handlers = {
        settings.STREAM_TICKS: _handle_tick,
        settings.STREAM_NEWS: _handle_news,
        settings.STREAM_FUNDAMENTALS: _handle_fundamental,
        settings.STREAM_SIGNALS: _handle_signal,
    }

    async def _process_messages(self, stream_name: bytes, messages: list):
        """Process a batch of messages from one Redis stream."""
        stream_key = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
        handler = self._handlers.get(stream_key)

        if handler is None:
            logger.warning(f"No handler for stream: {stream_key}")
            return

        for msg_id, fields in messages:
            try:
                payload = fields.get("payload") or fields.get(b"payload", "")
                if isinstance(payload, bytes):
                    payload = payload.decode()
                await handler(self, payload)
                # Update cursor so we don't re-read this message
                self.cursors[stream_key] = msg_id
            except Exception as e:
                logger.error(
                    f"Failed to process message from {stream_key}: {e} "
                    f"| msg_id={msg_id}"
                )
                # Don't crash — log and move on to the next message

    async def run(self):
        """
        Continuously read from all streams and write to the database.
        Uses XREAD with block=500ms so it doesn't spin when streams are empty.

        Requires prepare() to have already been awaited — see its docstring.
        """
        logger.info("Stream consumer started — listening for data...")

        while True:
            try:
                # Read up to 100 messages from all streams in one call
                results = await self.redis.xread(
                    streams=self.cursors,
                    count=100,
                    block=500,  # block for up to 500ms if no messages
                )

                if not results:
                    continue  # no new messages, loop again

                for stream_name, messages in results:
                    await self._process_messages(stream_name, messages)

            except Exception as e:
                logger.error(f"Consumer error: {e}. Retrying in 2s...")
                await asyncio.sleep(2)
