"""
ingestion/price_stream.py — Real-time price stream via Angel One SmartAPI WebSocket.

Connects to Angel One's WebSocket V2 feed, subscribes to LTP for all
symbols in your watchlist, and pushes each event into Redis.

Includes exponential backoff reconnection.
"""

import asyncio
from loguru import logger
import httpx
import pyotp
import redis.asyncio as aioredis
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from config import settings
from ingestion.normaliser import normalise_angel_tick


class PriceStream:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.symbols = settings.WATCHLIST
        self.symbol_to_token = {}
        self.token_to_symbol = {}

    async def _fetch_token_map(self):
        """Fetch the latest Scrip Master from Angel One and map symbols to tokens."""
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        logger.info("Fetching Angel One Scrip Master JSON...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        
        # We only care about NSE Equities (exch_seg = "NSE") for this simple bot.
        for item in data:
            if item.get("exch_seg") == "NSE":
                # E.g. RELIANCE might be RELIANCE-EQ in scrip master
                sym = item.get("symbol").replace("-EQ", "")
                if sym in self.symbols:
                    token = item.get("token")
                    self.symbol_to_token[sym] = token
                    self.token_to_symbol[token] = sym

        logger.info(f"Resolved tokens for watchlist: {self.symbol_to_token}")
        missing = set(self.symbols) - set(self.symbol_to_token.keys())
        if missing:
            logger.warning(f"Could not resolve tokens for: {missing}")

    async def _authenticate(self) -> tuple[str, str]:
        """Login to SmartAPI and return (auth_token, feed_token)."""
        # SmartConnect uses synchronous requests. We run it in executor so we don't block.
        def _do_login():
            obj = SmartConnect(api_key=settings.ANGEL_API_KEY)
            totp = pyotp.TOTP(settings.ANGEL_TOTP_SECRET).now()
            data = obj.generateSession(
                settings.ANGEL_CLIENT_CODE,
                settings.ANGEL_PASSWORD,
                totp
            )
            if not data or not data.get("status"):
                raise ValueError(f"Angel One login failed: {data}")
            
            auth_token = data["data"]["jwtToken"]
            feed_token = obj.getfeedToken()
            return auth_token, feed_token

        return await asyncio.to_thread(_do_login)

    async def _process_message(self, raw_message: dict):
        """Parse a parsed WebSocket dict and push to Redis."""
        token = raw_message.get("token")
        if not token:
            return

        symbol = self.token_to_symbol.get(token)
        if not symbol:
            return

        tick = normalise_angel_tick(symbol, raw_message)
        if tick is None:
            return

        payload = tick.model_dump_json()
        await self.redis.xadd(
            settings.STREAM_TICKS,
            {"payload": payload},
            maxlen=settings.STREAM_MAXLEN_TICKS,
        )
        await self.redis.hset("latest_prices", tick.symbol, str(tick.price))

        logger.debug(f"Tick → {tick.symbol} @ {tick.price}")

    async def _stream_loop(self):
        auth_token, feed_token = await self._authenticate()
        
        # Need to create a bridge between synchronous callback and our async process
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_message(ws, message):
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_open(ws):
            logger.info("Angel One WebSocket connected.")
            # Mode 1 is LTP.
            token_list = [{"exchangeType": 1, "tokens": list(self.symbol_to_token.values())}]
            ws.subscribe("price-stream-id", 1, token_list)
            logger.info(f"Subscribed to {token_list}")

        def on_error(ws, error):
            logger.error(f"Angel One WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"Angel One WebSocket closed: {close_msg}")
            # put a sentinel to exit the queue
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        sws = SmartWebSocketV2(
            auth_token,
            settings.ANGEL_API_KEY,
            settings.ANGEL_CLIENT_CODE,
            feed_token
        )

        sws.on_open = on_open
        sws.on_message = on_message
        sws.on_error = on_error
        sws.on_close = on_close

        # sws.connect() is blocking. Run in thread.
        ws_task = asyncio.to_thread(sws.connect)

        logger.info("Starting WebSocket message processor...")
        try:
            while True:
                msg = await queue.get()
                if msg is None: # Closed
                    break
                # Only process tick dictionaries
                if isinstance(msg, dict):
                    await self._process_message(msg)
        finally:
            # ensure websocket is closed if we are cancelled
            try:
                sws.close_connection()
            except Exception:
                pass
            # wait for task to complete if we cancelled
            try:
                await asyncio.wait_for(ws_task, timeout=5.0)
            except asyncio.TimeoutError:
                pass

    async def run(self):
        """Run the price stream with exponential backoff reconnection."""
        if not settings.ANGEL_API_KEY or not settings.ANGEL_CLIENT_CODE:
             logger.warning("Angel API key or client code missing. Price stream disabled.")
             return

        await self._fetch_token_map()
        if not self.symbol_to_token:
            logger.error("No tokens resolved for watchlist. Exiting price stream.")
            return

        backoff = 1
        while True:
            try:
                logger.info("Connecting to Angel One price stream...")
                await self._stream_loop()
                backoff = 1
            except Exception as e:
                logger.warning(
                    f"Price stream disconnected: {e}. "
                    f"Reconnecting in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, settings.MAX_RECONNECT_DELAY)
