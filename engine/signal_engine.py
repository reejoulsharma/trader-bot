import asyncio
import json
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
import redis.asyncio as redis

from config import settings
from ingestion.models import Signal
from analyzers.sentiment import FinBERTSentimentAnalyzer
from analyzers.volume import VolumeAnomalyDetector
from analyzers.pattern import PricePatternAnalyzer

class SignalEngine:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Load analyzers
        logger.info("Initializing Analyzers (FinBERT may take a moment to load...)")
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.volume_analyzer = VolumeAnomalyDetector(window_size=20, spike_threshold=2.0)
        self.pattern_analyzer = PricePatternAnalyzer()

        # Tracks the last article URL we ran sentiment on, per symbol, so we
        # don't re-analyse (and re-signal) the same article every cycle.
        self._last_sentiment_url: dict[str, str] = {}

    async def fetch_recent_ticks(self, symbol: str, count: int = 50) -> pd.DataFrame:
        """Fetch recent ticks from redis stream for a symbol and return as DataFrame."""
        # Using XREVRANGE to get recent messages. In a large production system, 
        # querying TimescaleDB or using symbol-specific Redis streams is preferred.
        messages = await self.redis.xrevrange(settings.STREAM_TICKS, max='+', min='-', count=count * len(settings.WATCHLIST))
        
        data = []
        for msg_id, msg_data in messages:
            tick_json = msg_data.get("payload", "{}")
            if not tick_json:
                continue
            tick = json.loads(tick_json)
            if tick.get("symbol") == symbol:
                data.append({
                    "timestamp": tick.get("timestamp"),
                    "price": tick.get("price"),
                    "volume": tick.get("volume")
                })
        
        # reverse so it's oldest to newest
        data.reverse()
        return pd.DataFrame(data)

    async def fetch_recent_news(self, symbol: str, count: int = 5) -> list:
        """Fetch recent news from redis stream"""
        messages = await self.redis.xrevrange(settings.STREAM_NEWS, max='+', min='-', count=count * 5)
        news_items = []
        for msg_id, msg_data in messages:
            news_json = msg_data.get("payload", "{}")
            if not news_json:
                continue
            article = json.loads(news_json)
            if symbol in article.get("symbols", []):
                news_items.append(article)
        return news_items

    async def process_sentiment(self, symbol: str):
        news = await self.fetch_recent_news(symbol)
        if not news:
            return
            
        latest_news = news[0]  # Just analyze the latest one
        url = latest_news.get("url", "")
        if url and url == self._last_sentiment_url.get(symbol):
            return  # already analysed this article for this symbol
        self._last_sentiment_url[symbol] = url

        text = latest_news.get("title", "") + " " + latest_news.get("description", "")

        # Run inference in a thread to not block the event loop
        result = await asyncio.to_thread(self.sentiment_analyzer.analyze, text)
        
        if result['label'] != 'neutral':
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type="sentiment",
                value=result['label'],
                confidence_score=result['score']
            )
            await self.publish_signal(signal)

    async def process_price_and_volume(self, symbol: str):
        df = await self.fetch_recent_ticks(symbol, count=30)
        if df.empty:
            return
            
        # 1. Volume Anomaly
        is_volume_spike = self.volume_analyzer.analyze(df)
        if is_volume_spike:
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type="volume",
                value="anomaly",
            )
            await self.publish_signal(signal)
            
        # 2. Pattern Analysis
        if self.pattern_analyzer.detect_breakout(df):
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type="pattern",
                value="breakout",
            )
            await self.publish_signal(signal)
            
        if self.pattern_analyzer.detect_support(df):
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type="pattern",
                value="support",
            )
            await self.publish_signal(signal)

    async def publish_signal(self, signal: Signal):
        try:
            await self.redis.xadd(
                settings.STREAM_SIGNALS,
                {"payload": signal.model_dump_json()},
                maxlen=settings.STREAM_MAXLEN_SIGNALS,
            )
            logger.info(f"Signal generated: {signal.symbol} | {signal.signal_type} | {signal.value}")
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")

    async def run(self):
        logger.info(f"SignalEngine started. Interval: {settings.SIGNAL_ENGINE_INTERVAL}s")
        while True:
            try:
                for symbol in settings.WATCHLIST:
                    await self.process_sentiment(symbol)
                    await self.process_price_and_volume(symbol)
                    
            except Exception as e:
                logger.error(f"Error in SignalEngine loop: {e}")
                
            await asyncio.sleep(settings.SIGNAL_ENGINE_INTERVAL)
