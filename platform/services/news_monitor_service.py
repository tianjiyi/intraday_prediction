"""
News Monitor Service — Background polling orchestrator.

Polls Alpaca News, Twitter/X, and Polymarket on independent schedules.
Deduplicates, categorizes, and broadcasts new items via WebSocket.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Keyword Lists for Classification
# ============================================================

TECH_KEYWORDS = {
    'tech', 'ai', 'chip', 'semiconductor', 'software', 'cloud', 'saas',
    'apple', 'google', 'nvidia', 'microsoft', 'meta', 'amazon', 'tesla',
    'aapl', 'googl', 'nvda', 'msft', 'amzn', 'tsla', 'qqq', 'nasdaq',
    'openai', 'chatgpt', 'robot', 'autonomous', 'data center',
}

GEO_KEYWORDS = {
    'war', 'conflict', 'sanction', 'military', 'geopolitic', 'troops',
    'iran', 'china', 'russia', 'ukraine', 'taiwan', 'tariff', 'israel',
    'missile', 'nuclear', 'invasion', 'nato', 'attack', 'strike',
    'ceasefire', 'peace', 'tension', 'diplomacy', 'defense',
}

FINANCIAL_KEYWORDS = {
    'fed', 'rate', 'inflation', 'gdp', 'employment', 'jobs', 'fomc',
    'treasury', 'bond', 'yield', 'earnings', 'revenue', 'cpi', 'ppi',
    'bank', 'financial', 'economic', 'recession', 'stimulus', 'debt',
    'interest rate', 'dow', 'spy', 's&p',
}

BULLISH_KEYWORDS = {
    'surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'bullish',
    'optimistic', 'growth', 'profit', 'beat', 'exceed', 'record',
    'high', 'upgrade', 'buy', 'breakout', 'moon',
}

BEARISH_KEYWORDS = {
    'fall', 'drop', 'decline', 'plunge', 'crash', 'bearish',
    'pessimistic', 'loss', 'miss', 'cut', 'downgrade', 'sell',
    'low', 'weak', 'concern', 'fear', 'dump', 'breakdown',
}


class NewsMonitorService:
    """Background service that polls news sources and pushes updates via WebSocket."""

    def __init__(
        self,
        config: Dict[str, Any],
        news_service,
        twitter_service,
        polymarket_service,
        broadcast_callback: Callable[[Dict], Awaitable],
    ):
        self.config = config
        self.monitor_config = config.get('news_monitor', {})
        self.enabled = self.monitor_config.get('enabled', True)

        # Source services
        self.news_service = news_service
        self.twitter_service = twitter_service
        self.polymarket_service = polymarket_service
        self.broadcast = broadcast_callback

        # Polling intervals (seconds)
        self.alpaca_interval = self.monitor_config.get('alpaca_interval', 300)
        self.twitter_interval = self.monitor_config.get('twitter_interval', 300)
        self.polymarket_interval = self.monitor_config.get('polymarket_interval', 600)

        # Search config
        self.watchlist = self.monitor_config.get('watchlist', [
            'QQQ', 'SPY', 'AAPL', 'NVDA', 'TSLA', 'MSFT', 'META',
        ])
        self.twitter_queries = self.monitor_config.get('twitter_queries', [
            '$QQQ OR $SPY OR $AAPL OR $NVDA',
            'US tech stocks market crash rally',
            'geopolitical risk market oil war Iran',
            'Fed rate decision inflation CPI',
        ])

        # Deduplication
        self._seen_ids: Dict[str, float] = {}
        self._seen_ttl = 3600 * 6  # 6 hours

        # In-memory buffer (newest first)
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_max = self.monitor_config.get('buffer_size', 200)

        # Unread tracking
        self._unread_count = 0

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    async def start(self):
        """Start all polling loops as background asyncio tasks."""
        if not self.enabled:
            logger.info("News monitor disabled in config")
            return

        self._running = True
        self._tasks = [
            asyncio.create_task(self._poll_alpaca_loop()),
            asyncio.create_task(self._poll_twitter_loop()),
            asyncio.create_task(self._poll_polymarket_loop()),
            asyncio.create_task(self._cleanup_seen_ids_loop()),
        ]
        logger.info(f"News monitor started with {len(self._tasks)} polling tasks")

    async def stop(self):
        """Cancel all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        logger.info("News monitor stopped")

    # ----------------------------------------------------------
    # Public Accessors
    # ----------------------------------------------------------

    def get_buffer(self, category: Optional[str] = None, limit: int = 50) -> List[Dict]:
        items = self._buffer
        if category and category != 'all':
            items = [i for i in items if i.get('category') == category]
        return items[:limit]

    def get_unread_count(self) -> int:
        return self._unread_count

    def reset_unread_count(self):
        self._unread_count = 0

    # ----------------------------------------------------------
    # Polling Loops
    # ----------------------------------------------------------

    async def _poll_alpaca_loop(self):
        await asyncio.sleep(2)  # Stagger startup
        while self._running:
            try:
                items = await self._fetch_alpaca_news()
                if items:
                    await self._process_new_items(items)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alpaca poll error: {e}")
            await asyncio.sleep(self.alpaca_interval)

    async def _poll_twitter_loop(self):
        await asyncio.sleep(5)  # Stagger startup
        while self._running:
            try:
                items = await self._fetch_twitter_news()
                if items:
                    await self._process_new_items(items)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Twitter poll error: {e}")
            await asyncio.sleep(self.twitter_interval)

    async def _poll_polymarket_loop(self):
        await asyncio.sleep(8)  # Stagger startup
        while self._running:
            try:
                items = await self._fetch_polymarket_data()
                if items:
                    await self._process_new_items(items)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polymarket poll error: {e}")
            await asyncio.sleep(self.polymarket_interval)

    async def _cleanup_seen_ids_loop(self):
        """Periodically purge old entries from the seen_ids set."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # Every 30 min
                now = time.time()
                expired = [k for k, v in self._seen_ids.items() if now - v > self._seen_ttl]
                for k in expired:
                    del self._seen_ids[k]
                if expired:
                    logger.debug(f"Cleaned up {len(expired)} expired seen IDs")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    # ----------------------------------------------------------
    # Fetch Methods
    # ----------------------------------------------------------

    async def _fetch_alpaca_news(self) -> List[Dict]:
        if not self.news_service or not getattr(self.news_service, 'is_available', lambda: False)():
            return []

        all_items = []
        # General market news
        try:
            market_news = await self.news_service.get_market_news(limit=15, hours_back=1)
            all_items.extend(market_news)
        except Exception as e:
            logger.warning(f"Alpaca market news error: {e}")

        # Per-symbol news for watchlist (limit to avoid rate limits)
        for symbol in self.watchlist[:5]:
            try:
                items = await self.news_service.get_news(symbol, limit=5, hours_back=1)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Alpaca news error for {symbol}: {e}")

        return all_items

    async def _fetch_twitter_news(self) -> List[Dict]:
        if not self.twitter_service or not self.twitter_service.is_available():
            return []

        all_items = []
        for query in self.twitter_queries:
            try:
                items = await self.twitter_service.search_by_query(query, limit=10, hours_back=6)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Twitter query error '{query[:40]}': {e}")

        return all_items

    async def _fetch_polymarket_data(self) -> List[Dict]:
        if not self.polymarket_service or not self.polymarket_service.is_available():
            return []

        try:
            return await self.polymarket_service.get_trending_markets(limit=15)
        except Exception as e:
            logger.warning(f"Polymarket fetch error: {e}")
            return []

    # ----------------------------------------------------------
    # Processing Pipeline
    # ----------------------------------------------------------

    async def _process_new_items(self, items: List[Dict]):
        """Deduplicate, categorize, buffer, and broadcast."""
        new_items = []
        for item in items:
            item_id = item.get('id', '')
            if not item_id or item_id in self._seen_ids:
                continue
            self._seen_ids[item_id] = time.time()

            # Categorize if not already set
            if not item.get('category'):
                item['category'] = self._classify_category(item)

            # Sentiment if not set
            if not item.get('sentiment'):
                item['sentiment'] = self._classify_sentiment(item)

            new_items.append(item)

        if not new_items:
            return

        # Prepend to buffer
        self._buffer = new_items + self._buffer
        self._buffer = self._buffer[:self._buffer_max]
        self._unread_count += len(new_items)

        # Broadcast via WebSocket
        try:
            await self.broadcast({
                'type': 'news_update',
                'items': new_items,
                'total_count': len(self._buffer),
                'unread_count': self._unread_count,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.warning(f"News broadcast error: {e}")

        logger.info(f"Broadcast {len(new_items)} new news items (buffer: {len(self._buffer)})")

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------

    @staticmethod
    def _classify_category(item: Dict) -> str:
        text = (item.get('headline', '') + ' ' + item.get('summary', '')).lower()

        if item.get('source') == 'Polymarket':
            return 'prediction_market'

        geo_score = sum(1 for kw in GEO_KEYWORDS if kw in text)
        tech_score = sum(1 for kw in TECH_KEYWORDS if kw in text)
        fin_score = sum(1 for kw in FINANCIAL_KEYWORDS if kw in text)

        if geo_score > tech_score and geo_score > fin_score and geo_score > 0:
            return 'geopolitical'
        if tech_score > fin_score and tech_score > 0:
            return 'tech'
        if fin_score > 0:
            return 'financial'
        return 'tech'  # Default for market news

    @staticmethod
    def _classify_sentiment(item: Dict) -> str:
        text = (item.get('headline', '') + ' ' + item.get('summary', '')).lower()
        bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        if bull > bear:
            return 'bullish'
        if bear > bull:
            return 'bearish'
        return 'neutral'
