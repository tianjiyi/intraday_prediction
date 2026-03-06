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

from services.news_impact_service import NewsImpactService, ScoredNewsItem
from services.sector_trend_service import SectorTrendService

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

        # Scoring services
        self.impact_service = NewsImpactService(config)
        self.sector_trend_service = SectorTrendService(config)

        # Polling intervals (seconds)
        self.alpaca_interval = self.monitor_config.get('alpaca_interval', 300)
        self.twitter_interval = self.monitor_config.get('twitter_interval', 300)
        self.polymarket_interval = self.monitor_config.get('polymarket_interval', 600)

        # Source configs
        self.twitter_config = config.get('twitter', {})
        self.polymarket_config = config.get('polymarket', {})

        # Search config
        self.watchlist = self.monitor_config.get('watchlist', [
            'QQQ', 'SPY', 'AAPL', 'NVDA', 'TSLA', 'MSFT', 'META',
        ])
        self.twitter_queries = self.monitor_config.get('twitter_queries', [
            '$QQQ OR $SPY OR $AAPL OR $NVDA OR $TSLA',
            '$VIX OR $DXY OR $TNX OR "10Y yield"',
            '"breaking" (market OR stocks OR Fed OR CPI)',
        ])

        # Incremental fetch timestamps
        self._twitter_last_fetch: Optional[datetime] = None
        self._alpaca_last_fetch: Optional[datetime] = None

        # Deduplication
        self._seen_ids: Dict[str, float] = {}
        self._seen_ttl = 3600 * 6  # 6 hours

        # In-memory buffer (newest first)
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_max = self.monitor_config.get('buffer_size', 200)

        # Scored buffer and critical queue
        scoring_cfg = config.get('news_scoring', {})
        self._scored_buffer: List[ScoredNewsItem] = []
        self._scored_buffer_max = scoring_cfg.get('scored_buffer_size', 1000)
        self._decay_floor = scoring_cfg.get('decay_floor', 10)

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
            asyncio.create_task(self._decay_loop()),
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

    def _refresh_decay(self):
        """Apply decay to all scored items so tiers are fresh at read time."""
        now = time.time()
        for item in self._scored_buffer:
            self.impact_service.apply_decay(item, now)

    def get_scored_buffer(self, category: Optional[str] = None, limit: int = 50) -> List[ScoredNewsItem]:
        """Return scored items, optionally filtered by category."""
        self._refresh_decay()
        items = self._scored_buffer
        if category and category != 'all':
            items = [i for i in items if i.category == category]
        return items[:limit]

    def get_critical_queue(self, sector: Optional[str] = None, limit: int = 30) -> List[ScoredNewsItem]:
        """Return items with impact_tier == 'critical', optionally filtered by sector."""
        self._refresh_decay()
        items = [i for i in self._scored_buffer if i.impact_tier == 'critical']
        if sector:
            items = [i for i in items if sector in i.sector_tags]
        return items[:limit]

    def get_trending_sectors(
        self, window: Optional[str] = None, limit: int = 8, critical_only: bool = False
    ) -> List:
        """Return ranked sector trends for the given time window."""
        self._refresh_decay()
        return self.sector_trend_service.get_trending(
            self._scored_buffer, window=window, limit=limit, critical_only=critical_only
        )

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

        # Incremental: first call uses 1h window, subsequent calls use interval + overlap
        now = datetime.now(timezone.utc)
        if self._alpaca_last_fetch is None:
            hours_back = 1
        else:
            elapsed = (now - self._alpaca_last_fetch).total_seconds() / 3600
            hours_back = elapsed + 0.1  # 6-second overlap buffer

        all_items = []
        # General market news
        try:
            market_news = await self.news_service.get_market_news(limit=15, hours_back=hours_back)
            all_items.extend(market_news)
        except Exception as e:
            logger.warning(f"Alpaca market news error: {e}")

        # Per-symbol news for watchlist (limit to avoid rate limits)
        for symbol in self.watchlist[:5]:
            try:
                items = await self.news_service.get_news(symbol, limit=5, hours_back=hours_back)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Alpaca news error for {symbol}: {e}")

        if all_items:
            self._alpaca_last_fetch = now
        logger.debug(f"Alpaca fetch: hours_back={hours_back:.2f}, got {len(all_items)} items")
        return all_items

    async def _fetch_twitter_news(self) -> List[Dict]:
        if not self.twitter_service or not self.twitter_service.is_available():
            return []

        # Incremental: first call uses config hours_back, subsequent calls use interval + overlap
        now = datetime.now(timezone.utc)
        if self._twitter_last_fetch is None:
            hours_back = self.twitter_config.get('hours_back', 6)
        else:
            elapsed = (now - self._twitter_last_fetch).total_seconds() / 3600
            hours_back = elapsed + 0.1  # 6-second overlap buffer

        all_items = []
        for query in self.twitter_queries:
            try:
                items = await self.twitter_service.search_by_query(query, limit=10, hours_back=hours_back)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Twitter query error '{query[:40]}': {e}")

        if all_items:
            self._twitter_last_fetch = now
        logger.debug(f"Twitter fetch: hours_back={hours_back:.2f}, got {len(all_items)} items")
        return all_items

    async def _fetch_polymarket_data(self) -> List[Dict]:
        if not self.polymarket_service or not self.polymarket_service.is_available():
            return []

        limit = self.polymarket_config.get('trending_limit', 15)
        try:
            return await self.polymarket_service.get_trending_markets(limit=limit)
        except Exception as e:
            logger.warning(f"Polymarket fetch error: {e}")
            return []

    # ----------------------------------------------------------
    # Processing Pipeline
    # ----------------------------------------------------------

    async def _process_new_items(self, items: List[Dict]):
        """Deduplicate, categorize, score, buffer, and broadcast."""
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

        # --- Impact scoring pipeline (includes cross-source dedupe) ---
        scored_items = self.impact_service.batch_score(new_items, self._scored_buffer)
        if scored_items:
            self._scored_buffer = scored_items + self._scored_buffer
            self._scored_buffer = self._scored_buffer[:self._scored_buffer_max]

        # Filter raw items to only those that survived scoring dedupe
        scored_ids = {s.id for s in scored_items} if scored_items else set()
        deduped_items = [i for i in new_items if i.get('id', '') in scored_ids]

        # Prepend deduped items to legacy buffer (backward compat)
        self._buffer = deduped_items + self._buffer
        self._buffer = self._buffer[:self._buffer_max]
        self._unread_count += len(deduped_items)

        # Build broadcast payload with scored fields where available
        broadcast_items = []
        scored_by_id = {s.id: s for s in scored_items} if scored_items else {}
        for item in deduped_items:
            enriched = dict(item)
            scored = scored_by_id.get(item.get('id', ''))
            if scored:
                enriched['impact_score'] = scored.impact_score
                enriched['impact_tier'] = scored.impact_tier
                enriched['sector_tags'] = scored.sector_tags
                enriched['impact_reasons'] = scored.impact_reasons
                enriched['sentiment_strength'] = scored.sentiment_strength
                enriched['market_breadth'] = scored.market_breadth
                enriched['horizon'] = scored.horizon
                enriched['category'] = scored.category
            broadcast_items.append(enriched)

        # Broadcast via WebSocket
        try:
            await self.broadcast({
                'type': 'news_update',
                'items': broadcast_items,
                'total_count': len(self._buffer),
                'unread_count': self._unread_count,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.warning(f"News broadcast error: {e}")

        logger.info(f"Broadcast {len(deduped_items)} news items "
                     f"({len(scored_items)} scored, {len(new_items) - len(deduped_items)} deduped, "
                     f"buffer: {len(self._scored_buffer)})")

    async def _decay_loop(self):
        """Periodically re-compute decayed scores and prune stale items."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                if not self._scored_buffer:
                    continue
                now = time.time()
                for item in self._scored_buffer:
                    self.impact_service.apply_decay(item, now)
                before = len(self._scored_buffer)
                self._scored_buffer = [
                    i for i in self._scored_buffer if i.decayed_score >= self._decay_floor
                ]
                pruned = before - len(self._scored_buffer)
                if pruned:
                    logger.debug(f"Decay pruned {pruned} items (buffer: {len(self._scored_buffer)})")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decay loop error: {e}")

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------

    @staticmethod
    def _classify_category(item: Dict) -> str:
        text = (item.get('headline', '') + ' ' + item.get('summary', '')).lower()

        if item.get('source') == 'Polymarket':
            return 'market_structure'

        geo_score = sum(1 for kw in GEO_KEYWORDS if kw in text)
        tech_score = sum(1 for kw in TECH_KEYWORDS if kw in text)
        fin_score = sum(1 for kw in FINANCIAL_KEYWORDS if kw in text)

        if geo_score > tech_score and geo_score > fin_score and geo_score > 0:
            return 'geopolitics'
        if fin_score > tech_score and fin_score > 0:
            return 'macro'
        if tech_score > 0:
            return 'company_specific'
        return 'company_specific'

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
