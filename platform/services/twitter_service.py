"""
Twitter/X Service for Social Sentiment Analysis
Fetches recent tweets via X API v2 for market sentiment
"""

import os
import time
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)


class TwitterService:
    """Service for fetching tweets about trading symbols via X API v2"""

    API_BASE = "https://api.x.com/2"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.twitter_config = config.get('twitter', {})
        self.bearer_token = os.environ.get('X_BEARER_TOKEN', '')
        self.enabled = self.twitter_config.get('enabled', True) and bool(self.bearer_token)

        # Simple TTL cache: {symbol: (timestamp, results)}
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 300  # 5 minutes

        if self.enabled:
            logger.info("Twitter/X service initialized")
        else:
            if not self.bearer_token:
                logger.warning("Twitter/X service disabled: X_BEARER_TOKEN not set")
            else:
                logger.info("Twitter/X service disabled via config")

    def is_available(self) -> bool:
        return self.enabled

    async def search_tweets(
        self,
        symbol: str,
        limit: int = 20,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Search recent tweets about a symbol.

        Returns news items in the same format as NewsService for seamless merging.
        """
        if not self.enabled:
            return []

        clean_symbol = symbol.replace('/USD', '').replace('/', '').upper()
        lang = self.twitter_config.get('lang', 'en')
        query = f"(${clean_symbol} OR #{clean_symbol}) -is:retweet lang:{lang}"

        return await self._search_cached(query, limit, hours_back, symbols=[clean_symbol])

    async def search_by_query(
        self,
        query: str,
        limit: int = 20,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Search tweets with an arbitrary query string.

        Supports broad topic searches like "US tech stocks market" or "$QQQ OR $SPY".
        """
        if not self.enabled:
            return []

        lang = self.twitter_config.get('lang', 'en')
        full_query = f"({query}) -is:retweet lang:{lang}"
        return await self._search_cached(full_query, limit, hours_back)

    async def _search_cached(
        self,
        query: str,
        limit: int,
        hours_back: int,
        symbols: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cache wrapper around _fetch_tweets_raw."""
        cache_key = f"{query}:{limit}:{hours_back}"
        if cache_key in self._cache:
            cached_time, cached_results = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_results

        try:
            results = await self._fetch_tweets_raw(query, limit, hours_back, symbols)
            self._cache[cache_key] = (time.time(), results)
            return results
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []

    async def _fetch_tweets_raw(
        self,
        query: str,
        limit: int,
        hours_back: int,
        symbols: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Call X API v2 /tweets/search/recent with any query."""
        min_likes = self.twitter_config.get('min_likes', 5)
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        params = {
            'query': query,
            'max_results': max(10, min(limit, 100)),
            'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'tweet.fields': 'created_at,public_metrics,author_id,text',
            'user.fields': 'username,name',
            'expansions': 'author_id',
            'sort_order': self.twitter_config.get('sort_order', 'recency'),
        }

        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
        }

        logger.info(f"Fetching tweets: query=\"{query[:80]}\" limit={limit} hours_back={hours_back}")

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self.API_BASE}/tweets/search/recent",
                params=params,
                headers=headers,
            )

            if resp.status_code == 429:
                logger.warning("Twitter API rate limited (429). Returning empty results.")
                return []

            if resp.status_code != 200:
                logger.error(f"Twitter API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()

        tweets = data.get('data', [])
        if not tweets:
            logger.info(f"No tweets found for query")
            return []

        # Build author lookup from includes
        authors = {}
        for user in data.get('includes', {}).get('users', []):
            authors[user['id']] = user.get('username', '')

        # Filter by min_likes and convert to news item format
        results = []
        for tweet in tweets:
            metrics = tweet.get('public_metrics', {})
            if metrics.get('like_count', 0) < min_likes:
                continue

            author_id = tweet.get('author_id', '')
            username = authors.get(author_id, '')
            tweet_id = tweet['id']
            text = tweet.get('text', '')

            results.append({
                'id': f"x_{tweet_id}",
                'headline': text[:120] + ('...' if len(text) > 120 else ''),
                'summary': text,
                'source': 'X.com',
                'author': f"@{username}" if username else 'Unknown',
                'created_at': tweet.get('created_at', ''),
                'url': f"https://x.com/{username}/status/{tweet_id}" if username else '',
                'symbols': symbols or [],
                'images': [],
                'likes': metrics.get('like_count', 0),
                'retweets': metrics.get('retweet_count', 0),
                'views': metrics.get('impression_count', 0),
            })

        # Sort by engagement score: retweets strongest signal, then likes, then views
        results.sort(key=lambda t: (
            t.get('retweets', 0) * 3 + t.get('likes', 0) + t.get('views', 0) * 0.001
        ), reverse=True)

        logger.info(f"Fetched {len(results)} tweets (from {len(tweets)} total, min_likes={min_likes})")
        return results

    def get_sentiment_keywords(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Simple keyword-based sentiment (same as NewsService)"""
        bullish_keywords = [
            'surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'bullish', 'optimistic',
            'growth', 'profit', 'beat', 'exceed', 'record', 'high', 'upgrade', 'buy',
            'moon', 'calls', 'long', 'breakout', 'green'
        ]
        bearish_keywords = [
            'fall', 'drop', 'decline', 'plunge', 'crash', 'bearish', 'pessimistic',
            'loss', 'miss', 'cut', 'downgrade', 'sell', 'low', 'weak', 'concern', 'fear',
            'puts', 'short', 'dump', 'red', 'breakdown'
        ]

        bullish_count = 0
        bearish_count = 0

        for item in items:
            text = (item.get('headline', '') + ' ' + item.get('summary', '')).lower()
            for kw in bullish_keywords:
                if kw in text:
                    bullish_count += 1
            for kw in bearish_keywords:
                if kw in text:
                    bearish_count += 1

        return {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'net_sentiment': bullish_count - bearish_count,
        }
