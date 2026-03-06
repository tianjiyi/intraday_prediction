"""
Polymarket Service — Fetches trending prediction markets from the Gamma Markets API.

Public read-only API, no authentication required.
Docs: https://docs.polymarket.com/developers/gamma-markets-api/overview
"""

import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger(__name__)


class PolymarketService:
    """Fetches trending prediction markets from Polymarket."""

    API_BASE = "https://gamma-api.polymarket.com"

    # Tags to include (finance/geopolitics/tech relevant)
    RELEVANT_TAG_SLUGS = {
        "politics", "fed", "fed-rates", "global-elections", "world-elections",
        "us-presidential-election", "trump-presidency", "elections",
        "crypto", "bitcoin", "ethereum", "ai", "tech",
        "geopolitics", "iran", "china", "russia", "ukraine",
        "economics", "inflation", "recession",
    }

    # Tag slugs to exclude (sports, entertainment, memes)
    EXCLUDE_TAG_SLUGS = {
        "sports", "nba", "nfl", "mlb", "soccer", "football", "basketball",
        "tennis", "mma", "boxing", "hockey", "cricket", "golf",
        "pop-culture", "entertainment", "music", "movies", "tv",
        "gaming", "esports",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.polymarket_config = config.get("polymarket", {})
        self.enabled = self.polymarket_config.get("enabled", True)
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = self.polymarket_config.get("cache_ttl", 600)
        self._max_age_days = self.polymarket_config.get("max_age_days", 30)

        if self.enabled:
            logger.info("Polymarket service initialized")
        else:
            logger.info("Polymarket service disabled via config")

    def is_available(self) -> bool:
        return self.enabled

    async def get_trending_markets(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch active events ordered by volume, filtered for finance/geopolitics/tech."""
        if not self.enabled:
            return []

        cache_key = f"trending:{limit}"
        if cache_key in self._cache:
            cached_time, cached_results = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_results

        try:
            results = await self._fetch_events(limit * 3)  # over-fetch to account for filtering
            # Filter to relevant topics
            filtered = self._filter_relevant(results)[:limit]
            self._cache[cache_key] = (time.time(), filtered)
            logger.info(f"Polymarket: {len(filtered)} relevant events (from {len(results)} total)")
            return filtered
        except Exception as e:
            logger.error(f"Polymarket fetch error: {e}")
            return []

    async def _fetch_events(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch events from the Gamma Markets API."""
        params = {
            "active": "true",
            "closed": "false",
            "order": "volume",
            "ascending": "false",
            "limit": min(limit, 100),
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self.API_BASE}/events", params=params)

            if resp.status_code != 200:
                logger.error(f"Polymarket API error {resp.status_code}: {resp.text[:200]}")
                return []

            events = resp.json()

        results = []
        for event in events:
            normalized = self._normalize_event(event)
            if normalized:
                results.append(normalized)

        return results

    def _filter_relevant(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter events to finance/geopolitics/tech topics, exclude sports/entertainment and stale markets."""
        filtered = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)

        for item in items:
            # Skip markets created before the recency cutoff
            created_str = item.get("created_at", "")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if created < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            tags = item.get("_tags_slugs", set())

            # Exclude sports/entertainment
            if tags & self.EXCLUDE_TAG_SLUGS:
                continue

            # Include if has relevant tag OR keyword match
            if tags & self.RELEVANT_TAG_SLUGS:
                filtered.append(item)
                continue

            # Fallback: keyword match on headline
            headline_lower = item.get("headline", "").lower()
            relevant_keywords = [
                "fed", "rate", "inflation", "tariff", "war", "iran", "china",
                "trump", "election", "recession", "gdp", "oil", "bitcoin",
                "crypto", "ai", "tech", "market", "stock", "trade",
                "nato", "military", "sanction", "nuclear",
            ]
            if any(kw in headline_lower for kw in relevant_keywords):
                filtered.append(item)

        # Remove internal tags field
        for item in filtered:
            item.pop("_tags_slugs", None)

        return filtered

    def _normalize_event(self, event: Dict) -> Optional[Dict[str, Any]]:
        """Convert a Polymarket event to common news item format."""
        title = event.get("title", "")
        if not title:
            return None

        slug = event.get("slug", "")
        volume = self._parse_float(event.get("volume", 0))
        liquidity = self._parse_float(event.get("liquidity", 0))

        # Extract tags
        tags = event.get("tags", [])
        tag_slugs = set()
        for tag in tags:
            if isinstance(tag, dict):
                tag_slugs.add(tag.get("slug", "").lower())

        # Get the first market's probability (main question)
        probability = None
        markets = event.get("markets", [])
        if markets and len(markets) > 0:
            first_market = markets[0]
            probability = self._parse_probability(first_market.get("outcomePrices"))

        return {
            "id": f"poly_{event.get('id', '')}",
            "headline": title,
            "summary": event.get("description", "")[:300] if event.get("description") else "",
            "source": "Polymarket",
            "author": "",
            "created_at": event.get("startDate", ""),
            "url": f"https://polymarket.com/event/{slug}" if slug else "",
            "category": "prediction_market",
            "symbols": [],
            "images": [],
            "sentiment": "neutral",
            "probability": probability,
            "volume": volume,
            "liquidity": liquidity,
            "_tags_slugs": tag_slugs,
        }

    @staticmethod
    def _parse_probability(outcome_prices) -> Optional[float]:
        """Parse outcomePrices field — JSON string like '["0.72","0.28"]', index 0 = Yes."""
        if not outcome_prices:
            return None
        try:
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices
            if isinstance(prices, list) and len(prices) > 0:
                return float(prices[0])
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _parse_float(value) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
