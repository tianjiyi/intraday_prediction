"""
Sector Trend Aggregation Service.

Aggregates scored news items by sector for rolling time windows (1h, 6h, 24h).
Ranks sectors by impact sum + weighted critical/high counts.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.news_impact_service import ScoredNewsItem

logger = logging.getLogger(__name__)

WINDOW_SECONDS = {
    '1h': 3600,
    '6h': 21600,
    '24h': 86400,
}


@dataclass
class SectorTrend:
    sector: str = ''
    window: str = '6h'
    rank_score: float = 0.0
    impact_sum: float = 0.0
    impact_avg: float = 0.0
    critical_count: int = 0
    high_count: int = 0
    dominant_sentiment: str = 'neutral'
    top_headline: str = ''
    item_count: int = 0
    last_updated: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SectorTrendService:
    """Aggregates scored news by sector and provides ranked trend lists."""

    def __init__(self, config: Dict[str, Any]):
        scoring_cfg = config.get('news_scoring', {})
        self.taxonomy: List[str] = scoring_cfg.get('sector_taxonomy', [
            'Technology', 'Semiconductors', 'Financials', 'Energy',
            'Healthcare', 'Industrials', 'Consumer Discretionary',
            'Consumer Staples', 'Utilities', 'Real Estate',
            'Communication Services', 'Materials', 'Crypto',
        ])
        self.default_window = scoring_cfg.get('default_window', '6h')
        logger.info(f"SectorTrendService initialized ({len(self.taxonomy)} sectors)")

    def get_trending(
        self,
        scored_items: List[ScoredNewsItem],
        window: Optional[str] = None,
        limit: int = 8,
        critical_only: bool = False,
    ) -> List[SectorTrend]:
        """Return top sectors ranked by impact for the given time window."""
        window = window or self.default_window
        trends = self.aggregate(scored_items, window, critical_only)
        trends.sort(key=lambda t: t.rank_score, reverse=True)
        return trends[:limit]

    def aggregate(
        self,
        scored_items: List[ScoredNewsItem],
        window: str = '6h',
        critical_only: bool = False,
    ) -> List[SectorTrend]:
        """Group items by sector within the time window and compute metrics."""
        cutoff = time.time() - WINDOW_SECONDS.get(window, 21600)

        # Filter by window
        in_window = [
            item for item in scored_items
            if item.scored_at >= cutoff
        ]

        # Optional: critical-only filter
        if critical_only:
            in_window = [i for i in in_window if i.impact_tier == 'critical']

        # Group by sector
        sector_items: Dict[str, List[ScoredNewsItem]] = defaultdict(list)
        for item in in_window:
            for sector in item.sector_tags:
                if sector in self.taxonomy:
                    sector_items[sector].append(item)

        # Build trends
        now_iso = datetime.now(timezone.utc).isoformat()
        trends = []
        for sector in self.taxonomy:
            items = sector_items.get(sector, [])
            if not items:
                continue

            scores = [i.decayed_score for i in items]
            critical = sum(1 for i in items if i.impact_tier == 'critical')
            high = sum(1 for i in items if i.impact_tier == 'high')
            impact_sum = sum(scores)
            impact_avg = impact_sum / len(scores) if scores else 0

            # Dominant sentiment
            sentiments = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            for i in items:
                sentiments[i.sentiment] = sentiments.get(i.sentiment, 0) + 1
            dominant = max(sentiments, key=sentiments.get)

            # Top headline (highest scored item)
            top_item = max(items, key=lambda i: i.decayed_score)

            rank_score = impact_sum + 10 * critical + 4 * high

            trends.append(SectorTrend(
                sector=sector,
                window=window,
                rank_score=round(rank_score, 1),
                impact_sum=round(impact_sum, 1),
                impact_avg=round(impact_avg, 1),
                critical_count=critical,
                high_count=high,
                dominant_sentiment=dominant,
                top_headline=top_item.headline[:200],
                item_count=len(items),
                last_updated=now_iso,
            ))

        return trends
