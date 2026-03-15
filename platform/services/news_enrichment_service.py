"""
News Enrichment Service.

Persists critical/high-impact news items to PostgreSQL with pgvector embeddings.
Designed to be called after NewsImpactService scoring — items with
impact_score >= threshold are enriched (embedded) and stored for theme analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


class NewsEnrichmentService:
    """Enriches and persists high-impact news with embeddings."""

    def __init__(
        self,
        config: Dict[str, Any],
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: Any,  # EmbeddingService
    ):
        ti_cfg = config.get("theme_intelligence", {})
        enrich_cfg = ti_cfg.get("enrichment", {})
        self.min_impact_score = enrich_cfg.get("min_impact_score", 60)
        self.session_factory = session_factory
        self.embedding_service = embedding_service
        self._seen_keys: set[str] = set()  # in-memory dedupe cache
        logger.info(
            f"NewsEnrichmentService initialized (min_impact={self.min_impact_score})"
        )

    async def load_recent_keys(self, days: int = 7):
        """Load recent headline hashes to avoid re-inserting."""
        from db.models import EnrichedNews

        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with self.session_factory() as session:
            stmt = select(EnrichedNews.headline).where(
                EnrichedNews.timestamp >= cutoff
            )
            result = await session.execute(stmt)
            self._seen_keys = {row[0] for row in result.fetchall()}
        logger.info(f"Loaded {len(self._seen_keys)} recent headline keys for dedupe")

    async def enrich_scored_items(
        self,
        scored_items: List[Any],  # List[ScoredNewsItem]
    ) -> int:
        """
        Filter scored items by impact threshold, embed, and persist.
        Returns count of newly stored items.
        """
        from db.models import EnrichedNews

        # Filter to critical/high only
        high_items = [
            item for item in scored_items
            if item.impact_score >= self.min_impact_score
        ]
        if not high_items:
            return 0

        # Dedupe against seen headlines
        new_items = [
            item for item in high_items
            if item.headline not in self._seen_keys
        ]
        if not new_items:
            return 0

        # Batch embed headlines + summaries
        texts = [
            f"{item.headline} {item.summary}".strip()
            for item in new_items
        ]
        embeddings: List[Optional[List[float]]] = []
        if self.embedding_service and self.embedding_service.is_available():
            try:
                embeddings = self.embedding_service.embed_batch(texts)
            except Exception as e:
                logger.warning(f"Embedding failed, storing without vectors: {e}")
                embeddings = [None] * len(new_items)
        else:
            embeddings = [None] * len(new_items)

        # Persist to DB
        stored = 0
        async with self.session_factory() as session:
            for item, embedding in zip(new_items, embeddings):
                # Parse timestamp
                ts = _parse_timestamp(item.created_at)

                news = EnrichedNews(
                    timestamp=ts,
                    headline=item.headline,
                    summary=item.summary or None,
                    embedding=embedding,
                    sentiment=item.sentiment,
                    sectors=item.sector_tags or [],
                    tickers=item.symbols or [],
                    impact_score=item.impact_score,
                    source=item.source,
                    category=item.category,
                    url=item.url or None,
                )
                session.add(news)
                self._seen_keys.add(item.headline)
                stored += 1

            await session.commit()

        if stored:
            logger.info(f"Enriched and stored {stored} high-impact news items")
        return stored

    async def enrich_backfill_item(
        self,
        headline: str,
        summary: str,
        timestamp: datetime,
        source: str = "benzinga",
        url: str | None = None,
        symbols: List[str] | None = None,
        sectors: List[str] | None = None,
        sentiment: str = "neutral",
        impact_score: float = 60.0,
        category: str | None = None,
    ) -> bool:
        """
        Enrich and store a single backfilled news item.
        Returns True if stored, False if duplicate.
        """
        from db.models import EnrichedNews

        if headline in self._seen_keys:
            return False

        # Embed
        embedding = None
        text_to_embed = f"{headline} {summary}".strip()
        if self.embedding_service and self.embedding_service.is_available():
            try:
                embedding = self.embedding_service.embed(text_to_embed)
            except Exception as e:
                logger.warning(f"Embedding failed for backfill item: {e}")

        async with self.session_factory() as session:
            news = EnrichedNews(
                timestamp=timestamp,
                headline=headline,
                summary=summary or None,
                embedding=embedding,
                sentiment=sentiment,
                sectors=sectors or [],
                tickers=symbols or [],
                impact_score=impact_score,
                source=source,
                category=category,
                url=url,
            )
            session.add(news)
            await session.commit()

        self._seen_keys.add(headline)
        return True

    async def get_enriched_news(
        self,
        days: int = 30,
        min_score: float = 0.0,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve enriched news for theme analysis.
        Returns dicts with headline, summary, embedding, sectors, tickers, etc.
        """
        from db.models import EnrichedNews

        async with self.session_factory() as session:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = (
                select(EnrichedNews)
                .where(
                    EnrichedNews.timestamp >= cutoff,
                    EnrichedNews.impact_score >= min_score,
                )
                .order_by(EnrichedNews.timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            {
                "id": str(row.id),
                "timestamp": row.timestamp.isoformat(),
                "headline": row.headline,
                "summary": row.summary,
                "embedding": row.embedding,
                "sentiment": row.sentiment,
                "sectors": row.sectors or [],
                "tickers": row.tickers or [],
                "impact_score": float(row.impact_score),
                "source": row.source,
                "category": row.category,
                "url": row.url or "",
            }
            for row in rows
        ]


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse various timestamp formats to timezone-aware datetime."""
    if not ts_str:
        return datetime.now(timezone.utc)
    try:
        # ISO format
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)
