"""
News Backfill Service.

Fetches historical news from Alpaca News API and feeds them through
the enrichment pipeline for theme analysis bootstrapping.

Uses a sliding-window approach: fetches 50 items at a time sorted desc,
then moves the end cursor to just before the oldest item's timestamp.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsBackfillService:
    """Backfills historical news from Alpaca into the enrichment pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        enrichment_service: Any,  # NewsEnrichmentService
        news_service: Any,        # NewsService (Alpaca)
        impact_service: Any | None = None,  # NewsImpactService (optional)
    ):
        ti_cfg = config.get("theme_intelligence", {})
        bf_cfg = ti_cfg.get("backfill", {})
        self.default_months = bf_cfg.get("default_months", 6)
        self.batch_size = min(bf_cfg.get("batch_size", 50), 50)  # Alpaca max is 50
        self.enrichment_service = enrichment_service
        self.news_service = news_service
        self.impact_service = impact_service
        self._running = False
        self._progress = {"status": "idle", "total": 0, "processed": 0, "stored": 0}

        if news_service and news_service.is_available():
            logger.info("NewsBackfillService: Alpaca News API configured")
        else:
            logger.warning("NewsBackfillService: Alpaca News API not available")

    @property
    def progress(self) -> Dict[str, Any]:
        return dict(self._progress)

    async def backfill(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Dict[str, Any]:
        """
        Backfill historical news from Alpaca using sliding window pagination.

        Args:
            start_date: ISO date string (e.g. '2025-09-14'). Defaults to N months ago.
            end_date: ISO date string. Defaults to today.

        Returns:
            Summary dict with counts.
        """
        if self._running:
            return {"error": "Backfill already running", **self._progress}

        if not self.news_service or not self.news_service.is_available():
            return {"error": "Alpaca News API not available"}

        self._running = True
        now = datetime.now(timezone.utc)

        # Load existing headlines to avoid duplicates
        await self.enrichment_service.load_recent_keys(days=self.default_months * 30 + 30)

        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_dt = now - timedelta(days=self.default_months * 30)
            start_date = start_dt.strftime("%Y-%m-%d")

        self._progress = {
            "status": "running",
            "start_date": start_date,
            "end_date": end_date,
            "total": 0,
            "processed": 0,
            "stored": 0,
        }

        logger.info(f"Starting Alpaca backfill: {start_date} to {end_date}")

        try:
            from alpaca.data.requests import NewsRequest

            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            cursor_end = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )

            total_stored = 0
            total_fetched = 0
            page = 0

            while cursor_end > start_dt:
                page += 1
                request = NewsRequest(
                    start=start_dt,
                    end=cursor_end,
                    limit=self.batch_size,
                    sort="desc",
                )
                response = self.news_service.news_client.get_news(request)

                # Parse items
                raw_data = response.data if hasattr(response, "data") else {}
                news_list = (
                    raw_data.get("news", []) if isinstance(raw_data, dict) else []
                )

                if not news_list:
                    break

                items = []
                oldest_ts = None
                for item in news_list:
                    d = (
                        item
                        if isinstance(item, dict)
                        else (
                            item.model_dump()
                            if hasattr(item, "model_dump")
                            else item.__dict__
                        )
                    )
                    created = d.get("created_at")
                    # Track oldest timestamp for sliding window
                    if hasattr(created, "isoformat"):
                        if oldest_ts is None or created < oldest_ts:
                            oldest_ts = created

                    items.append(
                        {
                            "headline": d.get("headline") or "",
                            "summary": d.get("summary") or "",
                            "source": d.get("source") or "Alpaca",
                            "created_at": (
                                created.isoformat()
                                if hasattr(created, "isoformat")
                                else str(created or "")
                            ),
                            "url": d.get("url") or "",
                            "symbols": d.get("symbols") or [],
                        }
                    )

                total_fetched += len(items)
                self._progress["total"] = total_fetched

                # Process through enrichment pipeline
                stored = await self._process_items(items)
                total_stored += stored
                self._progress["processed"] = total_fetched
                self._progress["stored"] = total_stored

                logger.info(
                    f"Backfill page {page}: {len(items)} fetched, {stored} stored, "
                    f"total={total_fetched}/{total_stored}"
                )

                # Slide window: move end cursor to 1 second before oldest item
                if oldest_ts is None or len(items) < self.batch_size:
                    break
                cursor_end = oldest_ts - timedelta(seconds=1)

                # Rate limit
                await asyncio.sleep(0.3)

            self._progress["status"] = "complete"
            logger.info(
                f"Backfill complete: {total_fetched} fetched, {total_stored} stored"
            )
            return {
                "status": "complete",
                "fetched": total_fetched,
                "stored": total_stored,
                "start_date": start_date,
                "end_date": end_date,
            }

        except Exception as e:
            logger.error(f"Backfill failed: {e}", exc_info=True)
            self._progress["status"] = f"error: {e}"
            return {"error": str(e), **self._progress}

        finally:
            self._running = False

    async def _process_items(self, items: List[Dict]) -> int:
        """Process a batch of Alpaca news items through enrichment."""
        stored = 0
        for item in items:
            headline = item.get("headline", "")
            if not headline:
                continue

            summary = item.get("summary", "")
            created_at = item.get("created_at", "")
            symbols = item.get("symbols", [])

            ts = _parse_timestamp(created_at)

            # Score if impact service available
            impact_score = 60.0
            sentiment = "neutral"
            category = None
            sectors: List[str] = []

            if self.impact_service:
                try:
                    scored = self.impact_service.score_item(
                        {
                            "headline": headline,
                            "summary": summary,
                            "source": item.get("source", "Alpaca"),
                            "symbols": symbols,
                            "created_at": created_at,
                            "url": item.get("url", ""),
                        },
                        [],
                    )
                    impact_score = scored.impact_score
                    sentiment = scored.sentiment
                    category = scored.category
                    sectors = scored.sector_tags or []

                    if impact_score < self.enrichment_service.min_impact_score:
                        continue
                except Exception as e:
                    logger.debug(f"Scoring failed for backfill item: {e}")

            result = await self.enrichment_service.enrich_backfill_item(
                headline=headline,
                summary=summary[:1000] if summary else "",
                timestamp=ts,
                source=item.get("source", "alpaca").lower(),
                url=item.get("url"),
                symbols=symbols,
                sectors=sectors,
                sentiment=sentiment,
                impact_score=impact_score,
                category=category,
            )
            if result:
                stored += 1

        return stored


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse various timestamp formats to timezone-aware datetime."""
    if not ts_str:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)
