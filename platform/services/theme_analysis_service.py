"""
Theme Analysis Service.

Periodically clusters enriched news by embedding similarity and uses
LLM to identify persistent market themes and their lifecycle stages.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


class ThemeAnalysisService:
    """Clusters news embeddings and identifies market themes via LLM."""

    def __init__(
        self,
        config: Dict[str, Any],
        session_factory: async_sessionmaker[AsyncSession],
        enrichment_service: Any,  # NewsEnrichmentService
        llm_service: Any,         # LLMService
    ):
        ti_cfg = config.get("theme_intelligence", {})
        analysis_cfg = ti_cfg.get("analysis", {})
        self.schedule_hours = analysis_cfg.get("schedule_hours", 6)
        self.lookback_days = analysis_cfg.get("lookback_days", 30)
        self.cluster_threshold = analysis_cfg.get("cluster_threshold", 0.3)
        self.max_themes = analysis_cfg.get("max_themes", 10)
        self.session_factory = session_factory
        self.enrichment_service = enrichment_service
        self.llm_service = llm_service
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._claude_client = None
        logger.info(
            f"ThemeAnalysisService initialized "
            f"(schedule={self.schedule_hours}h, lookback={self.lookback_days}d, "
            f"max_themes={self.max_themes})"
        )

    def start_background(self):
        """Start the periodic analysis loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info("Theme analysis background loop started")

    def stop(self):
        """Stop the background loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Theme analysis background loop stopped")

    async def _loop(self):
        """Run analysis on schedule."""
        while True:
            try:
                await self.run_analysis()
            except Exception as e:
                logger.error(f"Theme analysis failed: {e}", exc_info=True)
            await asyncio.sleep(self.schedule_hours * 3600)

    async def run_analysis(self) -> Dict[str, Any]:
        """
        Main analysis pipeline:
        1. Fetch enriched news from DB
        2. Cluster by embedding similarity
        3. Send clusters to LLM for theme identification
        4. Upsert themes and append history snapshots
        """
        if self._running:
            return {"status": "already_running"}

        self._running = True
        try:
            # Step 1: Fetch enriched news
            news_items = await self.enrichment_service.get_enriched_news(
                days=self.lookback_days, limit=2000
            )
            if not news_items:
                logger.info("No enriched news for theme analysis")
                return {"status": "no_data", "themes": 0}

            # Step 2: Cluster by embedding
            clusters = self._cluster_news(news_items)
            if not clusters:
                logger.info("No meaningful clusters found")
                return {"status": "no_clusters", "themes": 0}

            logger.info(f"Found {len(clusters)} news clusters")

            # Step 3: LLM analysis
            themes_data = await self._analyze_with_llm(clusters)
            if not themes_data:
                return {"status": "llm_failed", "themes": 0}

            # Step 4: Upsert themes
            count = await self._upsert_themes(themes_data)

            logger.info(f"Theme analysis complete: {count} themes updated")
            return {"status": "complete", "themes": count}

        finally:
            self._running = False

    def _cluster_news(
        self, news_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cluster news by embedding cosine similarity using agglomerative clustering.
        Returns list of cluster dicts with headlines, summaries, sectors, tickers.
        """
        # Filter items with embeddings
        items_with_emb = [
            item for item in news_items if item.get("embedding") is not None
        ]
        if len(items_with_emb) < 3:
            return []

        embeddings = np.array([item["embedding"] for item in items_with_emb])

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        # Agglomerative clustering with cosine distance
        try:
            from sklearn.cluster import AgglomerativeClustering

            distance_threshold = 1 - self.cluster_threshold  # cosine distance
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings_norm)
        except ImportError:
            logger.warning("scikit-learn not available, using simple sector grouping")
            return self._fallback_sector_grouping(items_with_emb)

        # Build cluster summaries
        clusters: Dict[int, List[Dict]] = {}
        for label, item in zip(labels, items_with_emb):
            if label == -1:  # noise
                continue
            clusters.setdefault(label, []).append(item)

        # Filter: minimum 2 items per cluster, sort by size
        result = []
        for label, items in sorted(clusters.items(), key=lambda x: -len(x[1])):
            if len(items) < 2:
                continue

            all_sectors = set()
            all_tickers = set()
            for item in items:
                all_sectors.update(item.get("sectors", []))
                all_tickers.update(item.get("tickers", []))

            # Top headlines by impact score
            items_sorted = sorted(items, key=lambda x: -x.get("impact_score", 0))

            result.append({
                "size": len(items),
                "headlines": [i["headline"] for i in items_sorted[:10]],
                "summaries": [i["summary"] for i in items_sorted[:5] if i.get("summary")],
                "sectors": list(all_sectors)[:5],
                "tickers": list(all_tickers)[:15],
                "avg_impact": np.mean([i.get("impact_score", 0) for i in items]),
                "sentiments": [i.get("sentiment", "neutral") for i in items],
            })

        return result[:self.max_themes * 2]  # send more clusters, LLM will filter

    def _fallback_sector_grouping(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple sector-based grouping when sklearn is unavailable."""
        sector_groups: Dict[str, List[Dict]] = {}
        for item in items:
            for sector in item.get("sectors", ["General"]):
                sector_groups.setdefault(sector, []).append(item)

        result = []
        for sector, group_items in sorted(
            sector_groups.items(), key=lambda x: -len(x[1])
        ):
            if len(group_items) < 2:
                continue
            items_sorted = sorted(group_items, key=lambda x: -x.get("impact_score", 0))
            all_tickers = set()
            for item in group_items:
                all_tickers.update(item.get("tickers", []))

            result.append({
                "size": len(group_items),
                "headlines": [i["headline"] for i in items_sorted[:10]],
                "summaries": [i["summary"] for i in items_sorted[:5] if i.get("summary")],
                "sectors": [sector],
                "tickers": list(all_tickers)[:15],
                "avg_impact": np.mean([i.get("impact_score", 0) for i in group_items]),
                "sentiments": [i.get("sentiment", "neutral") for i in group_items],
            })

        return result[:self.max_themes * 2]

    def _init_claude(self):
        """Lazily initialize Claude client for theme analysis."""
        if self._claude_client is not None:
            return True
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not configured for theme analysis")
                return False
            self._claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude initialized for theme analysis")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            return False

    async def _analyze_with_llm(
        self, clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Send clusters to Claude and parse structured theme response."""
        if not self._init_claude():
            logger.warning("Claude not available for theme analysis")
            return []

        # Build cluster descriptions for the prompt
        cluster_texts = []
        for i, cluster in enumerate(clusters):
            sentiment_counts = {}
            for s in cluster["sentiments"]:
                sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"

            desc = (
                f"Cluster {i + 1} ({cluster['size']} articles, "
                f"avg impact: {cluster['avg_impact']:.1f}, "
                f"dominant sentiment: {dominant_sentiment}):\n"
                f"  Sectors: {', '.join(cluster['sectors'])}\n"
                f"  Tickers: {', '.join(cluster['tickers'][:10])}\n"
                f"  Headlines:\n"
            )
            for h in cluster["headlines"][:8]:
                desc += f"    - {h}\n"
            if cluster["summaries"]:
                desc += f"  Key summary: {cluster['summaries'][0][:200]}\n"
            cluster_texts.append(desc)

        prompt = f"""You are a market theme analyst. Analyze these news clusters and identify persistent market themes.

For each meaningful theme, provide:
1. A concise theme name (e.g., "AI Infrastructure Buildout", "HBM Memory Demand", "Fed Rate Cut Cycle")
2. A 2-3 sentence narrative summary explaining the theme
3. Lifecycle stage: "emerging" (just starting), "hot" (peak activity), "cooling" (declining interest), or "faded" (mostly over)
4. Confidence score (0.0 to 1.0)
5. Top related ticker symbols (up to 10)
6. Related sectors

NEWS CLUSTERS:
{"".join(cluster_texts)}

Respond ONLY with a JSON array. No other text. Example:
[
  {{
    "name": "AI Infrastructure Buildout",
    "summary": "Major cloud providers are accelerating datacenter construction...",
    "lifecycle_stage": "hot",
    "confidence": 0.85,
    "related_tickers": ["NVDA", "AMD", "AVGO", "MSFT", "GOOGL"],
    "related_sectors": ["Technology", "Semiconductors"]
  }}
]

Return at most {self.max_themes} themes. Only include themes with clear, persistent narratives (not one-off events)."""

        try:
            import asyncio
            response = await asyncio.to_thread(
                self._claude_client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            themes = self._parse_llm_response(text)
            logger.info(f"Claude identified {len(themes)} themes")
            return themes
        except Exception as e:
            logger.error(f"Claude theme analysis failed: {e}", exc_info=True)
            return []

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response."""
        # Try direct parse first
        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse LLM theme response: {text[:200]}")
        return []

    async def _upsert_themes(self, themes_data: List[Dict[str, Any]]) -> int:
        """Replace all active themes with fresh analysis results."""
        from db.models import Theme, ThemeHistory

        now = datetime.now(timezone.utc)
        new_names = {td.get("name", "").strip() for td in themes_data if td.get("name", "").strip()}
        count = 0

        async with self.session_factory() as session:
            # Fade all themes not in the new results
            all_themes_stmt = select(Theme).where(Theme.lifecycle_stage != "faded")
            result = await session.execute(all_themes_stmt)
            existing_themes = result.scalars().all()

            for theme in existing_themes:
                if theme.name not in new_names:
                    theme.lifecycle_stage = "faded"
                    theme.last_updated = now

            # Upsert new themes
            for td in themes_data:
                name = td.get("name", "").strip()
                if not name:
                    continue

                # Try to find existing theme by name
                stmt = select(Theme).where(Theme.name == name)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    existing.summary = td.get("summary", existing.summary)
                    existing.lifecycle_stage = td.get("lifecycle_stage", existing.lifecycle_stage)
                    existing.confidence = td.get("confidence", existing.confidence)
                    existing.related_tickers = td.get("related_tickers", existing.related_tickers)
                    existing.related_sectors = td.get("related_sectors", existing.related_sectors)
                    existing.news_count = td.get("news_count", existing.news_count)
                    existing.last_updated = now
                    theme_id = existing.id
                else:
                    theme = Theme(
                        name=name,
                        summary=td.get("summary", ""),
                        lifecycle_stage=td.get("lifecycle_stage", "emerging"),
                        confidence=td.get("confidence", 0.5),
                        related_tickers=td.get("related_tickers", []),
                        related_sectors=td.get("related_sectors", []),
                        news_count=td.get("news_count", 0),
                        first_seen=now,
                        last_updated=now,
                    )
                    session.add(theme)
                    await session.flush()
                    theme_id = theme.id

                # Append history snapshot
                snapshot = ThemeHistory(
                    theme_id=theme_id,
                    snapshot_at=now,
                    lifecycle_stage=td.get("lifecycle_stage", "emerging"),
                    confidence=td.get("confidence", 0.5),
                    summary=td.get("summary", ""),
                    news_count=td.get("news_count", 0),
                )
                session.add(snapshot)
                count += 1

            await session.commit()

        return count

    async def get_active_themes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get current active themes (not faded) for frontend display."""
        from db.models import Theme

        async with self.session_factory() as session:
            stmt = (
                select(Theme)
                .where(Theme.lifecycle_stage != "faded")
                .order_by(Theme.confidence.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            themes = result.scalars().all()

        return [
            {
                "id": str(t.id),
                "name": t.name,
                "summary": t.summary,
                "lifecycle_stage": t.lifecycle_stage,
                "confidence": float(t.confidence),
                "related_tickers": t.related_tickers or [],
                "related_sectors": t.related_sectors or [],
                "news_count": t.news_count,
                "first_seen": t.first_seen.isoformat() if t.first_seen else None,
                "last_updated": t.last_updated.isoformat() if t.last_updated else None,
            }
            for t in themes
        ]
