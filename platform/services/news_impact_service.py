"""
News Impact Scoring Service.

Scores news items on a 0-100 scale using a weighted heuristic formula:
  impact_score = 100 * (w_source*S + w_recency*R + w_breadth*B + w_surprise*U + w_sentiment*T + w_linkage*L)

Classifies into tiers: critical (>=75), high (60-74), medium (40-59), low (<40).
Resolves sector tags, canonical categories, and market breadth.
"""

import hashlib
import math
import re
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords for enhanced classification
# ---------------------------------------------------------------------------

EARNINGS_KEYWORDS = {
    'earnings', 'revenue', 'eps', 'beat', 'miss', 'guidance', 'forecast',
    'quarterly', 'q1', 'q2', 'q3', 'q4', 'profit', 'loss', 'margin',
    'outlook', 'report', 'results', 'fiscal',
}

POLICY_KEYWORDS = {
    'fed', 'fomc', 'rate', 'inflation', 'cpi', 'ppi', 'gdp', 'employment',
    'jobs', 'stimulus', 'treasury', 'yield', 'debt ceiling', 'fiscal',
    'monetary', 'taper', 'quantitative', 'powell', 'yellen', 'central bank',
    'interest rate', 'rate cut', 'rate hike', 'dovish', 'hawkish',
}

MACRO_KEYWORDS = {
    'economy', 'economic', 'recession', 'growth', 'unemployment', 'consumer',
    'housing', 'manufacturing', 'pmi', 'ism', 'retail sales', 'trade',
    'deficit', 'surplus', 'gdp',
}

GEO_KEYWORDS = {
    'war', 'conflict', 'sanction', 'military', 'geopolitic', 'troops',
    'iran', 'china', 'russia', 'ukraine', 'taiwan', 'tariff', 'israel',
    'missile', 'nuclear', 'invasion', 'nato', 'attack', 'strike',
    'ceasefire', 'peace', 'tension', 'diplomacy', 'defense',
}

MARKET_STRUCTURE_KEYWORDS = {
    'ipo', 'merger', 'acquisition', 'buyout', 'spinoff', 'split',
    'delist', 'listing', 'sec', 'regulation', 'compliance', 'etf',
    'index rebalance', 'options expiration', 'witching',
}

BROAD_MARKET_KEYWORDS = {
    'market', 'markets', 'wall street', 's&p', 'dow', 'nasdaq', 'russell',
    'broad', 'sell-off', 'selloff', 'rally', 'correction', 'bull', 'bear',
    'risk-on', 'risk-off', 'volatility', 'vix',
}

SECTOR_HEADLINE_KEYWORDS = {
    'Technology': {'tech', 'software', 'cloud', 'saas', 'ai', 'data center', 'chip', 'app'},
    'Semiconductors': {'semiconductor', 'chip', 'wafer', 'foundry', 'fab', 'gpu', 'cpu', 'memory', 'asml', 'tsmc'},
    'Financials': {'bank', 'financial', 'insurance', 'lending', 'mortgage', 'credit'},
    'Energy': {'oil', 'gas', 'energy', 'opec', 'crude', 'petroleum', 'solar', 'wind', 'renewable'},
    'Healthcare': {'health', 'pharma', 'biotech', 'fda', 'drug', 'vaccine', 'hospital', 'medical'},
    'Industrials': {'industrial', 'manufacturing', 'aerospace', 'defense', 'transport', 'logistics'},
    'Consumer Discretionary': {'retail', 'consumer', 'luxury', 'auto', 'travel', 'leisure', 'restaurant'},
    'Consumer Staples': {'food', 'beverage', 'grocery', 'household', 'tobacco', 'staple'},
    'Utilities': {'utility', 'power', 'electric', 'water', 'grid'},
    'Real Estate': {'real estate', 'reit', 'housing', 'property', 'mortgage'},
    'Communication Services': {'media', 'telecom', 'streaming', 'social media', 'advertising'},
    'Materials': {'mining', 'steel', 'copper', 'gold', 'silver', 'lithium', 'chemical', 'material'},
    'Crypto': {'crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'token', 'btc', 'eth'},
}

BULLISH_KEYWORDS = {
    'surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'bullish',
    'optimistic', 'growth', 'profit', 'beat', 'exceed', 'record',
    'high', 'upgrade', 'buy', 'breakout', 'moon', 'boost', 'outperform',
}

BEARISH_KEYWORDS = {
    'fall', 'drop', 'decline', 'plunge', 'crash', 'bearish',
    'pessimistic', 'loss', 'miss', 'cut', 'downgrade', 'sell',
    'low', 'weak', 'concern', 'fear', 'dump', 'breakdown', 'underperform',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoredNewsItem:
    # Original fields
    id: str = ''
    headline: str = ''
    summary: str = ''
    source: str = ''
    author: str = ''
    created_at: str = ''
    url: str = ''
    symbols: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

    # Scoring fields
    impact_score: float = 0.0
    impact_tier: str = 'low'
    sector_tags: List[str] = field(default_factory=list)
    category: str = 'company_specific'
    sentiment: str = 'neutral'
    sentiment_strength: float = 0.0
    market_breadth: str = 'single_stock'
    horizon: str = 'intraday'
    impact_reasons: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    dedupe_key: str = ''
    scored_at: float = 0.0
    decayed_score: float = 0.0

    # Pass-through optional fields (Twitter / Polymarket)
    likes: Optional[int] = None
    retweets: Optional[int] = None
    views: Optional[int] = None
    probability: Optional[float] = None
    volume: Optional[float] = None
    liquidity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class NewsImpactService:
    """Scores news items and resolves sector tags."""

    def __init__(self, config: Dict[str, Any]):
        scoring_cfg = config.get('news_scoring', {})

        self.weights = scoring_cfg.get('weights', {
            'source': 0.20, 'recency': 0.20, 'breadth': 0.20,
            'surprise': 0.15, 'sentiment': 0.10, 'linkage': 0.15,
        })
        self.source_trust = scoring_cfg.get('source_trust', {
            'Reuters': 0.95, 'Bloomberg': 0.95, 'Benzinga': 0.75,
            'Alpaca': 0.70, 'X.com': 0.50, 'Polymarket': 0.60,
        })
        self.critical_threshold = scoring_cfg.get('critical_threshold', 75)
        self.high_threshold = scoring_cfg.get('high_threshold', 60)
        self.medium_threshold = scoring_cfg.get('medium_threshold', 40)

        self.decay_half_lives = scoring_cfg.get('decay_half_lives', {
            'macro': 14400, 'earnings': 7200, 'policy': 21600,
            'geopolitics': 28800, 'company_specific': 3600,
            'market_structure': 7200,
        })
        self.symbol_sector_map: Dict[str, List[str]] = scoring_cfg.get('symbol_sector_map', {})
        self.sector_taxonomy: List[str] = scoring_cfg.get('sector_taxonomy', list(SECTOR_HEADLINE_KEYWORDS.keys()))

        logger.info(f"NewsImpactService initialized (weights={self.weights}, "
                     f"thresholds={self.critical_threshold}/{self.high_threshold}/{self.medium_threshold})")

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------

    def batch_score(
        self,
        raw_items: List[Dict[str, Any]],
        existing_buffer: List[ScoredNewsItem],
    ) -> List[ScoredNewsItem]:
        """Deduplicate, score, and return sorted items (highest score first)."""
        existing_keys = {item.dedupe_key for item in existing_buffer if item.dedupe_key}
        recent_headlines = [item.headline for item in existing_buffer[:100]]

        scored = []
        for raw in raw_items:
            dedupe_key = self.compute_dedupe_key(raw.get('headline', ''), raw.get('created_at', ''))
            if dedupe_key in existing_keys:
                continue
            existing_keys.add(dedupe_key)

            item = self.score_item(raw, recent_headlines)
            item.dedupe_key = dedupe_key
            scored.append(item)

        scored.sort(key=lambda x: x.impact_score, reverse=True)

        tiers = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for s in scored:
            tiers[s.impact_tier] = tiers.get(s.impact_tier, 0) + 1
        if scored:
            logger.info(f"Scored {len(scored)} items: {tiers['critical']} critical, "
                        f"{tiers['high']} high, {tiers['medium']} medium, {tiers['low']} low")

        return scored

    def score_item(self, raw: Dict[str, Any], recent_headlines: List[str]) -> ScoredNewsItem:
        """Score a single raw news item."""
        headline = raw.get('headline', '')
        summary = raw.get('summary', '')
        text = f"{headline} {summary}".lower()
        source = raw.get('source', '')
        symbols = raw.get('symbols', []) or []
        created_at = raw.get('created_at', '')
        sentiment = raw.get('sentiment', '')

        # Resolve metadata
        sector_tags = self.resolve_sectors(symbols, text)
        category = self.classify_category(raw, text)
        if not sentiment:
            sentiment = self._classify_sentiment(text)
        market_breadth = self.classify_breadth(symbols, sector_tags, text)
        horizon = self.classify_horizon(category, source)

        # Compute component scores (each 0-1)
        source_score = self.compute_source_score(source)
        recency_score = self.compute_recency_score(created_at)
        breadth_score = self.compute_breadth_score(market_breadth)
        surprise_score = self.compute_surprise_score(headline, recent_headlines)
        sentiment_score, sentiment_strength = self.compute_sentiment_score(sentiment, text)
        linkage_score = self.compute_linkage_score(symbols, sector_tags)

        # Weighted sum
        w = self.weights
        raw_score = (
            w.get('source', 0.20) * source_score
            + w.get('recency', 0.20) * recency_score
            + w.get('breadth', 0.20) * breadth_score
            + w.get('surprise', 0.15) * surprise_score
            + w.get('sentiment', 0.10) * sentiment_score
            + w.get('linkage', 0.15) * linkage_score
        )
        impact_score = round(min(100.0, max(0.0, 100.0 * raw_score)), 1)

        # Tier
        if impact_score >= self.critical_threshold:
            impact_tier = 'critical'
        elif impact_score >= self.high_threshold:
            impact_tier = 'high'
        elif impact_score >= self.medium_threshold:
            impact_tier = 'medium'
        else:
            impact_tier = 'low'

        # Impact reasons
        reasons = []
        if source_score >= 0.8:
            reasons.append(f"Trusted source ({source})")
        if recency_score >= 0.8:
            reasons.append("Very recent")
        if breadth_score >= 0.8:
            reasons.append(f"Broad market breadth ({market_breadth})")
        if surprise_score >= 0.7:
            reasons.append("Novel headline")
        if sentiment_strength >= 0.6:
            reasons.append(f"Strong {sentiment} sentiment")
        if linkage_score >= 0.6:
            reasons.append(f"Cross-sector linkage ({len(sector_tags)} sectors)")
        if category in ('policy', 'macro'):
            reasons.append(f"{category.title()} catalyst")

        now = time.time()
        return ScoredNewsItem(
            id=raw.get('id', ''),
            headline=headline,
            summary=summary,
            source=source,
            author=raw.get('author', ''),
            created_at=created_at,
            url=raw.get('url', ''),
            symbols=symbols,
            images=raw.get('images', []) or [],
            impact_score=impact_score,
            impact_tier=impact_tier,
            sector_tags=sector_tags,
            category=category,
            sentiment=sentiment,
            sentiment_strength=round(sentiment_strength, 3),
            market_breadth=market_breadth,
            horizon=horizon,
            impact_reasons=reasons,
            confidence_score=round(min(1.0, 0.4 + 0.1 * len(reasons)), 2),
            scored_at=now,
            decayed_score=impact_score,
            likes=raw.get('likes'),
            retweets=raw.get('retweets'),
            views=raw.get('views'),
            probability=raw.get('probability'),
            volume=raw.get('volume'),
            liquidity=raw.get('liquidity'),
        )

    # ---------------------------------------------------------------
    # Component scorers
    # ---------------------------------------------------------------

    def compute_source_score(self, source: str) -> float:
        """Lookup source trust. Default 0.5 for unknown sources."""
        return self.source_trust.get(source, 0.5)

    @staticmethod
    def compute_recency_score(created_at: str) -> float:
        """Exponential decay: 1.0 at t=0, ~0.5 at 1h, ~0.1 at 3h."""
        if not created_at:
            return 0.3
        try:
            if isinstance(created_at, str):
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                dt = created_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_seconds = max(0, (datetime.now(timezone.utc) - dt).total_seconds())
            return math.exp(-age_seconds / 3600)  # half-life ~42 min
        except Exception:
            return 0.3

    @staticmethod
    def compute_breadth_score(market_breadth: str) -> float:
        return {'broad_market': 1.0, 'sector': 0.7, 'single_stock': 0.3}.get(market_breadth, 0.3)

    @staticmethod
    def compute_surprise_score(headline: str, recent_headlines: List[str]) -> float:
        """1 - max Jaccard similarity with recent headlines."""
        if not headline or not recent_headlines:
            return 0.7  # novel by default

        words = set(re.findall(r'\w+', headline.lower()))
        if not words:
            return 0.7

        max_sim = 0.0
        for prev in recent_headlines:
            prev_words = set(re.findall(r'\w+', prev.lower()))
            if not prev_words:
                continue
            intersection = len(words & prev_words)
            union = len(words | prev_words)
            if union > 0:
                max_sim = max(max_sim, intersection / union)

        return round(1.0 - max_sim, 3)

    @staticmethod
    def compute_sentiment_score(sentiment: str, text: str) -> tuple:
        """Returns (score 0-1, strength 0-1). Strong sentiment = higher score."""
        bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        total = bull_count + bear_count
        if total == 0:
            return 0.3, 0.0

        strength = min(1.0, total / 6.0)
        # Strong directional sentiment is more impactful
        polarity = abs(bull_count - bear_count) / total
        score = 0.3 + 0.7 * strength * polarity
        return round(score, 3), round(strength, 3)

    @staticmethod
    def compute_linkage_score(symbols: List[str], sector_tags: List[str]) -> float:
        """More distinct sectors/symbols = higher cross-linkage."""
        n_sectors = len(set(sector_tags))
        n_symbols = len(set(symbols))
        if n_sectors >= 3 or n_symbols >= 5:
            return 1.0
        if n_sectors >= 2 or n_symbols >= 3:
            return 0.7
        if n_symbols >= 2:
            return 0.5
        return 0.2

    # ---------------------------------------------------------------
    # Classification helpers
    # ---------------------------------------------------------------

    def resolve_sectors(self, symbols: List[str], text: str) -> List[str]:
        """Resolve sectors from symbols (static map) + headline keywords."""
        sectors = set()

        # Static map lookup
        for sym in symbols:
            mapped = self.symbol_sector_map.get(sym.upper(), [])
            sectors.update(mapped)

        # Headline keyword fallback
        for sector, keywords in SECTOR_HEADLINE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                sectors.add(sector)

        # Filter to taxonomy
        if self.sector_taxonomy:
            sectors = sectors & set(self.sector_taxonomy)

        return sorted(sectors) if sectors else []

    @staticmethod
    def classify_category(raw: Dict, text: str) -> str:
        """Classify into 6 canonical categories."""
        source = raw.get('source', '')
        if source == 'Polymarket':
            return 'market_structure'

        # Score all categories
        earnings_score = sum(1 for kw in EARNINGS_KEYWORDS if kw in text)
        policy_score = sum(1 for kw in POLICY_KEYWORDS if kw in text)
        geo_score = sum(1 for kw in GEO_KEYWORDS if kw in text)
        ms_score = sum(1 for kw in MARKET_STRUCTURE_KEYWORDS if kw in text)
        macro_score = sum(1 for kw in MACRO_KEYWORDS if kw in text)

        # Pick highest-scoring category (with minimum thresholds)
        scores = []
        if earnings_score >= 2:
            scores.append(('earnings', earnings_score))
        if policy_score >= 2:
            scores.append(('policy', policy_score))
        if geo_score >= 2:
            scores.append(('geopolitics', geo_score))
        if macro_score >= 1:
            scores.append(('macro', macro_score))
        if ms_score >= 2:
            scores.append(('market_structure', ms_score))

        if scores:
            return max(scores, key=lambda x: x[1])[0]

        return 'company_specific'

    BROAD_MARKET_SYMBOLS = {'SPY', 'QQQ', 'IWM', 'DIA', 'VIX', 'VOO', 'VTI'}

    @classmethod
    def classify_breadth(cls, symbols: List[str], sector_tags: List[str], text: str) -> str:
        broad_match = sum(1 for kw in BROAD_MARKET_KEYWORDS if kw in text)
        has_broad_symbol = bool(set(s.upper() for s in symbols) & cls.BROAD_MARKET_SYMBOLS)
        if broad_match >= 2 or has_broad_symbol:
            return 'broad_market'
        if not symbols:
            return 'single_stock'
        if len(set(sector_tags)) >= 2 or len(set(symbols)) >= 3:
            return 'sector'
        return 'single_stock'

    @staticmethod
    def classify_horizon(category: str, source: str) -> str:
        if category in ('policy', 'geopolitics'):
            return 'multi_day'
        if category == 'earnings':
            return 'immediate'
        if source == 'Polymarket':
            return 'multi_day'
        return 'intraday'

    @staticmethod
    def _classify_sentiment(text: str) -> str:
        bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        if bull > bear:
            return 'bullish'
        if bear > bull:
            return 'bearish'
        return 'neutral'

    # ---------------------------------------------------------------
    # Deduplication
    # ---------------------------------------------------------------

    @staticmethod
    def compute_dedupe_key(headline: str, created_at: str) -> str:
        """Normalized headline hash + 30-min time bucket."""
        normalized = re.sub(r'[^a-z0-9 ]', '', headline.lower().strip())
        # 30-minute bucket
        try:
            if isinstance(created_at, str) and created_at:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                bucket = int(dt.timestamp()) // 1800
            else:
                bucket = int(time.time()) // 1800
        except Exception:
            bucket = int(time.time()) // 1800

        raw_key = f"{normalized}:{bucket}"
        return hashlib.sha256(raw_key.encode()).hexdigest()[:16]

    # ---------------------------------------------------------------
    # Decay
    # ---------------------------------------------------------------

    def apply_decay(self, item: ScoredNewsItem, now: Optional[float] = None) -> float:
        """Apply exponential decay based on category half-life. Updates both score and tier."""
        if now is None:
            now = time.time()
        age = max(0, now - item.scored_at)
        half_life = self.decay_half_lives.get(item.category, 7200)
        decay_factor = math.exp(-0.693 * age / half_life)  # ln(2) ≈ 0.693
        item.decayed_score = round(item.impact_score * decay_factor, 1)

        # Re-tier based on decayed score
        if item.decayed_score >= self.critical_threshold:
            item.impact_tier = 'critical'
        elif item.decayed_score >= self.high_threshold:
            item.impact_tier = 'high'
        elif item.decayed_score >= self.medium_threshold:
            item.impact_tier = 'medium'
        else:
            item.impact_tier = 'low'

        return item.decayed_score
