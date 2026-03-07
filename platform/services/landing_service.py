"""
Landing Command Center Service.

Provides market pulse, macro tape, movers/losers, hot themes, and catalyst clock
data for the landing page command center.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LandingService:
    def __init__(self, config: Dict[str, Any], news_monitor, catalyst_service=None):
        self.config = config
        self.news_monitor = news_monitor
        self.catalyst_service = catalyst_service
        self.landing_config = config.get('landing', {})

        # Alpaca client for snapshots
        self._stock_client = None
        self._init_alpaca()

        # Caches
        self._pulse_cache: Optional[Dict] = None
        self._pulse_cache_at: float = 0
        self._movers_cache: Optional[Dict] = None
        self._movers_cache_at: float = 0
        self._themes_cache: Optional[Dict] = None
        self._themes_cache_at: float = 0
        self._tape_cache: Optional[Dict] = None
        self._tape_cache_at: float = 0

        # Config
        self._pulse_ttl = self.landing_config.get('pulse_cache_ttl', 60)
        self._movers_ttl = self.landing_config.get('movers_cache_ttl', 60)
        self._themes_ttl = self.landing_config.get('themes_cache_ttl', 300)
        self._tape_ttl = self.landing_config.get('macro_tape_cache_ttl', 60)
        self._index_symbols = self.landing_config.get('index_symbols', ['SPY', 'QQQ', 'IWM'])
        self._vol_chain = self.landing_config.get('vol_proxy_chain', ['VIX', 'VIXY', 'UVXY'])
        self._dollar_proxy = self.landing_config.get('macro_proxy_dollar', 'UUP')
        self._rates_proxy = self.landing_config.get('macro_proxy_rates', 'TLT')
        self._macro_tape_config = self.landing_config.get('macro_tape', [])

        # Build movers universe from symbol_sector_map
        scoring_config = config.get('news_scoring', {})
        self._sector_map: Dict[str, List[str]] = scoring_config.get('symbol_sector_map', {})
        self._movers_symbols = [
            s for s in self._sector_map.keys()
            if not s.endswith('USD') and s not in ('BTC', 'ETH')
        ]

    def _init_alpaca(self):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            api_key = os.environ.get('ALPACA_KEY_ID', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
            if api_key and secret_key:
                self._stock_client = StockHistoricalDataClient(
                    api_key=api_key, secret_key=secret_key
                )
                logger.info("Landing service: Alpaca stock client initialized")
            else:
                logger.warning("Landing service: No Alpaca credentials, snapshots disabled")
        except Exception as e:
            logger.warning(f"Landing service: Alpaca init failed: {e}")

    def _get_snapshots(self, symbols: List[str]) -> Dict:
        """Fetch snapshots, filtering out non-stock symbols like VIX."""
        if not self._stock_client or not symbols:
            return {}
        # VIX is an index, not a stock — skip it from stock snapshot requests
        stock_syms = [s for s in symbols if s not in ('VIX', '^VIX')]
        if not stock_syms:
            return {}
        try:
            from alpaca.data.requests import StockSnapshotRequest
            request = StockSnapshotRequest(symbol_or_symbols=stock_syms)
            return self._stock_client.get_stock_snapshot(request)
        except Exception as e:
            logger.warning(f"Snapshot fetch failed for {stock_syms}: {e}")
            return {}

    def _get_snapshot_pct(self, snap) -> Tuple[float, float]:
        """Return (price, pct_change_1d) from a snapshot's daily_bar."""
        if not snap or not snap.daily_bar:
            return (0.0, 0.0)
        price = float(snap.daily_bar.close)
        open_price = float(snap.daily_bar.open)
        pct = ((price - open_price) / open_price * 100) if open_price else 0
        return (price, pct)

    def _resolve_vol_snapshot(self, snapshots: Dict) -> Tuple[Optional[Any], str, bool]:
        """Try VIX fallback chain. Return (snapshot, symbol_used, is_fallback)."""
        for i, sym in enumerate(self._vol_chain):
            snap = snapshots.get(sym)
            if snap and snap.daily_bar:
                return (snap, sym, i > 0)
        return (None, self._vol_chain[0] if self._vol_chain else 'VIX', True)

    # ------------------------------------------------------------------
    # Market Pulse (spec §4.1.1)
    # ------------------------------------------------------------------

    def get_market_pulse(self) -> Dict[str, Any]:
        now = time.time()
        if self._pulse_cache and (now - self._pulse_cache_at) < self._pulse_ttl:
            return self._pulse_cache

        # Fetch all snapshots we need in one batch
        all_syms = list(set(
            self._vol_chain + [self._dollar_proxy, self._rates_proxy] +
            self._index_symbols + self._movers_symbols
        ))
        snapshots = self._get_snapshots(all_syms)

        # --- Volatility (VIX-based) ---
        vol_snap, vol_source, vol_is_fallback = self._resolve_vol_snapshot(snapshots)
        vol_level = 0.0
        vol_change_1d = 0.0
        if vol_snap:
            vol_level, vol_change_1d = self._get_snapshot_pct(vol_snap)
            vol_level = float(vol_snap.daily_bar.close) if vol_snap.daily_bar else 0

        if vol_level >= 22 or vol_change_1d >= 8:
            volatility_state = 'high'
        elif vol_level >= 16 or vol_change_1d >= 4:
            volatility_state = 'elevated'
        else:
            volatility_state = 'normal'

        # vol_component: high VIX = negative risk. Normalize: 15=neutral, >25=very negative
        vol_component = max(-1.0, min(1.0, -(vol_level - 15) / 10)) if vol_level > 0 else 0.0

        # --- News sentiment ---
        news_component = 0.0
        news_sentiment_raw = 0.0
        if self.news_monitor:
            scored = self.news_monitor.get_scored_buffer(limit=50)
            if scored:
                bullish = sum(1 for s in scored if s.sentiment == 'bullish')
                bearish = sum(1 for s in scored if s.sentiment == 'bearish')
                total = len(scored)
                news_sentiment_raw = (bullish - bearish) / total if total > 0 else 0
                news_component = max(-1.0, min(1.0, news_sentiment_raw * 2.5))

        # --- Breadth sentiment (advance/decline from movers universe) ---
        advancers = 0
        decliners = 0
        for sym in self._movers_symbols:
            snap = snapshots.get(sym)
            if not snap or not snap.daily_bar:
                continue
            _, pct = self._get_snapshot_pct(snap)
            if pct > 0:
                advancers += 1
            elif pct < 0:
                decliners += 1

        breadth_total = advancers + decliners
        breadth_ratio = (advancers - decliners) / breadth_total if breadth_total > 0 else 0
        breadth_component = max(-1.0, min(1.0, breadth_ratio))

        # --- Macro component (dollar + rates) ---
        _, uup_pct = self._get_snapshot_pct(snapshots.get(self._dollar_proxy))
        _, tlt_pct = self._get_snapshot_pct(snapshots.get(self._rates_proxy))
        # Strong dollar = risk-off, falling bonds (rising yields) = risk-off
        macro_component = max(-1.0, min(1.0, (-uup_pct + tlt_pct) / 2))

        # --- Composite risk score ---
        risk_score = 100 * (
            0.35 * vol_component +
            0.20 * macro_component +
            0.25 * breadth_component +
            0.20 * news_component
        )
        risk_score = round(risk_score, 1)

        if risk_score >= 20:
            risk_mode = 'risk_on'
        elif risk_score <= -20:
            risk_mode = 'risk_off'
        else:
            risk_mode = 'mixed'

        # --- Sentiment score (-10 to +10) ---
        breadth_sentiment = breadth_ratio  # already -1..+1
        sentiment_score = round((0.7 * news_sentiment_raw + 0.3 * breadth_sentiment) * 10, 1)
        sentiment_score = max(-10.0, min(10.0, sentiment_score))

        # --- Change summary text ---
        summary_parts = []
        for sym in self._index_symbols:
            snap = snapshots.get(sym)
            if snap and snap.daily_bar:
                _, pct = self._get_snapshot_pct(snap)
                direction = '+' if pct >= 0 else ''
                summary_parts.append(f"{sym} {direction}{pct:.1f}%")
        if vol_level > 0:
            summary_parts.append(f"VIX {vol_level:.1f}")
        change_summary = ', '.join(summary_parts) if summary_parts else 'No data'

        result = {
            'risk_mode': risk_mode,
            'risk_score': risk_score,
            'sentiment_score': sentiment_score,
            'volatility_state': volatility_state,
            'volatility_source': vol_source,
            'volatility_level': round(vol_level, 2),
            'volatility_change_1d_pct': round(vol_change_1d, 2),
            'components': {
                'vol_component': round(vol_component, 3),
                'macro_component': round(macro_component, 3),
                'breadth_component': round(breadth_component, 3),
                'news_component': round(news_component, 3),
            },
            'proxies': {
                'dollar_symbol': self._dollar_proxy,
                'rates_symbol': self._rates_proxy,
                'breadth_universe_size': breadth_total,
            },
            'change_summary': change_summary,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._pulse_cache = result
        self._pulse_cache_at = now
        return result

    # ------------------------------------------------------------------
    # Macro Tape
    # ------------------------------------------------------------------

    def get_macro_tape(self) -> Dict[str, Any]:
        now = time.time()
        if self._tape_cache and (now - self._tape_cache_at) < self._tape_ttl:
            return self._tape_cache

        # Collect all symbols we need
        all_syms = set()
        for entry in self._macro_tape_config:
            if 'symbols' in entry:
                all_syms.update(entry['symbols'])
            elif 'symbol' in entry:
                all_syms.add(entry['symbol'])
        snapshots = self._get_snapshots(list(all_syms))

        items = []
        for entry in self._macro_tape_config:
            label = entry.get('label', '')
            is_fallback = False

            if 'symbols' in entry:
                # Fallback chain (e.g., VIX -> VIXY -> UVXY)
                resolved_snap = None
                resolved_sym = entry['symbols'][0]
                for i, sym in enumerate(entry['symbols']):
                    snap = snapshots.get(sym)
                    if snap and snap.daily_bar:
                        resolved_snap = snap
                        resolved_sym = sym
                        is_fallback = i > 0
                        break
                if resolved_snap:
                    price, pct = self._get_snapshot_pct(resolved_snap)
                    items.append({
                        'label': label,
                        'symbol': resolved_sym,
                        'price': round(price, 2),
                        'pct_1d': round(pct, 2),
                        'is_fallback': is_fallback,
                    })
                else:
                    items.append({
                        'label': label,
                        'symbol': resolved_sym,
                        'price': 0,
                        'pct_1d': 0,
                        'is_fallback': True,
                    })
            else:
                sym = entry.get('symbol', '')
                snap = snapshots.get(sym)
                price, pct = self._get_snapshot_pct(snap) if snap else (0.0, 0.0)
                items.append({
                    'label': label,
                    'symbol': sym,
                    'price': round(price, 2),
                    'pct_1d': round(pct, 2),
                    'is_fallback': False,
                })

        result = {
            'items': items,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._tape_cache = result
        self._tape_cache_at = now
        return result

    # ------------------------------------------------------------------
    # Movers / Losers
    # ------------------------------------------------------------------

    def get_movers(self, limit: int = 10) -> Dict[str, Any]:
        now = time.time()
        if self._movers_cache and (now - self._movers_cache_at) < self._movers_ttl:
            return self._movers_cache

        movers_data = []
        snapshots = self._get_snapshots(self._movers_symbols)

        for sym in self._movers_symbols:
            snap = snapshots.get(sym)
            if not snap or not snap.daily_bar:
                continue
            price, pct = self._get_snapshot_pct(snap)
            volume = int(snap.daily_bar.volume) if snap.daily_bar.volume else 0

            prev_vol = int(snap.previous_daily_bar.volume) if snap.previous_daily_bar and snap.previous_daily_bar.volume else 0
            rel_vol = round(volume / prev_vol, 1) if prev_vol > 0 else 0

            sectors = self._sector_map.get(sym, [])
            movers_data.append({
                'symbol': sym,
                'price': round(price, 2),
                'pct_change': round(pct, 2),
                'volume': volume,
                'rel_volume': rel_vol,
                'sector': sectors[0] if sectors else '',
            })

        movers_data.sort(key=lambda x: x['pct_change'], reverse=True)
        gainers = movers_data[:limit]
        losers = list(reversed(movers_data[-limit:]))

        result = {
            'gainers': gainers,
            'losers': losers,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._movers_cache = result
        self._movers_cache_at = now
        return result

    # ------------------------------------------------------------------
    # Hot Themes
    # ------------------------------------------------------------------

    def get_themes(self, limit: int = 6) -> Dict[str, Any]:
        now = time.time()
        if self._themes_cache and (now - self._themes_cache_at) < self._themes_ttl:
            return self._themes_cache

        themes = []
        if self.news_monitor and hasattr(self.news_monitor, 'sector_trend_service'):
            scored = self.news_monitor.get_scored_buffer(limit=200)
            sts = self.news_monitor.sector_trend_service
            trends_1h = sts.get_trending(scored, window='1h', limit=limit)
            trends_6h = sts.get_trending(scored, window='6h', limit=limit)

            score_6h = {t.sector: t.rank_score for t in trends_6h}

            for t in trends_1h[:limit]:
                prev_score = score_6h.get(t.sector, 0)
                if t.rank_score > prev_score * 1.2:
                    momentum = 'rising'
                elif t.rank_score < prev_score * 0.8:
                    momentum = 'falling'
                else:
                    momentum = 'stable'

                themes.append({
                    'name': t.sector,
                    'momentum': momentum,
                    'sentiment': t.dominant_sentiment,
                    'impact_score': round(t.rank_score, 1),
                    'top_headline': t.top_headline,
                    'item_count': t.item_count,
                })

        result = {
            'themes': themes,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._themes_cache = result
        self._themes_cache_at = now
        return result

    # ------------------------------------------------------------------
    # Catalyst Clock (V1 placeholder)
    # ------------------------------------------------------------------

    def get_catalyst_clock(self, hours: int = 72) -> Dict[str, Any]:
        if self.catalyst_service:
            return self.catalyst_service.get_events(hours=hours)
        return {
            'window_hours': hours,
            'events': [],
            'source': 'unavailable',
            'provider_status': 'unavailable',
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
