"""
Trade Context Service.

Computes intraday regime, VWAP state, S/R zones, and event-risk context
from Alpaca intraday bars + Catalyst Clock feed.
Applies only to minute timeframes (1m, 5m, 15m).
"""

import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VALID_TIMEFRAMES = {'1m', '5m', '15m'}
TIMEFRAME_MINUTES = {'1m': 1, '5m': 5, '15m': 15}


class TradeContextService:
    def __init__(self, config: Dict[str, Any], catalyst_service=None):
        self.config = config
        self.catalyst_service = catalyst_service

        self._stock_client = None
        self._init_alpaca()

        tc_config = config.get('landing', {}).get('trade_context', {})
        self._cache_ttl = tc_config.get('cache_ttl', 30)
        self._sr_lookback_bars = tc_config.get('sr_lookback_bars', 120)
        self._sr_zone_merge_pct = tc_config.get('sr_zone_merge_pct', 0.08)

        # Cache keyed by (symbol, timeframe)
        self._cache: Dict[Tuple[str, str], Dict] = {}
        self._cache_at: Dict[Tuple[str, str], float] = {}

    def _init_alpaca(self):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            api_key = os.environ.get('ALPACA_KEY_ID', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
            if api_key and secret_key:
                self._stock_client = StockHistoricalDataClient(
                    api_key=api_key, secret_key=secret_key
                )
                logger.info("TradeContextService: Alpaca client initialized")
            else:
                logger.warning("TradeContextService: No Alpaca credentials")
        except Exception as e:
            logger.warning(f"TradeContextService: Alpaca init failed: {e}")

    def get_trade_context(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        now = time.time()

        if timeframe not in VALID_TIMEFRAMES:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'state': 'not_applicable',
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }

        cache_key = (symbol, timeframe)
        cached = self._cache.get(cache_key)
        if cached and (now - self._cache_at.get(cache_key, 0)) < self._cache_ttl:
            return cached

        if not self._stock_client:
            return self._unavailable(symbol, timeframe, 'No market data client')

        try:
            bars = self._fetch_intraday_bars(symbol, timeframe)
        except Exception as e:
            logger.warning(f"TradeContext bar fetch failed for {symbol}/{timeframe}: {e}")
            return self._unavailable(symbol, timeframe, str(e))

        if bars is None or len(bars) < 20:
            return self._unavailable(symbol, timeframe, 'Insufficient bar data')

        closes = np.array([b['close'] for b in bars])
        highs = np.array([b['high'] for b in bars])
        lows = np.array([b['low'] for b in bars])
        volumes = np.array([b['volume'] for b in bars])
        typicals = (highs + lows + closes) / 3.0

        # VWAP
        vwap_val, vwap_std = self._compute_vwap(typicals, volumes)
        last_price = float(closes[-1])
        vwap_state = self._compute_vwap_state(last_price, vwap_val, vwap_std)

        # Regime
        regime, regime_confidence = self._detect_regime(closes, highs, lows, volumes)

        # S/R zones
        sr_zones = self._compute_sr_zones(highs, lows, closes, last_price)

        # Event risk
        event_risk = self._get_event_risk()

        # Summary
        summary = self._build_summary(regime, vwap_state, sr_zones, event_risk)

        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'state': 'ok',
            'intraday_regime': regime,
            'regime_confidence': round(regime_confidence, 2),
            'vwap_state': vwap_state,
            'sr_zones': sr_zones,
            'event_risk': event_risk,
            'summary': summary,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._cache[cache_key] = result
        self._cache_at[cache_key] = now
        return result

    def _unavailable(self, symbol: str, timeframe: str, reason: str) -> Dict:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'state': 'unavailable',
            'reason': reason,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }

    def _fetch_intraday_bars(self, symbol: str, timeframe: str) -> Optional[List[Dict]]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_minutes = TIMEFRAME_MINUTES[timeframe]
        tf = TimeFrame(tf_minutes, TimeFrameUnit.Minute)

        now_utc = datetime.now(timezone.utc)
        # Fetch today's session (go back ~18h to cover pre-market)
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=6)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=now_utc,
        )
        result = self._stock_client.get_stock_bars(request)
        df = result.df

        if df.empty:
            return None

        df = df.reset_index()
        bars = []
        for _, row in df.iterrows():
            bars.append({
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
            })
        return bars

    def _compute_vwap(self, typicals: np.ndarray, volumes: np.ndarray) -> Tuple[float, float]:
        cum_vol = np.cumsum(volumes)
        cum_tp_vol = np.cumsum(typicals * volumes)

        if cum_vol[-1] == 0:
            return (float(typicals[-1]), 0.0)

        vwap = cum_tp_vol[-1] / cum_vol[-1]

        # VWAP standard deviation
        squared_diff = (typicals - vwap) ** 2
        variance = np.sum(squared_diff * volumes) / cum_vol[-1]
        std = math.sqrt(variance) if variance > 0 else 0.0

        return (float(vwap), float(std))

    def _compute_vwap_state(self, price: float, vwap: float, std: float) -> Dict[str, Any]:
        distance = price - vwap
        distance_pct = (distance / vwap * 100) if vwap != 0 else 0.0

        # Relation
        if std > 0 and abs(distance) < std * 0.15:
            relation = 'crossing'
        elif distance > 0:
            relation = 'above'
        else:
            relation = 'below'

        # Sigma position
        if std > 0:
            sigma = abs(distance) / std
            if sigma < 1:
                sigma_position = 'inside_1sigma'
            elif sigma < 2:
                sigma_position = 'between_1_2sigma'
            else:
                sigma_position = 'outside_2sigma'
        else:
            sigma_position = 'inside_1sigma'

        return {
            'relation': relation,
            'sigma_position': sigma_position,
            'distance_to_vwap': round(distance, 4),
            'distance_to_vwap_pct': round(distance_pct, 2),
        }

    def _detect_regime(
        self, closes: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, volumes: np.ndarray
    ) -> Tuple[str, float]:
        """
        Simple regime detection using:
        - Efficiency ratio (directional movement / total movement)
        - ATR-normalized range expansion
        """
        n = len(closes)
        lookback = min(n, 30)
        recent = closes[-lookback:]

        # Efficiency ratio: net movement / sum of absolute bar-to-bar changes
        net_move = abs(recent[-1] - recent[0])
        total_move = np.sum(np.abs(np.diff(recent)))

        if total_move == 0:
            return ('range', 0.5)

        efficiency = net_move / total_move  # 0 = pure range, 1 = pure trend

        # Volatility expansion: recent ATR vs session ATR
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        atr_recent = np.mean(recent_highs - recent_lows)
        atr_session = np.mean(highs - lows)
        vol_ratio = atr_recent / atr_session if atr_session > 0 else 1.0

        # Classification
        if vol_ratio > 1.5 and efficiency < 0.35:
            regime = 'volatile'
            confidence = min(0.95, 0.5 + (vol_ratio - 1.5) * 0.3)
        elif efficiency > 0.45:
            regime = 'trend'
            confidence = min(0.95, 0.4 + efficiency * 0.6)
        elif efficiency < 0.2:
            regime = 'range'
            confidence = min(0.95, 0.5 + (0.2 - efficiency) * 2.0)
        else:
            regime = 'transition'
            confidence = 0.4 + abs(efficiency - 0.3) * 1.5
            confidence = min(0.7, confidence)

        return (regime, float(confidence))

    def _compute_sr_zones(
        self, highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, last_price: float
    ) -> Dict[str, List[Dict]]:
        """
        Find support/resistance zones from pivot highs/lows.
        Merge nearby pivots into zones with strength based on touch count.
        """
        n = len(highs)
        lookback = min(n, self._sr_lookback_bars)
        h = highs[-lookback:]
        l = lows[-lookback:]
        c = closes[-lookback:]

        pivots = []
        window = 5

        for i in range(window, len(h) - window):
            # Pivot high
            if h[i] == np.max(h[i - window:i + window + 1]):
                pivots.append(('resistance', float(h[i])))
            # Pivot low
            if l[i] == np.min(l[i - window:i + window + 1]):
                pivots.append(('support', float(l[i])))

        if not pivots:
            return {'nearest_support': [], 'nearest_resistance': []}

        # Merge nearby pivots into zones
        merge_threshold = last_price * (self._sr_zone_merge_pct / 100)
        zones = self._merge_pivots(pivots, merge_threshold, last_price)

        # Split and sort by distance to price
        supports = sorted(
            [z for z in zones if z['mid'] < last_price],
            key=lambda z: last_price - z['mid']
        )[:2]
        resistances = sorted(
            [z for z in zones if z['mid'] >= last_price],
            key=lambda z: z['mid'] - last_price
        )[:2]

        return {
            'nearest_support': supports,
            'nearest_resistance': resistances,
        }

    def _merge_pivots(
        self, pivots: List[Tuple[str, float]],
        threshold: float, last_price: float
    ) -> List[Dict]:
        # Sort by price level
        sorted_pivots = sorted(pivots, key=lambda p: p[1])

        zones: List[Dict] = []
        current_group: List[Tuple[str, float]] = [sorted_pivots[0]]

        for i in range(1, len(sorted_pivots)):
            if sorted_pivots[i][1] - current_group[-1][1] <= threshold:
                current_group.append(sorted_pivots[i])
            else:
                zones.append(self._group_to_zone(current_group, last_price))
                current_group = [sorted_pivots[i]]

        zones.append(self._group_to_zone(current_group, last_price))
        return zones

    def _group_to_zone(self, group: List[Tuple[str, float]], last_price: float) -> Dict:
        prices = [p[1] for p in group]
        kinds = [p[0] for p in group]
        low = min(prices)
        high = max(prices)
        mid = (low + high) / 2
        touch_count = len(group)

        # Determine kind by majority
        support_count = sum(1 for k in kinds if k == 'support')
        kind = 'support' if support_count > len(kinds) / 2 else 'resistance'

        # Strength
        if touch_count >= 4:
            strength = 'strong'
        elif touch_count >= 2:
            strength = 'medium'
        else:
            strength = 'weak'

        # Status
        dist = abs(mid - last_price)
        dist_pct = dist / last_price * 100
        if dist_pct < 0.1:
            status = 'tested'
        else:
            status = 'active'

        return {
            'kind': kind,
            'low': round(low, 2),
            'high': round(high, 2),
            'mid': round(mid, 2),
            'strength': strength,
            'status': status,
            'touch_count': touch_count,
        }

    def _get_event_risk(self) -> Dict[str, Any]:
        if not self.catalyst_service:
            return {'status': 'none'}

        try:
            data = self.catalyst_service.get_events(hours=72)
        except Exception:
            return {'status': 'none'}

        events = data.get('events', [])
        now_utc = datetime.now(timezone.utc)

        # Find next upcoming high-impact event
        for evt in events:
            if evt.get('status') == 'past':
                continue
            if evt.get('impact') != 'high':
                continue

            try:
                evt_time = datetime.fromisoformat(evt['time'])
            except Exception:
                continue

            countdown = max(0, int((evt_time - now_utc).total_seconds()))
            urgency = 'imminent' if countdown <= 3600 else 'upcoming'

            return {
                'status': urgency,
                'next_event': evt.get('title', ''),
                'impact': 'high',
                'time': evt['time'],
                'countdown_seconds': countdown,
            }

        return {'status': 'none'}

    def _build_summary(
        self, regime: str, vwap_state: Dict,
        sr_zones: Dict, event_risk: Dict
    ) -> str:
        parts = []

        # Regime
        parts.append(regime.capitalize())

        # VWAP
        rel = vwap_state['relation']
        sigma = vwap_state['sigma_position']
        dist_pct = vwap_state['distance_to_vwap_pct']
        sigma_label = sigma.replace('_', ' ').replace('inside 1sigma', '1σ').replace(
            'between 1 2sigma', '1-2σ').replace('outside 2sigma', '>2σ')
        parts.append(f"{rel} VWAP ({dist_pct:+.1f}%, {sigma_label})")

        # Nearest S/R
        supports = sr_zones.get('nearest_support', [])
        resistances = sr_zones.get('nearest_resistance', [])
        if supports:
            s = supports[0]
            parts.append(f"S {s['low']:.1f}-{s['high']:.1f} {s['strength']}")
        if resistances:
            r = resistances[0]
            parts.append(f"R {r['low']:.1f}-{r['high']:.1f} {r['strength']}")

        # Event risk
        if event_risk.get('status') in ('imminent', 'upcoming'):
            countdown_s = event_risk.get('countdown_seconds', 0)
            if countdown_s < 3600:
                time_str = f"{countdown_s // 60}m"
            elif countdown_s < 86400:
                time_str = f"{countdown_s // 3600}h"
            else:
                time_str = f"{countdown_s // 86400}d"
            parts.append(f"{event_risk['next_event']} in {time_str}")

        return '; '.join(parts)
