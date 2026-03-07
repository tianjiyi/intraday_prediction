"""
Fear & Greed Index + Put/Call Ratio Service.

Fetches CNN Fear & Greed Index and SPY put/call ratio (via Alpaca options)
for the Market Pulse strip.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# CNN Fear & Greed data endpoint (no auth required)
CNN_FNG_URL = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

FNG_LABELS = {
    (0, 25): 'Extreme Fear',
    (25, 45): 'Fear',
    (45, 55): 'Neutral',
    (55, 75): 'Greed',
    (75, 101): 'Extreme Greed',
}


def _fng_label(score: float) -> str:
    for (lo, hi), label in FNG_LABELS.items():
        if lo <= score < hi:
            return label
    return 'Neutral'


class FearGreedService:
    def __init__(self, config: Dict[str, Any]):
        fg_config = config.get('landing', {}).get('fear_greed', {})
        self._fng_cache_ttl = fg_config.get('fng_cache_ttl', 300)  # 5 min
        self._pcr_cache_ttl = fg_config.get('pcr_cache_ttl', 300)  # 5 min

        self._fng_cache: Optional[Dict] = None
        self._fng_cache_at: float = 0
        self._pcr_cache: Optional[Dict] = None
        self._pcr_cache_at: float = 0

        # Alpaca options client
        self._opt_client = None
        self._init_alpaca_options()

    def _init_alpaca_options(self):
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            api_key = os.environ.get('ALPACA_KEY_ID', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
            if api_key and secret_key:
                self._opt_client = OptionHistoricalDataClient(
                    api_key=api_key, secret_key=secret_key
                )
                logger.info("FearGreedService: Alpaca options client initialized")
            else:
                logger.warning("FearGreedService: No Alpaca credentials for options")
        except Exception as e:
            logger.warning(f"FearGreedService: Alpaca options init failed: {e}")

    def get_fear_greed(self) -> Dict[str, Any]:
        now = time.time()
        if self._fng_cache and (now - self._fng_cache_at) < self._fng_cache_ttl:
            return self._fng_cache

        try:
            resp = requests.get(
                CNN_FNG_URL,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            fng = data.get('fear_and_greed', {})
            score = fng.get('score', None)
            rating = fng.get('rating', '')

            if score is not None:
                score = round(float(score), 1)
                label = rating if rating else _fng_label(score)
                prev = fng.get('previous_close', None)
                change = round(score - float(prev), 1) if prev is not None else None

                result = {
                    'score': score,
                    'label': label,
                    'previous_close': round(float(prev), 1) if prev is not None else None,
                    'change': change,
                    'source': 'cnn',
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                }
            else:
                result = self._fng_unavailable('No score in response')

        except Exception as e:
            logger.warning(f"CNN Fear & Greed fetch failed: {e}")
            result = self._fng_unavailable(str(e))

        self._fng_cache = result
        self._fng_cache_at = now
        return result

    def get_put_call_ratio(self) -> Dict[str, Any]:
        now = time.time()
        if self._pcr_cache and (now - self._pcr_cache_at) < self._pcr_cache_ttl:
            return self._pcr_cache

        if not self._opt_client:
            result = self._pcr_unavailable('No options client available')
            self._pcr_cache = result
            self._pcr_cache_at = now
            return result

        try:
            from alpaca.data.requests import OptionChainRequest
            req = OptionChainRequest(underlying_symbol='SPY')
            chain = self._opt_client.get_option_chain(req)

            put_vol = 0.0
            call_vol = 0.0

            for symbol, snap in chain.items():
                # Get volume from latest trade
                trade = getattr(snap, 'latest_trade', None)
                vol = float(trade.size) if trade and hasattr(trade, 'size') and trade.size else 0.0

                # Determine put vs call from symbol suffix
                # Alpaca option symbols: SPY250307P00600000
                if 'P' in symbol[-10:]:
                    put_vol += vol
                elif 'C' in symbol[-10:]:
                    call_vol += vol

            if call_vol > 0:
                ratio = round(put_vol / call_vol, 3)
                result = {
                    'ratio': ratio,
                    'put_volume': int(put_vol),
                    'call_volume': int(call_vol),
                    'underlying': 'SPY',
                    'source': 'alpaca',
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                }
            else:
                result = self._pcr_unavailable('No call volume data')

        except Exception as e:
            logger.warning(f"Put/Call ratio fetch failed: {e}")
            result = self._pcr_unavailable(str(e))

        self._pcr_cache = result
        self._pcr_cache_at = now
        return result

    def _fng_unavailable(self, reason: str) -> Dict:
        return {
            'score': None,
            'label': None,
            'source': 'cnn',
            'error': reason,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }

    def _pcr_unavailable(self, reason: str) -> Dict:
        return {
            'ratio': None,
            'source': 'alpaca',
            'error': reason,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
