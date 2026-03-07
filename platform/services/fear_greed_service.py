"""
Fear & Greed Index + CBOE Put/Call Ratio Service.

Fetches CNN Fear & Greed Index and CBOE equity put/call ratio
for the Market Pulse strip.
"""

import io
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# CNN Fear & Greed data endpoint (no auth required)
CNN_FNG_URL = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

# CBOE equity put/call ratio CSV
CBOE_PCR_URL = 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv'

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
        self._pcr_cache_ttl = fg_config.get('pcr_cache_ttl', 3600)  # 1 hour

        self._fng_cache: Optional[Dict] = None
        self._fng_cache_at: float = 0
        self._pcr_cache: Optional[Dict] = None
        self._pcr_cache_at: float = 0

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

            # CNN returns nested structure with fear_and_greed.score and .rating
            fng = data.get('fear_and_greed', {})
            score = fng.get('score', None)
            rating = fng.get('rating', '')

            if score is not None:
                score = round(float(score), 1)
                label = rating if rating else _fng_label(score)
                # Also get previous close for change
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

        try:
            resp = requests.get(
                CBOE_PCR_URL,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                },
                timeout=15,
            )
            resp.raise_for_status()

            # Parse CSV — columns: TRADE DATE, CALLS, PUTS, TOTAL, P/C RATIO
            import pandas as pd
            df = pd.read_csv(io.StringIO(resp.text))

            # Normalize column names
            df.columns = [c.strip().upper() for c in df.columns]

            # Find the ratio column
            ratio_col = None
            for c in df.columns:
                if 'RATIO' in c or 'P/C' in c:
                    ratio_col = c
                    break

            if ratio_col is None:
                result = self._pcr_unavailable('No ratio column found in CSV')
            else:
                # Get latest row
                df = df.dropna(subset=[ratio_col])
                if df.empty:
                    result = self._pcr_unavailable('Empty CSV data')
                else:
                    latest = df.iloc[-1]
                    ratio = round(float(latest[ratio_col]), 3)

                    # Get previous day for change
                    prev_ratio = round(float(df.iloc[-2][ratio_col]), 3) if len(df) >= 2 else None
                    change = round(ratio - prev_ratio, 3) if prev_ratio is not None else None

                    # Find date column
                    date_col = None
                    for c in df.columns:
                        if 'DATE' in c:
                            date_col = c
                            break
                    trade_date = str(latest[date_col]).strip() if date_col else None

                    result = {
                        'ratio': ratio,
                        'previous': prev_ratio,
                        'change': change,
                        'trade_date': trade_date,
                        'source': 'cboe',
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                    }

        except Exception as e:
            logger.warning(f"CBOE Put/Call ratio fetch failed: {e}")
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
            'source': 'cboe',
            'error': reason,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
