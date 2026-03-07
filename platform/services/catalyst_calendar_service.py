"""
Catalyst Calendar Service.

Fetches economic calendar events from Benzinga and normalizes them
into CatalystEvent format for the Catalyst Clock component.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# Benzinga importance: 1=low, 2=medium, 3=high
IMPORTANCE_MAP = {3: 'high', 2: 'medium', 1: 'low'}
MIN_IMPORTANCE_MAP = {'high': 3, 'medium': 2, 'low': 1}


class CatalystCalendarService:
    def __init__(self, config: Dict[str, Any], news_monitor=None):
        self.config = config
        self.news_monitor = news_monitor
        catalyst_config = config.get('landing', {}).get('catalyst', {})

        self._api_key = os.environ.get('BENZINGA_API_KEY', '')
        self._base_url = 'https://api.benzinga.com/api/v2.1/calendar'
        self._cache_ttl = catalyst_config.get('cache_ttl', 300)
        self._country = catalyst_config.get('country', 'USA')
        self._min_importance = MIN_IMPORTANCE_MAP.get(
            catalyst_config.get('min_impact', 'medium'), 2
        )

        # Cache
        self._cache: Optional[Dict] = None
        self._cache_at: float = 0

        if self._api_key:
            logger.info("CatalystCalendarService: Benzinga API key configured")
        else:
            logger.warning("CatalystCalendarService: No BENZINGA_API_KEY, using inferred-news fallback only")

    def get_events(self, hours: int = 72) -> Dict[str, Any]:
        now = time.time()
        if self._cache and (now - self._cache_at) < self._cache_ttl:
            cached = self._cache
            # Recompute countdowns
            for evt in cached.get('events', []):
                evt_time = datetime.fromisoformat(evt['time'])
                evt['countdown_seconds'] = max(0, int((evt_time - datetime.now(timezone.utc)).total_seconds()))
            return cached

        events: List[Dict] = []
        source = 'inferred_news'
        provider_status = 'unavailable'

        # Try Benzinga
        if self._api_key:
            try:
                bz_events = self._fetch_benzinga_economics(hours)
                events.extend(bz_events)
                source = 'benzinga'
                provider_status = 'ok'
            except Exception as e:
                logger.warning(f"Benzinga economics fetch failed: {e}")
                provider_status = 'degraded'

        # Inferred-news fallback
        inferred = self._get_inferred_catalysts(hours)
        if inferred:
            if events:
                # Merge: deduplicate by checking if inferred headline matches existing event
                existing_titles = {e['title'].lower() for e in events}
                for inf in inferred:
                    if not any(kw in inf['title'].lower() for kw in
                               [t.lower() for t in existing_titles]):
                        events.append(inf)
                if source == 'benzinga':
                    source = 'mixed'
            else:
                events.extend(inferred)
                source = 'inferred_news'

        # Sort by time
        events.sort(key=lambda e: e['time'])

        result = {
            'window_hours': hours,
            'events': events,
            'source': source,
            'provider_status': provider_status,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._cache = result
        self._cache_at = now
        return result

    def _fetch_benzinga_economics(self, hours: int) -> List[Dict]:
        now_utc = datetime.now(timezone.utc)
        # Fetch past 72h + future window
        past_hours = 72
        date_from = (now_utc - timedelta(hours=past_hours)).strftime('%Y-%m-%d')
        date_to = (now_utc + timedelta(hours=hours)).strftime('%Y-%m-%d')

        params = {
            'token': self._api_key,
            'parameters[date_from]': date_from,
            'parameters[date_to]': date_to,
            'parameters[importance]': self._min_importance,
            'country': self._country,
            'pagesize': 100,
        }

        url = f"{self._base_url}/economics?{urlencode(params)}"
        resp = requests.get(url, headers={'Accept': 'application/json'}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        events = []
        now_utc = datetime.now(timezone.utc)
        cutoff_past = now_utc - timedelta(hours=past_hours)
        for item in data.get('economics', []):
            evt_time = self._parse_benzinga_time(item.get('date', ''), item.get('time', ''))
            if not evt_time:
                continue

            importance = item.get('importance', 1)
            is_past = evt_time < now_utc

            # Skip events outside our window
            if evt_time < cutoff_past:
                continue

            # For past events, only include high-importance (3)
            if is_past and importance < 3:
                continue

            countdown = max(0, int((evt_time - now_utc).total_seconds()))

            events.append({
                'id': f"bz_{item.get('id', '')}",
                'type': 'economic',
                'title': item.get('event_name', 'Unknown Event'),
                'time': evt_time.isoformat(),
                'status': 'past' if is_past else 'upcoming',
                'impact': IMPORTANCE_MAP.get(importance, 'low'),
                'detail': self._build_detail(item),
                'source': 'benzinga',
                'category': item.get('event_category', ''),
                'countdown_seconds': countdown,
                'consensus': item.get('consensus', ''),
                'prior': item.get('prior', ''),
                'actual': item.get('actual', ''),
            })

        return events

    def _parse_benzinga_time(self, date_str: str, time_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            if time_str and time_str != '00:00:00':
                # Benzinga times are US/Eastern
                from zoneinfo import ZoneInfo
                naive = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
                eastern = ZoneInfo('America/New_York')
                local_dt = naive.replace(tzinfo=eastern)
                return local_dt.astimezone(timezone.utc)
            else:
                # No time given — assume market open (9:30 ET)
                from zoneinfo import ZoneInfo
                naive = datetime.strptime(f"{date_str} 09:30:00", '%Y-%m-%d %H:%M:%S')
                eastern = ZoneInfo('America/New_York')
                local_dt = naive.replace(tzinfo=eastern)
                return local_dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _build_detail(self, item: Dict) -> str:
        parts = []
        if item.get('event_period'):
            parts.append(f"Period: {item['event_period']}")
        if item.get('prior') and item.get('prior_t'):
            parts.append(f"Prior: {item['prior']}{item['prior_t']}")
        elif item.get('prior'):
            parts.append(f"Prior: {item['prior']}")
        if item.get('consensus') and item.get('consensus_t'):
            parts.append(f"Consensus: {item['consensus']}{item['consensus_t']}")
        elif item.get('consensus'):
            parts.append(f"Consensus: {item['consensus']}")
        if item.get('actual') and item.get('actual_t'):
            parts.append(f"Actual: {item['actual']}{item['actual_t']}")
        elif item.get('actual'):
            parts.append(f"Actual: {item['actual']}")
        return ' | '.join(parts) if parts else ''

    def _get_inferred_catalysts(self, hours: int) -> List[Dict]:
        if not self.news_monitor:
            return []

        try:
            scored = self.news_monitor.get_scored_buffer(limit=100)
        except Exception:
            return []

        if not scored:
            return []

        CATALYST_KEYWORDS = {
            'CPI': ('economic', 'high'),
            'PPI': ('economic', 'high'),
            'NFP': ('economic', 'high'),
            'nonfarm': ('economic', 'high'),
            'non-farm': ('economic', 'high'),
            'FOMC': ('economic', 'high'),
            'Fed rate': ('economic', 'high'),
            'Fed meeting': ('economic', 'high'),
            'Fed speaker': ('economic', 'medium'),
            'PCE': ('economic', 'high'),
            'GDP': ('economic', 'medium'),
            'jobless claims': ('economic', 'medium'),
            'earnings': ('earnings', 'medium'),
            'earnings report': ('earnings', 'high'),
        }

        events = []
        seen_titles = set()
        now_utc = datetime.now(timezone.utc)

        for item in scored:
            headline = getattr(item, 'headline', '') or ''
            headline_lower = headline.lower()

            for keyword, (evt_type, impact) in CATALYST_KEYWORDS.items():
                if keyword.lower() in headline_lower:
                    # Avoid duplicates
                    title_key = f"{keyword}_{headline[:30]}"
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)

                    # Use the news timestamp
                    ts = getattr(item, 'created_at', None) or getattr(item, 'timestamp', None)
                    if ts and isinstance(ts, datetime):
                        evt_time = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    else:
                        evt_time = now_utc

                    events.append({
                        'id': f"inf_{hash(headline) & 0xFFFFFFFF:08x}",
                        'type': evt_type,
                        'title': headline[:120],
                        'time': evt_time.isoformat(),
                        'status': 'past' if evt_time < now_utc else 'upcoming',
                        'impact': impact,
                        'detail': f"Source: {getattr(item, 'source', 'news')}",
                        'source': 'inferred_news',
                        'category': '',
                        'countdown_seconds': max(0, int((evt_time - now_utc).total_seconds())),
                        'consensus': '',
                        'prior': '',
                        'actual': '',
                    })
                    break  # Only match first keyword per headline

        return events[:10]  # Limit inferred events
