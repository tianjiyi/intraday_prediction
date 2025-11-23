"""
News Service for Market Sentiment Analysis
Fetches news from Alpaca API for sentiment analysis
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pytz
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Alpaca news client
try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    ALPACA_NEWS_AVAILABLE = True
except ImportError:
    ALPACA_NEWS_AVAILABLE = False
    logger.warning("Alpaca news client not available. Update alpaca-py package.")


class NewsService:
    """Service for fetching and processing market news"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize News Service

        Args:
            config: Configuration dict containing Alpaca API keys
        """
        self.config = config
        self.api_key = config.get('ALPACA_KEY_ID') or os.environ.get('ALPACA_KEY_ID')
        self.secret_key = config.get('ALPACA_SECRET_KEY') or os.environ.get('ALPACA_SECRET_KEY')
        self.news_client = None

        self._init_client()

    def _init_client(self):
        """Initialize the Alpaca news client"""
        if not ALPACA_NEWS_AVAILABLE:
            logger.warning("Alpaca news client not available")
            return

        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not configured for news service")
            return

        try:
            self.news_client = NewsClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            logger.info("Alpaca News client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca News client: {e}")

    def is_available(self) -> bool:
        """Check if news service is available"""
        return self.news_client is not None

    async def get_news(
        self,
        symbol: str,
        limit: int = 10,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a symbol

        Args:
            symbol: Trading symbol (e.g., 'QQQ', 'AAPL')
            limit: Maximum number of news items to fetch
            hours_back: How many hours back to search for news

        Returns:
            List of news items with headline, summary, source, etc.
        """
        if not self.is_available():
            logger.warning("News service not available")
            return []

        try:
            # Calculate time range
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(hours=hours_back)

            # Clean symbol for news API (remove /USD for crypto)
            clean_symbol = symbol.replace('/USD', '').replace('/', '')

            logger.info(f"Fetching news for {clean_symbol}: {start_time.isoformat()} to {end_time.isoformat()} (limit={limit})")

            # Create news request
            request = NewsRequest(
                symbols=clean_symbol,
                start=start_time,
                end=end_time,
                limit=limit,
                sort='desc'  # Most recent first
            )

            # Fetch news
            news_response = self.news_client.get_news(request)

            # Convert to DataFrame - this is the recommended way per alpaca-py docs
            news_items = []
            try:
                df = news_response.df
                logger.info(f"News DataFrame shape: {df.shape}, columns: {list(df.columns)}")

                for idx, row in df.iterrows():
                    news_items.append({
                        'id': str(idx) if idx else '',
                        'headline': row.get('headline', 'No headline') or 'No headline',
                        'summary': row.get('summary', '') or '',
                        'source': row.get('source', 'Unknown') or 'Unknown',
                        'author': row.get('author', 'Unknown') or 'Unknown',
                        'created_at': row.get('created_at').isoformat() if pd.notna(row.get('created_at')) else '',
                        'updated_at': row.get('updated_at').isoformat() if pd.notna(row.get('updated_at')) else '',
                        'url': row.get('url', '') or '',
                        'symbols': row.get('symbols', []) or [],
                        'images': []
                    })
            except Exception as df_err:
                logger.warning(f"Could not convert news to DataFrame: {df_err}")

            logger.info(f"Fetched {len(news_items)} news items for {symbol}")
            # Log headlines for debugging
            for i, item in enumerate(news_items[:5]):
                logger.info(f"  News {i+1}: [{item.get('source')}] {item.get('headline')[:80]}...")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    async def get_market_news(
        self,
        limit: int = 20,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Fetch general market news (no specific symbol)

        Args:
            limit: Maximum number of news items
            hours_back: How many hours back to search

        Returns:
            List of general market news items
        """
        if not self.is_available():
            logger.warning("News service not available")
            return []

        try:
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(hours=hours_back)

            # Request without specific symbols for general market news
            request = NewsRequest(
                start=start_time,
                end=end_time,
                limit=limit,
                sort='desc'
            )

            news_response = self.news_client.get_news(request)

            # Convert to DataFrame
            news_items = []
            try:
                df = news_response.df
                logger.info(f"Market news DataFrame shape: {df.shape}")

                for idx, row in df.iterrows():
                    news_items.append({
                        'id': str(idx) if idx else '',
                        'headline': row.get('headline', 'No headline') or 'No headline',
                        'summary': row.get('summary', '') or '',
                        'source': row.get('source', 'Unknown') or 'Unknown',
                        'author': row.get('author', 'Unknown') or 'Unknown',
                        'created_at': row.get('created_at').isoformat() if pd.notna(row.get('created_at')) else '',
                        'url': row.get('url', '') or '',
                        'symbols': row.get('symbols', []) or []
                    })
            except Exception as df_err:
                logger.warning(f"Could not convert market news to DataFrame: {df_err}")

            logger.info(f"Fetched {len(news_items)} general market news items")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def format_news_for_display(self, news_items: List[Dict[str, Any]]) -> str:
        """
        Format news items for display in UI

        Args:
            news_items: List of news items

        Returns:
            Formatted HTML string for display
        """
        if not news_items:
            return "<p>No recent news available.</p>"

        html = ""
        for news in news_items:
            headline = news.get('headline', 'No headline')
            source = news.get('source', 'Unknown')
            created = news.get('created_at', '')[:16].replace('T', ' ')  # Truncate timestamp
            url = news.get('url', '')

            html += f"""
            <div class="news-item">
                <div class="news-source">{source} - {created}</div>
                <div class="news-headline">
                    {'<a href="' + url + '" target="_blank">' if url else ''}{headline}{'</a>' if url else ''}
                </div>
            </div>
            """

        return html

    def get_sentiment_keywords(self, news_items: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Simple keyword-based sentiment extraction

        Args:
            news_items: List of news items

        Returns:
            Dict with bullish/bearish keyword counts
        """
        bullish_keywords = [
            'surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'bullish', 'optimistic',
            'growth', 'profit', 'beat', 'exceed', 'record', 'high', 'upgrade', 'buy'
        ]
        bearish_keywords = [
            'fall', 'drop', 'decline', 'plunge', 'crash', 'bearish', 'pessimistic',
            'loss', 'miss', 'cut', 'downgrade', 'sell', 'low', 'weak', 'concern', 'fear'
        ]

        bullish_count = 0
        bearish_count = 0

        for news in news_items:
            text = (news.get('headline', '') + ' ' + news.get('summary', '')).lower()
            for keyword in bullish_keywords:
                if keyword in text:
                    bullish_count += 1
            for keyword in bearish_keywords:
                if keyword in text:
                    bearish_count += 1

        return {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'net_sentiment': bullish_count - bearish_count
        }
