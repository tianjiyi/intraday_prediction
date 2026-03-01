"""
Polygon.io Data Fetcher for Historical Tick/Quote Data

This module fetches historical trades and quotes from Polygon.io API
for OFI and VPIN calculations.

API Documentation: https://polygon.io/docs/stocks/get_v3_trades__stockticker
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import time
from functools import wraps

try:
    from polygon import RESTClient
    from polygon.rest.models import Trade, Quote
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    RESTClient = None

try:
    from .s3_fetcher import PolygonS3Fetcher
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    PolygonS3Fetcher = None

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if '429' in str(e) or 'too many' in str(e).lower():
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


class PolygonDataFetcher:
    """
    Fetches historical trades and quotes from Polygon.io API.

    Provides tick-level data with:
    - Trades: price, size, timestamp
    - Quotes: bid_price, bid_size, ask_price, ask_size, timestamp

    Supports caching to parquet files for fast reloads.
    """

    # Default cache directory (relative to this file)
    DEFAULT_CACHE_DIR = Path(__file__).parent / "data"

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        use_s3: bool = True,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
        s3_bucket: Optional[str] = None
    ):
        """
        Initialize Polygon.io client with optional S3 flat files support.

        Args:
            api_key: Polygon.io API key (for REST API fallback)
            cache_dir: Directory to cache data (default: market_regime/data/)
            use_cache: Whether to use file caching (default True)
            use_s3: Whether to use S3 flat files (default True, much faster)
            s3_access_key: Polygon S3 Access Key (or env POLYGON_S3_ACCESS_KEY)
            s3_secret_key: Polygon S3 Secret Key (or env POLYGON_S3_SECRET_KEY)
            s3_endpoint: S3 endpoint URL (default: https://files.polygon.io)
            s3_bucket: S3 bucket name (default: flatfiles)
        """
        if not POLYGON_AVAILABLE:
            raise ImportError(
                "polygon-api-client is not installed. "
                "Install with: pip install polygon-api-client"
            )

        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.use_cache = use_cache
        self.use_s3 = use_s3
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR

        # Initialize S3 fetcher if available and requested
        self.s3_fetcher = None
        if use_s3 and S3_AVAILABLE:
            try:
                self.s3_fetcher = PolygonS3Fetcher(
                    access_key=s3_access_key,
                    secret_key=s3_secret_key,
                    endpoint_url=s3_endpoint,
                    bucket=s3_bucket,
                    cache_dir=self.cache_dir / "s3_temp"
                )
                logger.info("S3 flat files fetcher initialized (10-20x faster)")
            except Exception as e:
                logger.warning(f"S3 fetcher init failed, will use API: {e}")
                self.s3_fetcher = None

        # Ensure cache directory exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Polygon.io client initialized (cache: {self.cache_dir if use_cache else 'disabled'}, S3: {self.s3_fetcher is not None})")

    def _get_cache_path(self, ticker: str, date: str, data_type: str) -> Path:
        """Get cache file path for a specific ticker/date/type."""
        ticker_dir = self.cache_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        return ticker_dir / f"{data_type}_{date}.parquet"

    def _load_from_cache(self, ticker: str, date: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(ticker, date, data_type)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded {len(df)} {data_type} from cache: {cache_path.name}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, df: pd.DataFrame, ticker: str, date: str, data_type: str) -> None:
        """Save data to cache."""
        if not self.use_cache or df.empty:
            return

        cache_path = self._get_cache_path(ticker, date, data_type)
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df)} {data_type} to: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def fetch_trades(
        self,
        ticker: str,
        date: str,
        limit: int = 50000,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch all trades for a ticker on a given date.

        Uses S3 flat files if available (10-20x faster), falls back to API.

        Args:
            ticker: Stock ticker symbol (e.g., 'QQQ')
            date: Date string in YYYY-MM-DD format
            limit: Max results per request (default 50000, only for API)
            force_refresh: Bypass cache and re-fetch

        Returns:
            DataFrame with columns: [timestamp, price, size, exchange, conditions]
        """
        # Check cache first
        if not force_refresh:
            cached = self._load_from_cache(ticker, date, 'trades')
            if cached is not None:
                return cached

        # Try S3 first (much faster)
        if self.s3_fetcher is not None:
            try:
                logger.info(f"Fetching trades via S3 for {ticker} on {date}")
                df = self.s3_fetcher.fetch_trades(ticker, date)

                if not df.empty:
                    # Save to cache
                    self._save_to_cache(df, ticker, date, 'trades')
                    return df
                else:
                    logger.warning(f"S3 returned no trades, trying API...")
            except Exception as e:
                logger.warning(f"S3 fetch failed, falling back to API: {e}")

        # Fall back to REST API
        return self._fetch_trades_api(ticker, date, limit)

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    def _fetch_trades_api(
        self,
        ticker: str,
        date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch trades using REST API (slower but reliable fallback).
        """
        logger.info(f"Fetching trades via API for {ticker} on {date}")

        trades_data = []

        try:
            # Use list_trades which handles pagination automatically
            for trade in self.client.list_trades(
                ticker=ticker,
                timestamp=date,
                limit=limit
            ):
                trades_data.append({
                    'timestamp': pd.to_datetime(
                        trade.participant_timestamp,
                        unit='ns',
                        utc=True
                    ),
                    'price': float(trade.price),
                    'size': int(trade.size),
                    'exchange': getattr(trade, 'exchange', None),
                    'conditions': getattr(trade, 'conditions', [])
                })

            logger.info(f"Fetched {len(trades_data)} trades for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            raise

        if not trades_data:
            return pd.DataFrame(columns=['timestamp', 'price', 'size', 'exchange', 'conditions'])

        df = pd.DataFrame(trades_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save to cache
        self._save_to_cache(df, ticker, date, 'trades')

        return df

    def fetch_quotes(
        self,
        ticker: str,
        date: str,
        limit: int = 50000,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch all quotes for a ticker on a given date.

        Uses S3 flat files if available (10-20x faster), falls back to API.

        Args:
            ticker: Stock ticker symbol (e.g., 'QQQ')
            date: Date string in YYYY-MM-DD format
            limit: Max results per request (default 50000, only for API)
            force_refresh: Bypass cache and re-fetch

        Returns:
            DataFrame with columns: [timestamp, bid_price, bid_size, ask_price, ask_size, ...]
        """
        # Check cache first
        if not force_refresh:
            cached = self._load_from_cache(ticker, date, 'quotes')
            if cached is not None:
                return cached

        # Try S3 first (much faster)
        if self.s3_fetcher is not None:
            try:
                logger.info(f"Fetching quotes via S3 for {ticker} on {date}")
                df = self.s3_fetcher.fetch_quotes(ticker, date)

                if not df.empty:
                    # Save to cache
                    self._save_to_cache(df, ticker, date, 'quotes')
                    return df
                else:
                    logger.warning(f"S3 returned no quotes, trying API...")
            except Exception as e:
                logger.warning(f"S3 fetch failed, falling back to API: {e}")

        # Fall back to REST API
        return self._fetch_quotes_api(ticker, date, limit)

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    def _fetch_quotes_api(
        self,
        ticker: str,
        date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch quotes using REST API (slower but reliable fallback).
        """
        logger.info(f"Fetching quotes via API for {ticker} on {date}")

        quotes_data = []

        try:
            # Use list_quotes which handles pagination automatically
            for quote in self.client.list_quotes(
                ticker=ticker,
                timestamp=date,
                limit=limit
            ):
                quotes_data.append({
                    'timestamp': pd.to_datetime(
                        quote.participant_timestamp,
                        unit='ns',
                        utc=True
                    ),
                    'bid_price': float(quote.bid_price) if quote.bid_price else 0.0,
                    'bid_size': int(quote.bid_size) if quote.bid_size else 0,
                    'ask_price': float(quote.ask_price) if quote.ask_price else 0.0,
                    'ask_size': int(quote.ask_size) if quote.ask_size else 0,
                    'bid_exchange': getattr(quote, 'bid_exchange', None),
                    'ask_exchange': getattr(quote, 'ask_exchange', None)
                })

            logger.info(f"Fetched {len(quotes_data)} quotes for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            raise

        if not quotes_data:
            return pd.DataFrame(columns=[
                'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size',
                'bid_exchange', 'ask_exchange'
            ])

        df = pd.DataFrame(quotes_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save to cache
        self._save_to_cache(df, ticker, date, 'quotes')

        return df

    def fetch_tick_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        include_quotes: bool = True
    ) -> pd.DataFrame:
        """
        Fetch combined tick data (trades + quotes) for a date range.

        Merges trades and quotes on nearest timestamp using asof merge.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_quotes: Whether to include quote data (default True)

        Returns:
            DataFrame with columns:
            [timestamp, price, volume, bid_price, bid_size, ask_price, ask_size]
        """
        logger.info(f"Fetching tick data for {ticker} from {start_date} to {end_date}")

        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []
        current = start

        while current <= end:
            date_str = current.strftime('%Y-%m-%d')

            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            try:
                # Fetch trades for this day
                trades_df = self.fetch_trades(ticker, date_str)

                if trades_df.empty:
                    current += timedelta(days=1)
                    continue

                if include_quotes:
                    # Fetch quotes for this day
                    quotes_df = self.fetch_quotes(ticker, date_str)

                    if not quotes_df.empty:
                        # Sort dataframes by timestamp
                        trades_df = trades_df.sort_values('timestamp')
                        quotes_df = quotes_df.sort_values('timestamp')

                        # Remove rows with null timestamps to avoid merge_asof error
                        # "Merge keys contain null values on right side"
                        quotes_df = quotes_df.dropna(subset=['timestamp'])

                        if quotes_df.empty:
                            logger.warning(f"All quotes have null timestamps for {date_str}, using trades only")
                            trades_df['bid_price'] = np.nan
                            trades_df['bid_size'] = np.nan
                            trades_df['ask_price'] = np.nan
                            trades_df['ask_size'] = np.nan
                            all_data.append(trades_df)
                            current += timedelta(days=1)
                            continue

                        # Merge trades with nearest quotes using asof merge
                        merged = pd.merge_asof(
                            trades_df,
                            quotes_df[['timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size']],
                            on='timestamp',
                            direction='backward'  # Use most recent quote
                        )
                        all_data.append(merged)
                    else:
                        # No quotes, just use trades
                        trades_df['bid_price'] = np.nan
                        trades_df['bid_size'] = np.nan
                        trades_df['ask_price'] = np.nan
                        trades_df['ask_size'] = np.nan
                        all_data.append(trades_df)
                else:
                    all_data.append(trades_df)

                # Rate limiting - paid tier has higher limits
                time.sleep(0.5)  # Short delay between days to avoid overwhelming API

            except Exception as e:
                logger.warning(f"Error fetching data for {date_str}: {e}")

            current += timedelta(days=1)

        if not all_data:
            logger.warning(f"No tick data found for {ticker}")
            return pd.DataFrame(columns=[
                'timestamp', 'price', 'size', 'bid_price', 'bid_size', 'ask_price', 'ask_size'
            ])

        # Combine all days
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)

        # Rename 'size' to 'volume' for consistency
        result = result.rename(columns={'size': 'volume'})

        # Select and order columns
        output_cols = ['timestamp', 'price', 'volume', 'bid_price', 'bid_size', 'ask_price', 'ask_size']
        available_cols = [c for c in output_cols if c in result.columns]
        result = result[available_cols]

        # Forward fill quote data (quotes don't update on every trade)
        quote_cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size']
        for col in quote_cols:
            if col in result.columns:
                result[col] = result[col].ffill()

        logger.info(f"Combined tick data: {len(result)} records from {start_date} to {end_date}")

        return result

    def fetch_intraday_tick_data(
        self,
        ticker: str,
        date: str,
        start_time: str = "09:30",
        end_time: str = "16:00"
    ) -> pd.DataFrame:
        """
        Fetch tick data for a single day, filtered to market hours.

        Args:
            ticker: Stock ticker symbol
            date: Date (YYYY-MM-DD)
            start_time: Start time (HH:MM) in Eastern Time
            end_time: End time (HH:MM) in Eastern Time

        Returns:
            DataFrame filtered to market hours
        """
        df = self.fetch_tick_data(ticker, date, date)

        if df.empty:
            return df

        # Convert to Eastern Time for filtering
        df['timestamp_et'] = df['timestamp'].dt.tz_convert('US/Eastern')

        # Filter to market hours
        df = df[
            (df['timestamp_et'].dt.strftime('%H:%M') >= start_time) &
            (df['timestamp_et'].dt.strftime('%H:%M') < end_time)
        ]

        # Drop helper column
        df = df.drop(columns=['timestamp_et'])

        return df.reset_index(drop=True)


    @retry_with_backoff(max_retries=5, base_delay=2.0)
    def fetch_agg_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timeframe_minutes: int = 1
    ) -> pd.DataFrame:
        """
        Fetch aggregated bar data (lighter rate limits than tick data).

        This is an alternative for free tier users who can't access tick data
        due to rate limits. Uses the /aggs endpoint instead of /trades.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe_minutes: Bar timeframe in minutes (default 1)

        Returns:
            DataFrame with OHLCV columns
        """
        logger.info(f"Fetching {timeframe_minutes}min bars for {ticker} from {start_date} to {end_date}")

        bars_data = []

        try:
            for bar in self.client.list_aggs(
                ticker=ticker,
                multiplier=timeframe_minutes,
                timespan="minute",
                from_=start_date,
                to=end_date,
                limit=50000
            ):
                bars_data.append({
                    'timestamp': pd.to_datetime(bar.timestamp, unit='ms', utc=True),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'vwap': float(bar.vwap) if bar.vwap else None,
                    'transactions': int(bar.transactions) if bar.transactions else None
                })

            logger.info(f"Fetched {len(bars_data)} bars for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            raise

        if not bars_data:
            return pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'
            ])

        df = pd.DataFrame(bars_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df


# Utility function for quick testing
def test_polygon_connection(api_key: str) -> bool:
    """
    Test if Polygon.io API connection works.

    Args:
        api_key: Polygon.io API key

    Returns:
        True if connection successful
    """
    try:
        fetcher = PolygonDataFetcher(api_key)
        # Try to fetch a small amount of data
        client = fetcher.client
        # Just check if we can make a request
        aggs = list(client.list_aggs("AAPL", 1, "minute", "2024-01-02", "2024-01-02", limit=1))
        logger.info("Polygon.io connection test successful")
        return True
    except Exception as e:
        logger.error(f"Polygon.io connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import os

    logging.basicConfig(level=logging.INFO)

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Please set POLYGON_API_KEY environment variable")
        exit(1)

    fetcher = PolygonDataFetcher(api_key)

    # Fetch one day of QQQ tick data
    df = fetcher.fetch_intraday_tick_data("QQQ", "2024-12-20")
    print(f"Fetched {len(df)} tick records")
    print(df.head())
    print(df.tail())
