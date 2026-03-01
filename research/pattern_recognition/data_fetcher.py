"""
Data fetching module using Alpaca API.

Fetches OHLCV (Open, High, Low, Close, Volume) bars for stock symbols.
Follows patterns established in prediction_service.py.
"""

import os
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import pytz

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca-py not installed. Install with: pip install alpaca-py")

from .models import DataFetchError

logger = logging.getLogger(__name__)


# Timeframe mapping to Alpaca TimeFrame objects
TIMEFRAME_MAP = {
    "1min": TimeFrame.Minute if ALPACA_AVAILABLE else None,
    "5min": TimeFrame(5, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "15min": TimeFrame(15, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "30min": TimeFrame(30, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "1hour": TimeFrame.Hour if ALPACA_AVAILABLE else None,
    "daily": TimeFrame.Day if ALPACA_AVAILABLE else None,
}

# Bars per RTH day for different timeframes (RTH = 390 minutes)
BARS_PER_DAY = {
    "1min": 390,
    "5min": 78,
    "15min": 26,
    "30min": 13,
    "1hour": 7,
    "daily": 1,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Searches for config.yaml in:
    1. Provided path
    2. Current directory
    3. Parent directories up to 3 levels

    Args:
        config_path: Optional path to config file

    Returns:
        Configuration dictionary
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Search for config.yaml
    search_paths = [
        Path("config.yaml"),
        Path("../config.yaml"),
        Path("../../config.yaml"),
        Path("../../../config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
        Path(__file__).parent.parent.parent / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found config at {path}")
            with open(path, 'r') as f:
                return yaml.safe_load(f)

    logger.warning("config.yaml not found, using environment variables")
    return {}


def get_alpaca_client(config: Optional[Dict] = None) -> 'StockHistoricalDataClient':
    """
    Get Alpaca historical data client.

    API keys are loaded from:
    1. Config dict
    2. Environment variables

    Args:
        config: Optional config dict with API keys

    Returns:
        StockHistoricalDataClient instance

    Raises:
        DataFetchError: If API keys not found or alpaca-py not installed
    """
    if not ALPACA_AVAILABLE:
        raise DataFetchError("alpaca-py not installed. Run: pip install alpaca-py")

    if config is None:
        config = load_config()

    api_key = config.get('ALPACA_KEY_ID') or os.getenv('ALPACA_KEY_ID')
    secret_key = config.get('ALPACA_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        raise DataFetchError(
            "Alpaca API keys required. Set ALPACA_KEY_ID and ALPACA_SECRET_KEY "
            "in config.yaml or environment variables."
        )

    return StockHistoricalDataClient(api_key, secret_key)


def calculate_days_to_fetch(
    timeframe: str,
    lookback_bars: int,
    buffer_factor: float = 1.5
) -> int:
    """
    Calculate number of calendar days to fetch based on timeframe.

    Accounts for:
    - Weekends (no trading)
    - Market hours (RTH = 390 minutes/day)
    - Buffer for holidays

    Args:
        timeframe: Timeframe string
        lookback_bars: Number of bars needed
        buffer_factor: Extra buffer for weekends/holidays

    Returns:
        Number of calendar days to request
    """
    bars_per_day = BARS_PER_DAY.get(timeframe, 390)

    # Calculate trading days needed
    trading_days = max(1, lookback_bars / bars_per_day)

    # Add buffer for weekends and holidays
    calendar_days = int(trading_days * buffer_factor) + 5

    # Cap to reasonable range (1825 days = ~5 years for SIP feed)
    return min(calendar_days, 1825)


def fetch_bars(
    tickers: List[str],
    lookback_bars: int = 200,
    timeframe: str = "5min",
    config: Optional[Dict] = None,
    rth_only: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV bars for multiple tickers from Alpaca.

    Args:
        tickers: List of stock symbols (e.g., ["QQQ", "SPY", "AAPL"])
        lookback_bars: Number of bars to fetch per ticker
        timeframe: Timeframe string ("1min", "5min", "15min", "30min", "daily")
        config: Optional config dict with API keys
        rth_only: Filter to Regular Trading Hours only (9:30 AM - 4:00 PM ET)

    Returns:
        Dict mapping ticker symbol to DataFrame with columns:
        - timestamp: DateTime
        - open, high, low, close: Prices
        - volume: Volume
        - vwap: Volume-weighted average price

    Example:
        >>> data = fetch_bars(["QQQ", "SPY"], lookback_bars=200, timeframe="5min")
        >>> print(data["QQQ"].head())
    """
    client = get_alpaca_client(config)

    # Get timeframe object
    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        raise DataFetchError(f"Unsupported timeframe: {timeframe}")

    # Calculate date range
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    # Calculate days needed
    days_to_fetch = calculate_days_to_fetch(timeframe, lookback_bars)
    start_time = now - timedelta(days=days_to_fetch)

    logger.info(f"Fetching {lookback_bars} bars of {timeframe} data "
               f"for {len(tickers)} tickers (looking back {days_to_fetch} days)")

    results = {}

    for ticker in tickers:
        try:
            logger.debug(f"Fetching {ticker}...")

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start_time,
                end=now,
                feed=DataFeed.SIP  # Use SIP feed for paid accounts (5+ years history)
            )

            bars = client.get_stock_bars(request)
            df = bars.df.reset_index()

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue

            # Convert timestamp to Eastern time
            df['timestamp'] = df['timestamp'].dt.tz_convert(eastern)

            # Filter to RTH if requested (not for daily bars)
            if rth_only and timeframe != "daily":
                df = df.set_index('timestamp')
                df = df.between_time('09:30', '15:59')
                df = df.reset_index()

            # Take last N bars
            df = df.tail(lookback_bars)

            # Rename columns to lowercase
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'VWAP': 'vwap',
                'trade_count': 'trade_count'
            })

            # Add metadata
            df['ticker'] = ticker
            df['timeframe'] = timeframe

            results[ticker] = df
            logger.info(f"Fetched {len(df)} bars for {ticker}")

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            continue

    return results


def fetch_single_ticker(
    ticker: str,
    lookback_bars: int = 200,
    timeframe: str = "5min",
    config: Optional[Dict] = None,
    rth_only: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a single ticker.

    Convenience function for single-ticker fetches.

    Args:
        ticker: Stock symbol
        lookback_bars: Number of bars to fetch
        timeframe: Timeframe string
        config: Optional config dict
        rth_only: Filter to RTH

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataFetchError: If fetch fails or no data returned
    """
    data = fetch_bars([ticker], lookback_bars, timeframe, config, rth_only)

    if ticker not in data:
        raise DataFetchError(f"Failed to fetch data for {ticker}")

    return data[ticker]


def validate_ticker(ticker: str, config: Optional[Dict] = None) -> bool:
    """
    Check if a ticker is valid and has data available.

    Args:
        ticker: Stock symbol to validate
        config: Optional config dict

    Returns:
        True if ticker is valid and has recent data
    """
    try:
        df = fetch_single_ticker(ticker, lookback_bars=10, timeframe="daily", config=config)
        return len(df) > 0
    except Exception:
        return False


def get_supported_timeframes() -> List[str]:
    """
    Get list of supported timeframe strings.

    Returns:
        List of timeframe strings
    """
    return list(TIMEFRAME_MAP.keys())


def get_market_status() -> Dict[str, Any]:
    """
    Get current market status (open/closed).

    Returns:
        Dict with market status information
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    is_weekday = now.weekday() < 5
    hour = now.hour
    minute = now.minute

    market_open = 9 * 60 + 30  # 9:30 AM
    market_close = 16 * 60     # 4:00 PM
    current_minutes = hour * 60 + minute

    is_market_hours = market_open <= current_minutes < market_close

    return {
        "timestamp": now.isoformat(),
        "is_weekday": is_weekday,
        "is_market_hours": is_market_hours,
        "is_open": is_weekday and is_market_hours,
        "current_time_et": now.strftime("%H:%M:%S ET")
    }
