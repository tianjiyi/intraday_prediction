"""
Download QQQ trades and quotes for Sep 10 - Nov 10, 2025
Uses Polygon S3 flat files for fast bulk download.
"""
import sys
import os
from pathlib import Path
from datetime import date, timedelta
import logging
import yaml

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

from market_regime.polygon_fetcher import PolygonDataFetcher

def load_config():
    """Load config from parent directory."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_trading_days(start_date: date, end_date: date) -> list:
    """Generate list of trading days (Mon-Fri, excluding holidays)."""
    # Major US market holidays in 2025
    holidays_2025 = {
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
    }

    days = []
    current = start_date
    while current <= end_date:
        # Skip weekends (5=Sat, 6=Sun)
        if current.weekday() < 5 and current not in holidays_2025:
            days.append(current)
        current += timedelta(days=1)
    return days


def main():
    # Date range: Sep 10 - Nov 10, 2025
    start_date = date(2025, 9, 10)
    end_date = date(2025, 11, 10)
    symbol = "QQQ"

    trading_days = get_trading_days(start_date, end_date)
    logger.info(f"Downloading {symbol} data for {len(trading_days)} trading days")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Get API key from environment
    config = load_config()
    api_key = config.get('polygon', {}).get('api_key')
    if not api_key:
        logger.error("Polygon API key not found in config.yaml")
        return

    # Initialize fetcher (will use S3 if credentials available)
    fetcher = PolygonDataFetcher(
        api_key=api_key,
        use_cache=True,
        use_s3=True
    )

    # Download trades and quotes for each day
    trades_count = 0
    quotes_count = 0

    for i, day in enumerate(trading_days):
        date_str = day.strftime('%Y-%m-%d')
        logger.info(f"[{i+1}/{len(trading_days)}] Downloading {date_str}...")

        try:
            # Download trades
            trades = fetcher.fetch_trades(symbol, date_str)
            if not trades.empty:
                trades_count += len(trades)
                logger.info(f"  Trades: {len(trades):,}")

            # Download quotes
            quotes = fetcher.fetch_quotes(symbol, date_str)
            if not quotes.empty:
                quotes_count += len(quotes)
                logger.info(f"  Quotes: {len(quotes):,}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    logger.info("=" * 50)
    logger.info(f"Download complete!")
    logger.info(f"Total trades: {trades_count:,}")
    logger.info(f"Total quotes: {quotes_count:,}")
    logger.info(f"Data cached in: market_regime/data/{symbol}/")


if __name__ == "__main__":
    main()
