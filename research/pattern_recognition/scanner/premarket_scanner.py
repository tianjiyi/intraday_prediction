"""
Premarket Scanner - Find unusual premarket movers.

Scans for stocks meeting criteria:
- Market cap >= $1 billion
- Gap up/down >= 5% from previous close
- Volume >= 2x average (unusual volume)

Runs from 4:00am - 9:30am ET, adding qualifying stocks to floating watchlist.
"""

import logging
import time as time_module
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from .config import SCANNER_CONFIG, load_config_from_yaml
from .watchlist_manager import WatchlistManager
from .fundamentals_service import get_market_cap

logger = logging.getLogger(__name__)


@dataclass
class PremarketMover:
    """
    Represents a stock identified as an unusual premarket mover.

    Attributes:
        symbol: Ticker symbol
        gap_percent: Gap up/down % from previous close
        premarket_volume: Total volume during premarket
        avg_volume: 20-day average volume
        volume_ratio: premarket_volume / avg_volume
        market_cap: Market capitalization in dollars
        prev_close: Previous day's closing price
        premarket_high: High during premarket
        premarket_low: Low during premarket
        premarket_open: Opening price in premarket (4am)
        current_price: Most recent premarket price
        reason: Why stock qualified ('gap_up', 'gap_down', 'volume_spike', 'both')
    """
    symbol: str
    gap_percent: float
    premarket_volume: int
    avg_volume: int
    volume_ratio: float
    market_cap: float
    prev_close: float
    premarket_high: float
    premarket_low: float
    premarket_open: float
    current_price: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'gap_percent': self.gap_percent,
            'premarket_volume': self.premarket_volume,
            'avg_volume': self.avg_volume,
            'volume_ratio': self.volume_ratio,
            'market_cap': self.market_cap,
            'prev_close': self.prev_close,
            'premarket_high': self.premarket_high,
            'premarket_low': self.premarket_low,
            'premarket_open': self.premarket_open,
            'reason': self.reason,
        }

    def __repr__(self):
        direction = "↑" if self.gap_percent > 0 else "↓"
        return (
            f"<PremarketMover {self.symbol} {direction}{abs(self.gap_percent):.1f}% "
            f"vol={self.volume_ratio:.1f}x mcap=${self.market_cap/1e9:.1f}B>"
        )


class PremarketScanner:
    """
    Scans for unusual premarket movers.

    Criteria (configurable):
    - Market cap >= $1 billion
    - Gap up/down >= 5% from previous close
    - Volume >= 2x average (unusual volume)

    Usage:
        scanner = PremarketScanner()
        movers = scanner.scan()

        for mover in movers:
            print(f"{mover.symbol}: {mover.gap_percent:+.1f}% gap, {mover.volume_ratio:.1f}x volume")
    """

    def __init__(self, config: Dict[str, Any] = None, target_date: date = None):
        """
        Initialize PremarketScanner.

        Args:
            config: Configuration dict. If None, loads from SCANNER_CONFIG.
            target_date: Date to scan. If None, defaults to today.
        """
        self.config = config or load_config_from_yaml()
        premarket_config = self.config.get('premarket', {})

        # Target date (default to today)
        self.target_date = target_date or date.today()

        # Criteria thresholds
        self.min_market_cap = premarket_config.get('min_market_cap', 1_000_000_000)
        self.min_gap_percent = premarket_config.get('min_gap_percent', 5.0)
        self.min_volume_ratio = premarket_config.get('min_volume_ratio', 2.0)
        self.max_symbols = premarket_config.get('max_symbols_to_scan', 500)
        self.batch_size = premarket_config.get('batch_size', 100)
        self.batch_delay = premarket_config.get('batch_delay_seconds', 1)

        # Data feed (SIP for full market, IEX for free tier)
        feed_type = premarket_config.get('data_feed', 'iex').lower()
        self.data_feed = DataFeed.SIP if feed_type == 'sip' else DataFeed.IEX

        # Timezone
        self.timezone = pytz.timezone('US/Eastern')

        # API clients
        self._init_clients()

        # Watchlist manager
        self.watchlist_manager = WatchlistManager()

    def _init_clients(self):
        """Initialize Alpaca API clients."""
        alpaca_config = self.config.get('alpaca', {})
        api_key = alpaca_config.get('api_key', '')
        secret_key = alpaca_config.get('secret_key', '')

        if not api_key or not secret_key:
            # Try loading from environment or config.yaml
            import os
            api_key = os.getenv('ALPACA_KEY_ID', '')
            secret_key = os.getenv('ALPACA_SECRET_KEY', '')

        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=alpaca_config.get('paper', False)
        )

    def is_premarket_hours(self) -> bool:
        """
        Check if current time is within premarket hours (4am - 9:30am ET).

        Returns:
            True if within premarket hours
        """
        now = datetime.now(self.timezone)
        premarket_config = self.config.get('premarket', {})

        start_time = datetime.strptime(
            premarket_config.get('start_time', '04:00'), '%H:%M'
        ).time()
        end_time = datetime.strptime(
            premarket_config.get('end_time', '09:30'), '%H:%M'
        ).time()

        return start_time <= now.time() <= end_time

    def get_tradable_stocks(self) -> List[str]:
        """
        Get list of tradable US stocks from Alpaca.

        Filters:
        - US equity only
        - Active status
        - Tradable

        Returns:
            List of ticker symbols
        """
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )
            assets = self.trading_client.get_all_assets(request)

            # Filter for tradable stocks
            symbols = [
                asset.symbol for asset in assets
                if asset.tradable and asset.shortable is not None
            ]

            logger.info(f"Found {len(symbols)} tradable US stocks")
            # Return all if max_symbols = 0, otherwise limit
            if self.max_symbols > 0:
                return symbols[:self.max_symbols]
            return symbols

        except Exception as e:
            logger.error(f"Error fetching tradable stocks: {e}")
            return []

    def get_previous_close(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get previous day's closing prices.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to close price
        """
        result = {}
        # End just before midnight of target_date (so we exclude target_date's bar)
        end = self.timezone.localize(
            datetime.combine(self.target_date, datetime.min.time())
        ) - timedelta(seconds=1)
        start = end - timedelta(days=5)  # Look back to handle weekends

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=self.data_feed
            )
            bars = self.data_client.get_stock_bars(request)
            bars_data = bars.data if hasattr(bars, 'data') else bars

            for symbol in symbols:
                if symbol in bars_data and len(bars_data[symbol]) > 0:
                    result[symbol] = float(bars_data[symbol][-1].close)

        except Exception as e:
            logger.error(f"Error fetching previous closes: {e}")

        return result

    def get_average_volume(self, symbols: List[str], days: int = 20) -> Dict[str, int]:
        """
        Get average daily volume over the past N days.

        Args:
            symbols: List of ticker symbols
            days: Number of days for average (default 20)

        Returns:
            Dict mapping symbol to average volume
        """
        result = {}
        # End at midnight of target_date (so we get days before)
        end = self.timezone.localize(
            datetime.combine(self.target_date, datetime.min.time())
        )
        start = end - timedelta(days=days + 5)  # Extra buffer for weekends

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=self.data_feed
            )
            bars = self.data_client.get_stock_bars(request)
            bars_data = bars.data if hasattr(bars, 'data') else bars

            for symbol in symbols:
                if symbol in bars_data and len(bars_data[symbol]) > 0:
                    volumes = [bar.volume for bar in bars_data[symbol]]
                    result[symbol] = int(sum(volumes) / len(volumes))

        except Exception as e:
            logger.error(f"Error fetching average volumes: {e}")

        return result

    def get_premarket_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get premarket OHLCV data for symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to premarket data dict with:
                - open: First trade price (4am)
                - high: Premarket high
                - low: Premarket low
                - close: Most recent price (9:30am for historical)
                - volume: Total premarket volume
        """
        result = {}

        # Premarket is 4:00am - 9:30am ET for target_date
        start = self.timezone.localize(
            datetime.combine(self.target_date, time(4, 0))
        )
        end = self.timezone.localize(
            datetime.combine(self.target_date, time(9, 30))
        )

        # For current date, use now if before 9:30am
        if self.target_date == date.today():
            now = datetime.now(self.timezone)
            if now.time() < time(9, 30):
                end = now

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=self.data_feed
            )
            bars = self.data_client.get_stock_bars(request)
            bars_data = bars.data if hasattr(bars, 'data') else bars

            for symbol in symbols:
                if symbol in bars_data and len(bars_data[symbol]) > 0:
                    symbol_bars = bars_data[symbol]
                    result[symbol] = {
                        'open': float(symbol_bars[0].open),
                        'high': max(bar.high for bar in symbol_bars),
                        'low': min(bar.low for bar in symbol_bars),
                        'close': float(symbol_bars[-1].close),
                        'volume': sum(bar.volume for bar in symbol_bars),
                    }

        except Exception as e:
            logger.error(f"Error fetching premarket data: {e}")

        return result

    def get_snapshots(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get current snapshots with latest quotes.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to snapshot data
        """
        result = {}
        try:
            # Get snapshots in batches to avoid API limits
            batch_size = 100
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                request = StockSnapshotRequest(symbol_or_symbols=batch)
                snapshots = self.data_client.get_stock_snapshot(request)

                for symbol, snapshot in snapshots.items():
                    if snapshot.latest_trade:
                        result[symbol] = {
                            'price': float(snapshot.latest_trade.price),
                            'prev_close': float(snapshot.previous_daily_bar.close)
                            if snapshot.previous_daily_bar else None,
                        }

        except Exception as e:
            logger.error(f"Error fetching snapshots: {e}")

        return result

    def _evaluate_movers(
        self,
        symbols: List[str],
        prev_closes: Dict[str, float],
        avg_volumes: Dict[str, int],
        premarket_data: Dict[str, Dict],
        snapshots: Dict[str, Any] = None
    ) -> List[PremarketMover]:
        """
        Evaluate which symbols meet mover criteria.

        Args:
            symbols: List of symbols to evaluate
            prev_closes: Dict of previous close prices
            avg_volumes: Dict of average volumes
            premarket_data: Dict of premarket OHLCV data
            snapshots: Dict of current snapshots (optional)

        Returns:
            List of PremarketMover instances meeting criteria
        """
        snapshots = snapshots or {}
        movers = []

        for symbol in symbols:
            try:
                # Need all data to evaluate
                if symbol not in prev_closes:
                    continue
                if symbol not in premarket_data:
                    continue

                prev_close = prev_closes[symbol]
                pm_data = premarket_data[symbol]
                avg_vol = avg_volumes.get(symbol, 0)

                # Current price (from snapshot or premarket close)
                current_price = pm_data['close']
                if symbol in snapshots:
                    current_price = snapshots[symbol]['price']

                # Calculate gap %
                gap_percent = ((current_price - prev_close) / prev_close) * 100

                # Calculate volume ratio
                volume_ratio = pm_data['volume'] / avg_vol if avg_vol > 0 else 0

                # Get market cap from database
                market_cap = get_market_cap(symbol)

                # Check criteria (must meet ALL: gap UP, volume, market cap)
                meets_gap = gap_percent >= self.min_gap_percent  # Only positive gaps
                meets_volume = volume_ratio >= self.min_volume_ratio
                meets_market_cap = market_cap >= self.min_market_cap

                if not (meets_gap and meets_volume and meets_market_cap):
                    continue

                reason = 'gap_up'

                # Create mover object
                mover = PremarketMover(
                    symbol=symbol,
                    gap_percent=gap_percent,
                    premarket_volume=pm_data['volume'],
                    avg_volume=avg_vol,
                    volume_ratio=volume_ratio,
                    market_cap=market_cap,
                    prev_close=prev_close,
                    premarket_high=pm_data['high'],
                    premarket_low=pm_data['low'],
                    premarket_open=pm_data['open'],
                    current_price=current_price,
                    reason=reason,
                )

                movers.append(mover)
                logger.debug(f"Found mover: {mover}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        return movers

    def scan(
        self,
        symbols: List[str] = None,
        save_to_db: bool = True,
        scan_all: bool = False
    ) -> List[PremarketMover]:
        """
        Run premarket scan for unusual movers.

        Args:
            symbols: List of symbols to scan. If None, scans all tradable stocks.
            save_to_db: If True, saves qualifying movers to floating watchlist.
            scan_all: If True, ignores max_symbols limit and scans all stocks.

        Returns:
            List of PremarketMover instances meeting criteria
        """
        logger.info(f"Starting premarket scan for {self.target_date}...")

        # Get symbols to scan
        if symbols is None:
            # Temporarily set max_symbols to 0 if scan_all is True
            original_max = self.max_symbols
            if scan_all:
                self.max_symbols = 0
            symbols = self.get_tradable_stocks()
            self.max_symbols = original_max

        if not symbols:
            logger.warning("No symbols to scan")
            return []

        logger.info(f"Scanning {len(symbols)} symbols in batches of {self.batch_size}...")

        all_movers = []
        total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(symbols), self.batch_size):
            batch = symbols[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)...")

            # Fetch data for this batch
            prev_closes = self.get_previous_close(batch)
            avg_volumes = self.get_average_volume(batch)
            premarket_data = self.get_premarket_data(batch)

            # Only fetch live snapshots for current date
            if self.target_date == date.today():
                snapshots = self.get_snapshots(batch)
            else:
                snapshots = {}  # Use premarket close for historical

            # Evaluate movers in this batch
            batch_movers = self._evaluate_movers(
                batch, prev_closes, avg_volumes, premarket_data, snapshots
            )
            all_movers.extend(batch_movers)

            logger.info(f"Batch {batch_num}: found {len(batch_movers)} movers")

            # Delay before next batch (except last)
            if batch_idx + self.batch_size < len(symbols):
                time_module.sleep(self.batch_delay)

        # Sort by absolute gap percentage
        all_movers.sort(key=lambda x: abs(x.gap_percent), reverse=True)

        logger.info(f"Found {len(all_movers)} unusual premarket movers total")

        # Save to database
        if save_to_db and all_movers:
            self.watchlist_manager.add_to_floating([m.to_dict() for m in all_movers])

        return all_movers

    def scan_with_default_watchlist(self) -> List[PremarketMover]:
        """
        Scan only the default watchlist symbols for premarket activity.

        Useful for quick scan of core symbols before full market scan.

        Returns:
            List of PremarketMover instances
        """
        default_symbols = self.watchlist_manager.get_default_watchlist()
        return self.scan(symbols=default_symbols, save_to_db=False)


def run_premarket_scanner():
    """
    Run premarket scanner (standalone execution).

    Can be called from CLI or scheduler.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
    )

    scanner = PremarketScanner()

    if not scanner.is_premarket_hours():
        logger.warning("Not within premarket hours (4:00am - 9:30am ET)")
        # Run anyway for testing
        logger.info("Running scan anyway for testing...")

    movers = scanner.scan()

    print("\n" + "=" * 70)
    print("  PREMARKET MOVERS")
    print("=" * 70)

    if movers:
        print(f"\n  Found {len(movers)} unusual movers:\n")
        for i, mover in enumerate(movers[:20], 1):  # Top 20
            direction = "↑" if mover.gap_percent > 0 else "↓"
            print(
                f"  {i:2d}. {mover.symbol:6s}  {direction}{abs(mover.gap_percent):5.1f}%  "
                f"Vol: {mover.volume_ratio:4.1f}x  "
                f"Reason: {mover.reason}"
            )
    else:
        print("\n  No unusual movers found meeting criteria.")

    print("\n" + "=" * 70)

    return movers


if __name__ == "__main__":
    run_premarket_scanner()
