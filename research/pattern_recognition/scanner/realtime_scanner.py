"""
Real-time Pattern Scanner - Scan watchlist for ascending triangles.

Runs during RTH (9:30am - 4:00pm ET), scanning every minute for patterns.
Logs pattern findings to TimescaleDB with deduplication.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import SCANNER_CONFIG, load_config_from_yaml
from .watchlist_manager import WatchlistManager
from .db_models import PatternFinding, RealtimeScanLog, get_session

# Import pattern detector from parent module
from ..ascending_triangle import AscendingTriangleDetector, AscendingTrianglePattern

logger = logging.getLogger(__name__)


class RealtimePatternScanner:
    """
    Real-time pattern scanner for ascending triangles.

    Scans combined watchlist every minute during RTH.
    Logs patterns to TimescaleDB with deduplication.

    Usage:
        scanner = RealtimePatternScanner()

        # Run continuous scanning
        asyncio.run(scanner.run())

        # Or single scan cycle
        patterns = scanner.scan_all()
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RealtimePatternScanner.

        Args:
            config: Configuration dict. If None, loads from SCANNER_CONFIG.
        """
        self.config = config or load_config_from_yaml()
        realtime_config = self.config.get('realtime', {})

        # Scan settings
        self.scan_interval = realtime_config.get('scan_interval_seconds', 60)
        self.timeframe = realtime_config.get('timeframe', '1min')
        self.lookback_bars = realtime_config.get('lookback_bars', 200)
        self.min_confidence = realtime_config.get('min_confidence', 0.5)

        # Deduplication settings
        self.dedup_time_minutes = realtime_config.get('dedup_time_minutes', 30)
        self.dedup_price_tolerance = realtime_config.get('dedup_price_tolerance', 0.01)

        # Pattern detection settings
        pattern_config = self.config.get('pattern_detection', {})
        self.detector = AscendingTriangleDetector(
            resistance_tolerance=pattern_config.get('resistance_tolerance'),
            min_support_slope=pattern_config.get('min_support_slope', 0.00001),
            zigzag_deviation=pattern_config.get('zigzag_deviation'),
            min_bars=pattern_config.get('min_bars'),
        )

        # Context settings
        self.pre_context_bars = pattern_config.get('pre_context_bars', 20)
        self.post_context_bars = pattern_config.get('post_context_bars', 20)
        self.analyze_breakout = pattern_config.get('analyze_breakout', True)

        # Timezone
        self.timezone = pytz.timezone('US/Eastern')

        # API client
        self._init_client()

        # Watchlist manager
        self.watchlist_manager = WatchlistManager()

        # Running state
        self._running = False

    def _init_client(self):
        """Initialize Alpaca data client."""
        alpaca_config = self.config.get('alpaca', {})
        api_key = alpaca_config.get('api_key', '')
        secret_key = alpaca_config.get('secret_key', '')

        if not api_key or not secret_key:
            import os
            api_key = os.getenv('ALPACA_KEY_ID', '')
            secret_key = os.getenv('ALPACA_SECRET_KEY', '')

        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def is_market_hours(self) -> bool:
        """
        Check if current time is within RTH (9:30am - 4:00pm ET).

        Returns:
            True if within regular trading hours
        """
        now = datetime.now(self.timezone)
        realtime_config = self.config.get('realtime', {})

        start_time = datetime.strptime(
            realtime_config.get('start_time', '09:30'), '%H:%M'
        ).time()
        end_time = datetime.strptime(
            realtime_config.get('end_time', '16:00'), '%H:%M'
        ).time()

        # Check weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return start_time <= now.time() <= end_time

    def get_timeframe_enum(self) -> TimeFrame:
        """Convert timeframe string to Alpaca TimeFrame enum."""
        timeframe_map = {
            '1min': TimeFrame.Minute,
            '5min': TimeFrame(5, 'Min'),
            '15min': TimeFrame(15, 'Min'),
            '30min': TimeFrame(30, 'Min'),
            '1hour': TimeFrame.Hour,
        }
        return timeframe_map.get(self.timeframe, TimeFrame.Minute)

    def fetch_bars(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with OHLCV data and timestamps, or None if error
        """
        try:
            now = datetime.now(self.timezone)

            # Calculate start time based on lookback
            # For 1min bars, 200 bars = ~3.3 hours
            # Add buffer for weekends/holidays
            if self.timeframe == '1min':
                start = now - timedelta(days=2)
            elif self.timeframe == '5min':
                start = now - timedelta(days=5)
            elif self.timeframe == '15min':
                start = now - timedelta(days=10)
            else:
                start = now - timedelta(days=20)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self.get_timeframe_enum(),
                start=start,
                end=now
            )

            bars = self.data_client.get_stock_bars(request)

            if symbol not in bars or len(bars[symbol]) == 0:
                logger.warning(f"No bars returned for {symbol}")
                return None

            symbol_bars = bars[symbol]

            # Filter to RTH only
            rth_bars = []
            for bar in symbol_bars:
                bar_time = bar.timestamp.astimezone(self.timezone)
                if datetime.strptime('09:30', '%H:%M').time() <= bar_time.time() <= datetime.strptime('16:00', '%H:%M').time():
                    rth_bars.append(bar)

            if len(rth_bars) < 50:  # Need minimum bars
                logger.warning(f"Insufficient RTH bars for {symbol}: {len(rth_bars)}")
                return None

            # Take last N bars
            rth_bars = rth_bars[-self.lookback_bars:]

            return {
                'symbol': symbol,
                'timestamps': [bar.timestamp for bar in rth_bars],
                'opens': [float(bar.open) for bar in rth_bars],
                'highs': [float(bar.high) for bar in rth_bars],
                'lows': [float(bar.low) for bar in rth_bars],
                'closes': [float(bar.close) for bar in rth_bars],
                'volumes': [int(bar.volume) for bar in rth_bars],
            }

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None

    def scan_symbol(self, symbol: str) -> Optional[AscendingTrianglePattern]:
        """
        Scan a single symbol for ascending triangle pattern.

        Args:
            symbol: Ticker symbol

        Returns:
            AscendingTrianglePattern if found, None otherwise
        """
        try:
            data = self.fetch_bars(symbol)
            if data is None:
                return None

            pattern = self.detector.detect(
                highs=data['highs'],
                lows=data['lows'],
                closes=data['closes'],
                ticker=symbol,
                timeframe=self.timeframe,
                timestamps=data['timestamps'],
                pre_context_bars=self.pre_context_bars,
                post_context_bars=self.post_context_bars,
                analyze_breakout=self.analyze_breakout,
            )

            if pattern and pattern.confidence >= self.min_confidence:
                logger.info(
                    f"Pattern found: {symbol} @ ${pattern.resistance_level:.2f} "
                    f"(confidence: {pattern.confidence:.1%})"
                )
                return pattern

            return None

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None

    def is_duplicate_pattern(
        self,
        pattern: AscendingTrianglePattern,
        session=None
    ) -> bool:
        """
        Check if pattern is a duplicate of recently logged pattern.

        Args:
            pattern: Pattern to check
            session: Database session (optional)

        Returns:
            True if duplicate, False if new pattern
        """
        if session is None:
            session = get_session()

        try:
            cutoff = datetime.utcnow() - timedelta(minutes=self.dedup_time_minutes)

            recent = session.query(PatternFinding).filter(
                PatternFinding.symbol == pattern.ticker,
                PatternFinding.timeframe == pattern.timeframe,
                PatternFinding.time > cutoff,
            ).order_by(PatternFinding.time.desc()).first()

            if recent is None:
                return False

            # Check resistance level similarity
            if recent.resistance_level:
                price_diff = abs(
                    float(recent.resistance_level) - pattern.resistance_level
                ) / pattern.resistance_level

                if price_diff < self.dedup_price_tolerance:
                    logger.debug(
                        f"Duplicate pattern: {pattern.ticker} resistance "
                        f"${pattern.resistance_level:.2f} vs ${recent.resistance_level:.2f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

    def log_pattern(self, pattern: AscendingTrianglePattern) -> bool:
        """
        Log pattern to database.

        Args:
            pattern: Pattern to log

        Returns:
            True if logged successfully
        """
        session = get_session()

        try:
            # Check for duplicate
            if self.is_duplicate_pattern(pattern, session):
                return False

            # Create database entry
            finding = PatternFinding.from_pattern(pattern)
            session.add(finding)
            session.commit()

            logger.info(f"Logged pattern: {pattern.ticker} to database")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error logging pattern: {e}")
            return False
        finally:
            session.close()

    def log_scan_cycle(
        self,
        symbols_scanned: int,
        patterns_found: int,
        duration_ms: int,
        status: str = 'completed',
        error_msg: str = None
    ):
        """
        Log scan cycle to database.

        Args:
            symbols_scanned: Number of symbols scanned
            patterns_found: Number of patterns found
            duration_ms: Scan duration in milliseconds
            status: Scan status ('completed', 'failed', 'timeout')
            error_msg: Error message if failed
        """
        session = get_session()

        try:
            log = RealtimeScanLog(
                scan_time=datetime.utcnow(),
                symbols_scanned=symbols_scanned,
                patterns_found=patterns_found,
                scan_duration_ms=duration_ms,
                status=status,
                error_msg=error_msg,
            )
            session.add(log)
            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error logging scan cycle: {e}")
        finally:
            session.close()

    def scan_all(self) -> List[AscendingTrianglePattern]:
        """
        Scan all symbols in combined watchlist.

        Returns:
            List of patterns found
        """
        start_time = datetime.now()
        watchlist = self.watchlist_manager.get_combined_watchlist()

        if not watchlist:
            logger.warning("Empty watchlist, nothing to scan")
            return []

        logger.info(f"Scanning {len(watchlist)} symbols...")

        patterns = []
        patterns_logged = 0

        for symbol in watchlist:
            pattern = self.scan_symbol(symbol)
            if pattern:
                patterns.append(pattern)
                if self.log_pattern(pattern):
                    patterns_logged += 1

        # Log scan cycle
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        self.log_scan_cycle(
            symbols_scanned=len(watchlist),
            patterns_found=len(patterns),
            duration_ms=duration_ms,
        )

        logger.info(
            f"Scan complete: {len(patterns)} patterns found, "
            f"{patterns_logged} logged ({duration_ms}ms)"
        )

        return patterns

    async def run(self):
        """
        Run continuous scanning loop during RTH.

        Scans every minute and logs patterns to database.
        """
        self._running = True
        logger.info("Starting real-time pattern scanner...")

        while self._running:
            try:
                if self.is_market_hours():
                    patterns = self.scan_all()

                    if patterns:
                        print(f"\n[{datetime.now(self.timezone).strftime('%H:%M:%S')}] "
                              f"Found {len(patterns)} patterns:")
                        for p in patterns:
                            print(f"  {p.ticker}: ${p.resistance_level:.2f} "
                                  f"({p.confidence:.1%} confidence)")
                else:
                    logger.debug("Outside market hours, skipping scan")

                # Wait for next interval
                await asyncio.sleep(self.scan_interval)

            except asyncio.CancelledError:
                logger.info("Scanner cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(self.scan_interval)

        logger.info("Real-time scanner stopped")

    def stop(self):
        """Stop the scanning loop."""
        self._running = False


async def run_realtime_scanner():
    """
    Run real-time scanner (standalone execution).

    Can be called from CLI or scheduler.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
    )

    scanner = RealtimePatternScanner()

    print("\n" + "=" * 70)
    print("  REAL-TIME PATTERN SCANNER")
    print("=" * 70)
    print(f"  Timeframe:      {scanner.timeframe}")
    print(f"  Lookback:       {scanner.lookback_bars} bars")
    print(f"  Min confidence: {scanner.min_confidence:.0%}")
    print(f"  Scan interval:  {scanner.scan_interval}s")
    print("=" * 70)

    if not scanner.is_market_hours():
        print("\n  WARNING: Outside market hours (9:30am - 4:00pm ET)")
        print("  Running single scan for testing...\n")

        patterns = scanner.scan_all()

        if patterns:
            print(f"\n  Found {len(patterns)} patterns:\n")
            for i, p in enumerate(patterns, 1):
                print(f"  {i}. {p.ticker}: ${p.resistance_level:.2f} "
                      f"({p.confidence:.1%} confidence)")
        else:
            print("\n  No patterns found.")

        print("\n" + "=" * 70)
        return

    # Run continuous loop
    print("\n  Starting continuous scanning...\n")
    try:
        await scanner.run()
    except KeyboardInterrupt:
        print("\n  Scanner stopped by user.")
        scanner.stop()

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_realtime_scanner())
