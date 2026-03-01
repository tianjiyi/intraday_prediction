"""
Scanner CLI Entry Point.

Main command-line interface for running premarket and real-time scanners.

Usage:
    # Run premarket scanner only
    python -m pattern_recognition.scanner.run_scanner --mode premarket

    # Run real-time scanner only
    python -m pattern_recognition.scanner.run_scanner --mode realtime

    # Run both (full day automation)
    python -m pattern_recognition.scanner.run_scanner --mode both

    # Initialize database tables
    python -m pattern_recognition.scanner.run_scanner --init-db

    # Show watchlist
    python -m pattern_recognition.scanner.run_scanner --show-watchlist
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, date
from typing import Optional
import pytz

from .config import SCANNER_CONFIG, load_config_from_yaml
from .db_models import init_db, get_engine
from .watchlist_manager import WatchlistManager
from .premarket_scanner import PremarketScanner, run_premarket_scanner
from .realtime_scanner import RealtimePatternScanner, run_realtime_scanner


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def init_database():
    """Initialize database tables."""
    print("Initializing database tables...")

    try:
        init_db()
        print("  ✓ Database tables created successfully")

        # Initialize default watchlist
        manager = WatchlistManager()
        count = manager.init_default_watchlist()
        print(f"  ✓ Default watchlist initialized ({count} symbols added)")

        print("\nDatabase initialization complete!")
        return True

    except Exception as e:
        print(f"  X Error: {e}")
        return False


def refresh_fundamentals(symbols: list = None, source: str = 'polygon'):
    """Refresh stock fundamentals from Polygon.io or Yahoo Finance."""
    from .fundamentals_service import FundamentalsService

    print("\n" + "=" * 60)
    print("  REFRESH STOCK FUNDAMENTALS")
    print("=" * 60)
    print(f"  Data Source: {source.upper()}")

    service = FundamentalsService()

    if symbols:
        print(f"  Symbols: {', '.join(symbols)}")
        results = service.refresh(symbols, source=source)
    else:
        print("  Refreshing ALL tradable stocks...")
        if source == 'polygon':
            print("  Estimated time: 10-20 minutes")
        else:
            print("  Estimated time: 30+ hours (Yahoo rate limits)")
        results = service.refresh_all(source=source)

    print(f"\n  Results:")
    print(f"    Updated: {results['success']}")
    print(f"    Failed:  {results['failed']}")
    print("=" * 60)

    return results['success'] > 0


def show_watchlist(target_date: date = None):
    """Display current watchlist."""
    manager = WatchlistManager()

    print("\n" + "=" * 60)
    print("  WATCHLIST OVERVIEW")
    print("=" * 60)

    # Default watchlist
    default = manager.get_default_watchlist()
    print(f"\n  Default Watchlist ({len(default)} symbols):")
    print(f"    {', '.join(default)}")

    # Floating watchlist
    floating_details = manager.get_floating_watchlist_details(target_date)
    print(f"\n  Floating Watchlist ({len(floating_details)} symbols):")

    if floating_details:
        for entry in floating_details[:10]:
            gap = entry.get('gap_percent')
            vol = entry.get('volume_ratio')
            gap_str = f"{gap:+.1f}%" if gap else "N/A"
            vol_str = f"{vol:.1f}x" if vol else "N/A"
            print(f"    {entry['symbol']:6s} gap={gap_str:>8s}  vol={vol_str}")
        if len(floating_details) > 10:
            print(f"    ... and {len(floating_details) - 10} more")
    else:
        print("    (empty - run premarket scanner to populate)")

    # Combined
    combined = manager.get_combined_watchlist(target_date)
    print(f"\n  Combined Watchlist ({len(combined)} symbols):")
    print(f"    {', '.join(combined)}")

    print("\n" + "=" * 60)


def run_premarket(config: dict = None, target_date: date = None, symbols: list = None, scan_all: bool = False):
    """Run premarket scanner."""
    scanner = PremarketScanner(config, target_date=target_date)

    print("\n" + "=" * 60)
    print(f"  PREMARKET SCANNER - {scanner.target_date}")
    print("=" * 60)

    if symbols:
        print(f"  Symbols: {', '.join(symbols)}")

    if scan_all:
        print(f"  Mode: Scanning ALL tradable stocks")
        print(f"  Batch size: {scanner.batch_size}")
        print(f"  Batch delay: {scanner.batch_delay}s")

    if not scanner.is_premarket_hours():
        print("\n  NOTE: Outside premarket hours (4:00am - 9:30am ET)")
        print("  Running scan anyway...\n")

    movers = scanner.scan(symbols=symbols, scan_all=scan_all)

    if movers:
        print(f"\n  Found {len(movers)} unusual movers:\n")
        for i, mover in enumerate(movers[:15], 1):
            direction = "+" if mover.gap_percent > 0 else "-"
            print(
                f"  {i:2d}. {mover.symbol:6s}  {direction}{abs(mover.gap_percent):5.1f}%  "
                f"Vol: {mover.volume_ratio:4.1f}x  Reason: {mover.reason}"
            )
        if len(movers) > 15:
            print(f"\n  ... and {len(movers) - 15} more movers")
    else:
        print("\n  No unusual movers found meeting criteria:")
        print(f"    - Gap >= {scanner.min_gap_percent}%")
        print(f"    - Volume >= {scanner.min_volume_ratio}x average")
        print(f"    - Market cap >= ${scanner.min_market_cap/1e9:.1f}B")

    print("\n" + "=" * 60)
    return movers


async def run_realtime(config: dict = None):
    """Run real-time pattern scanner."""
    scanner = RealtimePatternScanner(config)

    print("\n" + "=" * 60)
    print("  REAL-TIME PATTERN SCANNER")
    print("=" * 60)
    print(f"  Timeframe:      {scanner.timeframe}")
    print(f"  Lookback:       {scanner.lookback_bars} bars")
    print(f"  Min confidence: {scanner.min_confidence:.0%}")
    print(f"  Scan interval:  {scanner.scan_interval}s")
    print("=" * 60)

    if not scanner.is_market_hours():
        print("\n  NOTE: Outside market hours (9:30am - 4:00pm ET)")
        print("  Running single scan...\n")

        patterns = scanner.scan_all()

        if patterns:
            print(f"\n  Found {len(patterns)} patterns:\n")
            for i, p in enumerate(patterns, 1):
                status = f"[{p.breakout_status.upper()}]" if p.breakout_status else ""
                print(
                    f"  {i}. {p.ticker:6s} R=${p.resistance_level:.2f}  "
                    f"Conf={p.confidence:.1%}  {status}"
                )
        else:
            print("\n  No patterns found.")

        print("\n" + "=" * 60)
        return patterns

    # Run continuous loop
    print("\n  Starting continuous scanning (Ctrl+C to stop)...\n")

    try:
        await scanner.run()
    except KeyboardInterrupt:
        scanner.stop()
        print("\n  Scanner stopped.")

    print("=" * 60)


async def run_both(config: dict = None):
    """
    Run full day automation (premarket + real-time).

    Sequence:
    1. Run premarket scanner to find unusual movers
    2. Wait for RTH start (9:30am ET)
    3. Run real-time scanner during RTH
    """
    tz = pytz.timezone('US/Eastern')

    print("\n" + "=" * 60)
    print("  FULL DAY SCANNER")
    print("=" * 60)

    # Phase 1: Premarket
    now = datetime.now(tz)

    if now.time() < datetime.strptime('09:30', '%H:%M').time():
        print("\n  PHASE 1: Premarket Scanner")
        print("  " + "-" * 56)
        run_premarket(config)
    else:
        print("\n  Skipping premarket (after 9:30am)")

    # Phase 2: Wait for RTH if needed
    now = datetime.now(tz)
    rth_start = now.replace(hour=9, minute=30, second=0, microsecond=0)

    if now.time() < datetime.strptime('09:30', '%H:%M').time():
        wait_seconds = (rth_start - now).total_seconds()
        if wait_seconds > 0:
            print(f"\n  Waiting for RTH start ({wait_seconds/60:.1f} minutes)...")
            await asyncio.sleep(wait_seconds)

    # Phase 3: Real-time scanning
    now = datetime.now(tz)

    if now.time() < datetime.strptime('16:00', '%H:%M').time():
        print("\n  PHASE 2: Real-time Pattern Scanner")
        print("  " + "-" * 56)
        await run_realtime(config)
    else:
        print("\n  Market closed. Full day scan complete.")

    print("\n" + "=" * 60)
    print("  FULL DAY SCAN COMPLETE")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pattern Scanner - Premarket & Real-time Pattern Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run premarket scanner (today)
    python -m pattern_recognition.scanner.run_scanner --mode premarket

    # Run premarket scanner for a specific date
    python -m pattern_recognition.scanner.run_scanner --mode premarket --date 2024-12-20

    # Run real-time pattern scanner
    python -m pattern_recognition.scanner.run_scanner --mode realtime

    # Run both (full day)
    python -m pattern_recognition.scanner.run_scanner --mode both

    # Initialize database
    python -m pattern_recognition.scanner.run_scanner --init-db

    # Show current watchlist
    python -m pattern_recognition.scanner.run_scanner --show-watchlist
        """
    )

    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument(
        '--mode', '-m',
        choices=['premarket', 'realtime', 'both'],
        help='Scanner mode to run'
    )

    db_group = parser.add_argument_group('Database Operations')
    db_group.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database tables'
    )
    db_group.add_argument(
        '--show-watchlist',
        action='store_true',
        help='Show current watchlist'
    )
    db_group.add_argument(
        '--refresh-fundamentals',
        action='store_true',
        help='Refresh stock fundamentals (market cap, shares outstanding)'
    )
    db_group.add_argument(
        '--source',
        choices=['polygon', 'yahoo'],
        default='polygon',
        help='Data source for fundamentals: polygon (default, faster) or yahoo (has sector/industry)'
    )

    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml file'
    )
    config_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    scan_group = parser.add_argument_group('Scan Options')
    scan_group.add_argument(
        '--date', '-d',
        type=str,
        help='Target date for premarket scan (YYYY-MM-DD). Default: today'
    )
    scan_group.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to scan (overrides watchlist)'
    )
    scan_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Scan all tradable stocks (ignores max_symbols limit)'
    )
    scan_group.add_argument(
        '--batch-size',
        type=int,
        help='Symbols per batch (default: 100)'
    )
    scan_group.add_argument(
        '--timeframe',
        choices=['1min', '5min', '15min', '30min'],
        help='Bar timeframe for real-time scanner'
    )
    scan_group.add_argument(
        '--lookback',
        type=int,
        help='Number of bars to analyze'
    )

    return parser.parse_args()


def main():
    """Main entry point for scanner CLI."""
    args = parse_args()
    setup_logging(args.verbose)

    # Load config
    config = load_config_from_yaml(args.config)

    # Parse target date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"  ERROR: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
            return 1

    # Override config with CLI args
    if args.timeframe:
        config['realtime']['timeframe'] = args.timeframe
    if args.lookback:
        config['realtime']['lookback_bars'] = args.lookback
    if hasattr(args, 'batch_size') and args.batch_size:
        config['premarket']['batch_size'] = args.batch_size

    # Print header
    print()
    print("=" * 60)
    print("   KRONOS PATTERN SCANNER")
    print("   Premarket + Real-time Ascending Triangle Detection")
    print("=" * 60)

    # Handle operations
    if args.init_db:
        success = init_database()
        return 0 if success else 1

    if args.show_watchlist:
        show_watchlist(target_date)
        return 0

    if args.refresh_fundamentals:
        success = refresh_fundamentals(args.symbols, source=args.source)
        return 0 if success else 1

    if args.mode == 'premarket':
        scan_all = getattr(args, 'all', False)
        run_premarket(config, target_date, symbols=args.symbols, scan_all=scan_all)
        return 0

    if args.mode == 'realtime':
        asyncio.run(run_realtime(config))
        return 0

    if args.mode == 'both':
        asyncio.run(run_both(config))
        return 0

    # No mode specified, show help
    print("\n  No mode specified. Use --help for options.\n")
    print("  Quick start:")
    print("    --init-db         Initialize database tables")
    print("    --show-watchlist  Show current watchlist")
    print("    --mode premarket  Run premarket scanner")
    print("    --mode realtime   Run real-time pattern scanner")
    print("    --mode both       Run full day automation")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
