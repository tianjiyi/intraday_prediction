"""
CLI interface for ascending triangle pattern recognition.

Usage:
    # Scan single ticker
    python -m pattern_recognition.cli --tickers QQQ --timeframe 5min

    # Scan multiple tickers with charts
    python -m pattern_recognition.cli --tickers QQQ SPY AAPL --timeframe 15min --chart

    # Generate YOLO training data
    python -m pattern_recognition.cli --tickers QQQ --yolo --yolo-dir ./yolo_dataset

    # Full scan with all outputs
    python -m pattern_recognition.cli --tickers QQQ SPY AAPL TSLA NVDA \\
        --timeframe 5min --lookback 300 --chart --json --yolo
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_recognition.data_fetcher import fetch_bars, load_config, get_supported_timeframes
from pattern_recognition.ascending_triangle import AscendingTriangleDetector, get_pattern_summary
from pattern_recognition.chart_generator import PatternChartGenerator
from pattern_recognition.yolo_exporter import (
    export_yolo_annotation, create_yolo_dataset_structure, generate_data_yaml
)
from pattern_recognition.models import DataFetchError


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_symbols_by_market_cap(min_market_cap: float = 1_000_000_000) -> List[str]:
    """
    Get all symbols with market cap >= threshold from database.

    Args:
        min_market_cap: Minimum market cap in dollars (default: $1B)

    Returns:
        List of ticker symbols
    """
    try:
        from pattern_recognition.scanner.db_models import StockFundamentals, get_session

        session = get_session()
        rows = session.query(StockFundamentals.symbol).filter(
            StockFundamentals.market_cap >= min_market_cap
        ).order_by(StockFundamentals.market_cap.desc()).all()
        session.close()

        symbols = [row.symbol for row in rows]
        return symbols
    except Exception as e:
        print(f"ERROR: Failed to query database: {e}")
        print("Make sure the database is running and stock_fundamentals table is populated.")
        print("Run: python -m pattern_recognition.scanner.run_scanner --refresh-fundamentals")
        return []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ascending Triangle Pattern Recognizer - Scan stocks for ascending triangle patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic scan
    python -m pattern_recognition.cli --tickers QQQ SPY

    # With specific timeframe and lookback
    python -m pattern_recognition.cli --tickers QQQ --timeframe 15min --lookback 300

    # Generate annotated charts
    python -m pattern_recognition.cli --tickers QQQ SPY AAPL --chart --output ./charts

    # Generate YOLO training data
    python -m pattern_recognition.cli --tickers QQQ --yolo --yolo-dir ./yolo_data

    # Full output (JSON + charts + YOLO)
    python -m pattern_recognition.cli --tickers QQQ SPY --json --chart --yolo
        """
    )

    # Input parameters
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--tickers', '-t',
        nargs='+',
        help='List of ticker symbols (e.g., QQQ SPY AAPL TSLA)'
    )
    input_group.add_argument(
        '--scan-all',
        action='store_true',
        help='Scan all stocks with market cap >= threshold (queries database)'
    )
    input_group.add_argument(
        '--min-mcap',
        type=float,
        default=1_000_000_000,
        help='Minimum market cap for --scan-all in dollars (default: 1B)'
    )
    input_group.add_argument(
        '--timeframe', '-tf',
        choices=get_supported_timeframes(),
        default='5min',
        help='Bar timeframe (default: 5min)'
    )
    input_group.add_argument(
        '--lookback', '-l',
        type=int,
        default=200,
        help='Number of bars to analyze (default: 200)'
    )
    input_group.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml file'
    )

    # Detection parameters
    detect_group = parser.add_argument_group('Detection Parameters')
    detect_group.add_argument(
        '--resistance-tolerance',
        type=float,
        default=0.015,
        help='Tolerance for flat resistance (default: auto based on timeframe, 0.5%% for 1min to 1.5%% for daily)'
    )
    detect_group.add_argument(
        '--min-support-slope',
        type=float,
        default=0.00001,
        help='Minimum support slope (default: 0.00001)'
    )
    detect_group.add_argument(
        '--zigzag-deviation',
        type=float,
        default=None,
        help='ZigZag deviation (default: auto based on timeframe)'
    )
    detect_group.add_argument(
        '--min-bars',
        type=int,
        default=None,
        help='Minimum bars for pattern (default: auto based on timeframe)'
    )

    # Context and breakout parameters
    context_group = parser.add_argument_group('Context & Breakout Options')
    context_group.add_argument(
        '--pre-context',
        type=int,
        default=20,
        help='Number of bars before pattern for context (default: 20)'
    )
    context_group.add_argument(
        '--post-context',
        type=int,
        default=20,
        help='Number of bars after pattern for breakout analysis (default: 20)'
    )
    context_group.add_argument(
        '--no-breakout-analysis',
        action='store_true',
        help='Skip breakout detection (just detect patterns)'
    )

    # Output parameters
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output', '-o',
        default='./pattern_output',
        help='Output directory (default: ./pattern_output)'
    )
    output_group.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON file'
    )
    output_group.add_argument(
        '--chart',
        action='store_true',
        help='Generate PNG chart with annotations'
    )
    output_group.add_argument(
        '--yolo',
        action='store_true',
        help='Generate YOLO training format'
    )
    output_group.add_argument(
        '--yolo-dir',
        default='./yolo_dataset',
        help='YOLO dataset output directory (default: ./yolo_dataset)'
    )
    output_group.add_argument(
        '--full-chart',
        action='store_true',
        help='Plot all lookback bars instead of just pattern area'
    )

    # Other options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    other_group.add_argument(
        '--no-rth',
        action='store_true',
        help='Include extended hours data (default: RTH only)'
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Determine tickers to scan
    if args.scan_all:
        tickers = get_symbols_by_market_cap(args.min_mcap)
        if not tickers:
            print("ERROR: No symbols found. Make sure database is populated.")
            sys.exit(1)
        mcap_display = f"${args.min_mcap/1e9:.1f}B" if args.min_mcap >= 1e9 else f"${args.min_mcap/1e6:.0f}M"
        ticker_display = f"{len(tickers)} stocks (market cap >= {mcap_display})"
    elif args.tickers:
        tickers = args.tickers
        ticker_display = ', '.join(tickers)
    else:
        print("ERROR: Either --tickers or --scan-all is required")
        sys.exit(1)

    # Print header
    print("=" * 70)
    print("  Ascending Triangle Pattern Recognizer")
    print("  Based on ZigZag algorithm with multi-timeframe support")
    print("=" * 70)
    print(f"  Tickers:    {ticker_display}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Lookback:   {args.lookback} bars")
    print(f"  Output:     {args.output}")
    print("=" * 70)
    print()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print("\nERROR: Could not load configuration.")
        print("Make sure config.yaml exists with ALPACA_KEY_ID and ALPACA_SECRET_KEY")
        sys.exit(1)

    # Fetch data
    print("Fetching market data from Alpaca...")
    try:
        data_dict = fetch_bars(
            tickers=tickers,
            lookback_bars=args.lookback,
            timeframe=args.timeframe,
            config=config,
            rth_only=not args.no_rth
        )
    except DataFetchError as e:
        logger.error(f"Data fetch error: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error fetching data: {e}")
        print(f"\nERROR: Failed to fetch data - {e}")
        sys.exit(1)

    if not data_dict:
        print("\nNo data fetched. Check your API keys and ticker symbols.")
        sys.exit(1)

    print(f"Fetched data for {len(data_dict)} tickers\n")

    # Initialize detector
    detector = AscendingTriangleDetector(
        resistance_tolerance=args.resistance_tolerance,
        min_support_slope=args.min_support_slope,
        zigzag_deviation=args.zigzag_deviation,
        min_bars=args.min_bars
    )

    # Scan for patterns
    print("Scanning for ascending triangle patterns...")
    print("-" * 50)

    patterns = []
    pattern_data = []  # Store (pattern, dataframe) pairs

    for ticker, df in data_dict.items():
        try:
            pattern = detector.detect(
                highs=df['high'].values,
                lows=df['low'].values,
                closes=df['close'].values,
                ticker=ticker,
                timeframe=args.timeframe,
                timestamps=df['timestamp'].values if 'timestamp' in df.columns else None,
                pre_context_bars=args.pre_context,
                post_context_bars=args.post_context,
                analyze_breakout=not args.no_breakout_analysis
            )

            if pattern:
                patterns.append(pattern)
                pattern_data.append((pattern, df))
                # Include breakout status in output
                breakout_info = f"[{pattern.breakout_status.upper()}]"
                if pattern.bars_to_breakout:
                    breakout_info += f" ({pattern.bars_to_breakout} bars)"
                print(f"  [FOUND] {ticker}: resistance=${pattern.resistance_level:.2f}, "
                      f"confidence={pattern.confidence:.1%}, breakout={breakout_info}")
            else:
                print(f"  [----]  {ticker}: No pattern detected")

        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
            print(f"  [ERROR] {ticker}: {e}")

    print("-" * 50)
    print()

    # Summary
    if patterns:
        summary = get_pattern_summary(patterns)
        print(f"Found {summary['count']} ascending triangle pattern(s)")
        print(f"  Average confidence: {summary['avg_confidence']:.1%}")
        print(f"  Highest confidence: {summary['highest_confidence_ticker']} "
              f"({summary['max_confidence']:.1%})")
        # Breakout statistics
        if 'breakout_stats' in summary:
            bs = summary['breakout_stats']
            print(f"  Breakout stats: {bs['success']} success, {bs['failure']} failure, "
                  f"{bs['pending']} pending, {bs['expired']} expired")
            if summary.get('success_rate') is not None:
                print(f"  Success rate: {summary['success_rate']:.1%}")
        print()
    else:
        print("No ascending triangle patterns found in the scanned tickers.")
        print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Generate outputs
    if patterns:
        # JSON output
        if args.json:
            print("Generating JSON output...")
            json_data = {
                "timestamp": timestamp,
                "timeframe": args.timeframe,
                "lookback_bars": args.lookback,
                "pre_context_bars": args.pre_context,
                "post_context_bars": args.post_context,
                "patterns": [p.to_dict() for p in patterns],
                "summary": get_pattern_summary(patterns)
            }
            json_path = output_dir / f"patterns_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"  Saved: {json_path}")

        # Chart output
        if args.chart:
            print("Generating annotated charts...")
            generator = PatternChartGenerator()
            for pattern, df in pattern_data:
                chart_path = output_dir / f"{pattern.ticker}_{args.timeframe}_{timestamp}.png"
                generator.generate_annotated_chart(
                    df, pattern, str(chart_path),
                    show_full_range=args.full_chart
                )
                print(f"  Saved: {chart_path}")

        # YOLO output
        if args.yolo:
            print("Generating YOLO training data...")
            yolo_dir = Path(args.yolo_dir)
            dirs = create_yolo_dataset_structure(str(yolo_dir))

            generator = PatternChartGenerator()
            for pattern, df in pattern_data:
                # Generate image
                img_filename = f"{pattern.ticker}_{timestamp}.png"
                img_path = dirs['images_train'] / img_filename
                generator.generate_yolo_training_image(df, pattern, str(img_path))

                # Generate label
                label_filename = f"{pattern.ticker}_{timestamp}.txt"
                label_path = dirs['labels_train'] / label_filename
                export_yolo_annotation(pattern, str(label_path))

                print(f"  Saved: {img_path}")

            # Generate data.yaml
            generate_data_yaml(str(yolo_dir))
            print(f"  Saved: {yolo_dir}/data.yaml")

    print()
    print("=" * 70)
    print("  Scan complete!")
    print("=" * 70)

    return 0 if patterns else 1


if __name__ == "__main__":
    sys.exit(main())
