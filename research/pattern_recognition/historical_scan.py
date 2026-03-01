"""
Historical Pattern Scan - Scan historical data for ascending triangle patterns.

Scans a sliding window through historical data to find all ascending triangle
patterns and their breakout outcomes.

Usage:
    python -m pattern_recognition.historical_scan --ticker QQQ --days 30 --timeframe 1min

Output:
    - Console summary of all patterns found
    - JSON file with detailed pattern data
    - Charts for each pattern detected
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_recognition.data_fetcher import fetch_bars, load_config, BARS_PER_DAY
from pattern_recognition.ascending_triangle import AscendingTriangleDetector
from pattern_recognition.chart_generator import PatternChartGenerator
from pattern_recognition.models import AscendingTrianglePattern
from pattern_recognition.coordinate_mapper import ChartCoordinateMapper, convert_detections_to_tradable


def split_by_trading_day(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame into separate DataFrames for each trading day.

    Args:
        df: DataFrame with 'timestamp' column

    Returns:
        Dict mapping date string (YYYY-MM-DD) to DataFrame for that day
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column for day splitting")

    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    days = {}
    for date, group in df.groupby('date'):
        days[str(date)] = group.drop(columns=['date']).reset_index(drop=True)

    return days


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Historical Pattern Scan - Find ascending triangles in historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan QQQ for 30 days of 1-min patterns
    python -m pattern_recognition.historical_scan --ticker QQQ --days 30

    # Scan with custom window size
    python -m pattern_recognition.historical_scan --ticker QQQ --days 30 --window-size 400

    # Scan SPY for 15 days
    python -m pattern_recognition.historical_scan --ticker SPY --days 15 --output ./spy_scan
        """
    )

    parser.add_argument(
        '--ticker', '-t',
        default='QQQ',
        help='Ticker symbol to scan (default: QQQ)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Number of trading days to scan (default: 30)'
    )
    parser.add_argument(
        '--timeframe', '-tf',
        default='1min',
        choices=['1min', '5min', '15min', '30min'],
        help='Bar timeframe (default: 1min)'
    )
    parser.add_argument(
        '--window-size', '-w',
        type=int,
        default=500,
        help='Bars per scan window (default: 500)'
    )
    parser.add_argument(
        '--step-size', '-s',
        type=int,
        default=100,
        help='Bars to advance between scans (default: 100)'
    )
    parser.add_argument(
        '--post-context',
        type=int,
        default=50,
        help='Bars after pattern for breakout analysis (default: 50)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./historical_scan_output',
        help='Output directory (default: ./historical_scan_output)'
    )
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip chart generation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml file'
    )
    parser.add_argument(
        '--intraday-only',
        action='store_true',
        help='Only detect patterns within a single trading day (no cross-day patterns)'
    )

    # YOLO detection arguments
    parser.add_argument(
        '--yolo-detect',
        action='store_true',
        help='Run YOLOv8 pattern detection on daily chart images'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default=None,
        help='Path to YOLO model (default: download from HuggingFace)'
    )
    parser.add_argument(
        '--yolo-conf',
        type=float,
        default=0.25,
        help='YOLO confidence threshold (default: 0.25)'
    )

    return parser.parse_args()


def is_duplicate_pattern(
    new_global_end_idx: int,
    new_resistance: float,
    existing_patterns: List[Tuple[AscendingTrianglePattern, int]],
    time_tolerance_bars: int = 100,
    price_tolerance: float = 0.015
) -> bool:
    """
    Check if a pattern is a duplicate of an existing one.

    Patterns are considered duplicates if they have similar:
    - End time (within time_tolerance_bars)
    - Resistance level (within price_tolerance %)

    Args:
        new_global_end_idx: Global end index of new pattern
        new_resistance: Resistance level of new pattern
        existing_patterns: List of (pattern, global_end_index) tuples
        time_tolerance_bars: Maximum bar difference to consider duplicate
        price_tolerance: Maximum price difference ratio (0.015 = 1.5%)

    Returns:
        True if duplicate, False otherwise
    """
    for existing, global_end_idx in existing_patterns:
        # Check time proximity (using global index)
        if abs(global_end_idx - new_global_end_idx) < time_tolerance_bars:
            # Check price proximity
            price_diff = abs(existing.resistance_level - new_resistance)
            price_ratio = price_diff / existing.resistance_level
            if price_ratio < price_tolerance:
                return True
    return False


def scan_historical_data(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    window_size: int = 500,
    step_size: int = 100,
    post_context: int = 50,
    verbose: bool = False
) -> List[Tuple[AscendingTrianglePattern, pd.DataFrame]]:
    """
    Scan historical data with sliding window for ascending triangle patterns.

    Args:
        df: Full historical OHLCV DataFrame
        ticker: Ticker symbol
        timeframe: Timeframe string
        window_size: Number of bars per scan window
        step_size: Bars to advance between scans
        post_context: Bars after pattern for breakout analysis
        verbose: Enable verbose logging

    Returns:
        List of (pattern, window_df) tuples for each unique pattern found
    """
    logger = logging.getLogger(__name__)
    detector = AscendingTriangleDetector()

    patterns_found = []  # List of (pattern, global_end_index)
    pattern_data = []    # List of (pattern, window_df)

    total_bars = len(df)
    total_windows = (total_bars - window_size - post_context) // step_size + 1

    logger.info(f"Scanning {total_bars} bars with {total_windows} windows "
                f"(window={window_size}, step={step_size})")

    for i, start_idx in enumerate(range(0, total_bars - window_size - post_context, step_size)):
        end_idx = start_idx + window_size + post_context
        window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        if verbose and i % 20 == 0:
            progress = (i + 1) / total_windows * 100
            logger.debug(f"Progress: {progress:.1f}% ({i+1}/{total_windows} windows)")

        # Run pattern detection on window
        try:
            pattern = detector.detect(
                highs=window_df['high'].values,
                lows=window_df['low'].values,
                closes=window_df['close'].values,
                ticker=ticker,
                timeframe=timeframe,
                timestamps=window_df['timestamp'].values if 'timestamp' in window_df.columns else None,
                post_context_bars=post_context,
                analyze_breakout=True
            )

            if pattern:
                # Convert local end_index to global index for deduplication
                global_end_idx = start_idx + pattern.end_index

                # Check for duplicates BEFORE modifying pattern indices
                if not is_duplicate_pattern(global_end_idx, pattern.resistance_level, patterns_found):
                    # Get the window slice for charting (keep original indices for chart)
                    chart_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

                    # Store pattern with original local indices for charting
                    patterns_found.append((pattern, global_end_idx))
                    pattern_data.append((pattern, chart_df))

                    # Get timestamp for logging
                    if 'timestamp' in window_df.columns:
                        end_time = window_df['timestamp'].iloc[pattern.end_index]
                        logger.info(f"Pattern #{len(patterns_found)} found at {end_time} "
                                   f"(resistance=${pattern.resistance_level:.2f}, "
                                   f"breakout={pattern.breakout_status})")
                    else:
                        logger.info(f"Pattern #{len(patterns_found)} at bar {global_end_idx}")

        except Exception as e:
            if verbose:
                logger.debug(f"Window {i}: Detection error - {e}")
            continue

    logger.info(f"Scan complete: {len(patterns_found)} unique patterns found")
    return pattern_data


def scan_intraday(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    window_size: int = 390,
    step_size: int = 50,
    post_context: int = 50,
    verbose: bool = False
) -> List[Tuple[AscendingTrianglePattern, pd.DataFrame]]:
    """
    Scan historical data for intraday-only patterns (within single trading day).

    Splits data by trading day and scans each day separately, ensuring
    patterns don't span multiple days.

    Args:
        df: Full historical OHLCV DataFrame
        ticker: Ticker symbol
        timeframe: Timeframe string
        window_size: Number of bars per scan window (capped at 390 for intraday)
        step_size: Bars to advance between scans
        post_context: Bars after pattern for breakout analysis
        verbose: Enable verbose logging

    Returns:
        List of (pattern, window_df) tuples for each unique pattern found
    """
    logger = logging.getLogger(__name__)

    # Cap window size to fit within one trading day
    max_intraday_bars = 390  # 6.5 hours * 60 min
    if window_size > max_intraday_bars:
        logger.warning(f"Capping window_size from {window_size} to {max_intraday_bars} for intraday mode")
        window_size = max_intraday_bars

    # Split data by trading day
    daily_data = split_by_trading_day(df)
    logger.info(f"Split data into {len(daily_data)} trading days")

    all_pattern_data = []
    pattern_count = 0

    for date_str in sorted(daily_data.keys()):
        day_df = daily_data[date_str]

        # Skip days with insufficient data
        min_bars_needed = window_size + post_context
        if len(day_df) < min_bars_needed:
            if verbose:
                logger.debug(f"Skipping {date_str}: only {len(day_df)} bars (need {min_bars_needed})")
            continue

        logger.info(f"Scanning {date_str}: {len(day_df)} bars")

        # Scan this day's data
        day_patterns = scan_historical_data(
            df=day_df,
            ticker=ticker,
            timeframe=timeframe,
            window_size=window_size,
            step_size=step_size,
            post_context=post_context,
            verbose=verbose
        )

        # Add date context to patterns for proper naming
        for pattern, chart_df in day_patterns:
            pattern_count += 1
            all_pattern_data.append((pattern, chart_df))
            logger.info(f"  Pattern #{pattern_count} on {date_str}: "
                       f"resistance=${pattern.resistance_level:.2f}, "
                       f"breakout={pattern.breakout_status}")

    logger.info(f"Intraday scan complete: {len(all_pattern_data)} patterns across {len(daily_data)} days")
    return all_pattern_data


def generate_daily_charts_for_yolo(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    output_dir: Path,
    verbose: bool = False
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Generate full daily chart images for YOLO detection.

    Args:
        df: Full historical OHLCV DataFrame
        ticker: Ticker symbol
        timeframe: Timeframe string
        output_dir: Output directory for images
        verbose: Enable verbose logging

    Returns:
        List of (date_str, chart_path, mapper_dict) tuples
        mapper_dict contains coordinate mapping metadata for pixel-to-price conversion
    """
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    logger = logging.getLogger(__name__)

    # Split by trading day
    daily_data = split_by_trading_day(df)
    chart_paths = []

    # Custom style for clear YOLO detection
    mc = mpf.make_marketcolors(
        up='#26a69a',
        down='#ef5350',
        edge='inherit',
        wick='inherit',
        volume={'up': '#26a69a', 'down': '#ef5350'}
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridcolor='#e0e0e0',
        gridstyle='-',
        gridaxis='both',
        facecolor='white'
    )

    # DPI for chart generation (must match what we use for saving)
    chart_dpi = 100

    for date_str in sorted(daily_data.keys()):
        day_df = daily_data[date_str]

        # Skip days with too few bars
        if len(day_df) < 30:
            if verbose:
                logger.debug(f"Skipping {date_str}: only {len(day_df)} bars")
            continue

        # Prepare data for mplfinance
        chart_df = day_df.copy()
        chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
        chart_df.set_index('timestamp', inplace=True)

        # Rename columns for mplfinance
        chart_df = chart_df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Generate chart
        date_part = date_str.replace("-", "")
        chart_path = output_dir / f"{ticker}_{timeframe}_{date_part}_chart.png"

        try:
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=style,
                volume=True,
                title=f'{ticker} - {date_str} ({timeframe})',
                figsize=(12, 8),
                returnfig=True,
                tight_layout=True
            )

            # Create coordinate mapper BEFORE saving (captures exact coordinates)
            # Note: We use the chart_df which has timestamps as index
            mapper = ChartCoordinateMapper(
                df=chart_df,
                fig=fig,
                axes=axes,
                chart_path=str(chart_path),
                dpi=chart_dpi
            )
            mapper_dict = mapper.to_dict()

            # Save the chart
            fig.savefig(str(chart_path), dpi=chart_dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            chart_paths.append((date_str, str(chart_path), mapper_dict))
            if verbose:
                logger.debug(f"Generated chart: {chart_path}")

        except Exception as e:
            logger.error(f"Error generating chart for {date_str}: {e}")
            continue

    logger.info(f"Generated {len(chart_paths)} daily charts for YOLO")
    return chart_paths


def run_yolo_detection_on_charts(
    chart_paths: List[Tuple[str, str, Dict[str, Any]]],
    output_dir: Path,
    model_path: Optional[str] = None,
    confidence: float = 0.25,
    tradable_min_confidence: float = 0.40,
    tradable_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run YOLOv8 pattern detection on generated charts.

    Args:
        chart_paths: List of (date_str, chart_path, mapper_dict) tuples
        output_dir: Output directory for annotated images
        model_path: Path to YOLO model (None = download from HuggingFace)
        confidence: Detection confidence threshold
        tradable_min_confidence: Minimum confidence for tradable signals (default: 0.40)
        tradable_patterns: Pattern types to convert to tradable signals (default: ['W_Bottom'])

    Returns:
        Dictionary with detection results including tradable signals
    """
    logger = logging.getLogger(__name__)

    if tradable_patterns is None:
        tradable_patterns = ['W_Bottom']

    try:
        from pattern_recognition.yolo_inference import YOLOPatternDetector
    except ImportError as e:
        logger.error(f"YOLOv8 import error: {e}")
        print("ERROR: ultralytics package required. Install with: pip install ultralytics")
        return {}

    # Initialize detector
    print("\nInitializing YOLOv8 pattern detector...")
    detector = YOLOPatternDetector(
        model_path=model_path,
        confidence=confidence
    )

    results = {
        'detections': {},
        'stats': {
            'total_charts': len(chart_paths),
            'charts_with_patterns': 0,
            'total_detections': 0,
            'tradable_signals': 0,
            'class_counts': {}
        }
    }

    print(f"Running YOLO detection on {len(chart_paths)} charts (confidence={confidence})...")
    print(f"Tradable signals: {tradable_patterns} with min confidence {tradable_min_confidence:.0%}")
    print()

    for date_str, chart_path, mapper_dict in chart_paths:
        # Generate output path for annotated image
        input_path = Path(chart_path)
        yolo_path = input_path.parent / f"{input_path.stem}_yolo.png"

        # Run detection
        detections = detector.detect_and_save(chart_path, str(yolo_path))

        # Convert qualifying detections to tradable signals
        if detections and mapper_dict:
            detections = convert_detections_to_tradable(
                detections=detections,
                mapper_dict=mapper_dict,
                pattern_filter=tradable_patterns,
                min_confidence=tradable_min_confidence
            )

        results['detections'][date_str] = {
            'chart_path': chart_path,
            'yolo_path': str(yolo_path),
            'patterns': detections
        }

        if detections:
            results['stats']['charts_with_patterns'] += 1
            results['stats']['total_detections'] += len(detections)

            # Count by class and tradable signals
            for det in detections:
                class_name = det['class_name']
                results['stats']['class_counts'][class_name] = \
                    results['stats']['class_counts'].get(class_name, 0) + 1

                if 'tradable_signal' in det:
                    results['stats']['tradable_signals'] += 1

            # Print detection info
            patterns_str = ', '.join(
                f"{d['class_name']} ({d['confidence']:.1%})" +
                (f" [TRADABLE: ${d['tradable_signal']['range_high']:.2f}/${d['tradable_signal']['range_low']:.2f}]"
                 if 'tradable_signal' in d else "")
                for d in detections
            )
            print(f"  {date_str}: {patterns_str}")
            print(f"    -> {yolo_path}")
        else:
            print(f"  {date_str}: No patterns detected")

    print()
    return results


def print_yolo_summary(results: Dict[str, Any]):
    """Print YOLO detection summary."""
    stats = results.get('stats', {})

    print("\n" + "=" * 70)
    print("  YOLO Pattern Detection Results")
    print("=" * 70)
    print(f"  Total Charts:          {stats.get('total_charts', 0)}")
    print(f"  Charts with Patterns:  {stats.get('charts_with_patterns', 0)}")
    print(f"  Total Detections:      {stats.get('total_detections', 0)}")
    print(f"  Tradable Signals:      {stats.get('tradable_signals', 0)}")
    print()

    class_counts = stats.get('class_counts', {})
    if class_counts:
        print("  Patterns by Type:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {class_name}: {count}")

    # Print tradable signals summary
    tradable_count = stats.get('tradable_signals', 0)
    if tradable_count > 0:
        print()
        print("  Tradable Signals (W_Bottom >50% confidence):")
        for date_str, data in results.get('detections', {}).items():
            for det in data.get('patterns', []):
                if 'tradable_signal' in det:
                    sig = det['tradable_signal']
                    # Extract time portion (HH:MM) from timestamp string
                    time_start = sig['time_start'].split(' ')[1][:5] if ' ' in sig['time_start'] else sig['time_start'][-14:-9]
                    time_end = sig['time_end'].split(' ')[1][:5] if ' ' in sig['time_end'] else sig['time_end'][-14:-9]
                    print(f"    {date_str}: ${sig['range_high']:.2f} / ${sig['range_low']:.2f} "
                          f"({time_start} - {time_end}) "
                          f"[{det['confidence']:.1%}]")

    print("=" * 70)


def generate_daily_summary(
    patterns: List[Tuple[AscendingTrianglePattern, pd.DataFrame]]
) -> Dict[str, Dict[str, int]]:
    """
    Generate daily summary statistics.

    Returns:
        Dict mapping date string to {patterns, success, failure, pending, expired}
    """
    daily_stats = defaultdict(lambda: {
        'patterns': 0, 'success': 0, 'failure': 0, 'pending': 0, 'expired': 0
    })

    for pattern, df in patterns:
        # Get date from pattern end timestamp
        if pattern.end_timestamp:
            date_str = str(pattern.end_timestamp)[:10]
        elif 'timestamp' in df.columns and len(df) > 0:
            # Use timestamp from dataframe
            end_idx = min(pattern.end_index, len(df) - 1)
            date_str = str(df.iloc[end_idx]['timestamp'])[:10]
        else:
            date_str = "Unknown"

        daily_stats[date_str]['patterns'] += 1
        daily_stats[date_str][pattern.breakout_status] += 1

    return dict(daily_stats)


def print_results(
    patterns: List[Tuple[AscendingTrianglePattern, pd.DataFrame]],
    ticker: str,
    df: pd.DataFrame
):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print(f"  {ticker} Historical Pattern Scan Results")
    print("=" * 70)

    if 'timestamp' in df.columns:
        print(f"  Date Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Total Bars: {len(df)}")
    print(f"  Total Patterns Found: {len(patterns)}")
    print("=" * 70)

    if not patterns:
        print("\n  No ascending triangle patterns found.\n")
        return

    # Print each pattern
    print("\n--- Pattern Details ---\n")
    for i, (pattern, _) in enumerate(patterns, 1):
        timestamp = pattern.end_timestamp or f"Bar {pattern.end_index}"

        print(f"Pattern #{i}: {timestamp}")
        print(f"  Resistance: ${pattern.resistance_level:.2f}")
        print(f"  Confidence: {pattern.confidence:.1%}")
        print(f"  Breakout: {pattern.breakout_status.upper()}", end="")

        if pattern.breakout_status == "success" and pattern.breakout_price:
            gain = (pattern.breakout_price - pattern.resistance_level) / pattern.resistance_level
            print(f" (+{gain:.2%} at ${pattern.breakout_price:.2f})")
        elif pattern.breakout_status == "failure" and pattern.breakout_price:
            loss = (pattern.breakout_price - pattern.resistance_level) / pattern.resistance_level
            print(f" ({loss:.2%} at ${pattern.breakout_price:.2f})")
        else:
            print()

        if pattern.bars_to_breakout:
            print(f"  Bars to Breakout: {pattern.bars_to_breakout}")
        print()

    # Daily summary
    daily_stats = generate_daily_summary(patterns)

    print("--- Daily Summary ---\n")
    print(f"{'Date':<12} | {'Patterns':>8} | {'Success':>7} | {'Failure':>7} | {'Pending':>7}")
    print("-" * 55)

    for date in sorted(daily_stats.keys()):
        stats = daily_stats[date]
        print(f"{date:<12} | {stats['patterns']:>8} | {stats['success']:>7} | "
              f"{stats['failure']:>7} | {stats['pending']:>7}")

    # Overall stats
    print("\n--- Overall Statistics ---\n")

    success_count = sum(1 for p, _ in patterns if p.breakout_status == "success")
    failure_count = sum(1 for p, _ in patterns if p.breakout_status == "failure")
    resolved_count = success_count + failure_count

    print(f"  Total Patterns: {len(patterns)}")

    if resolved_count > 0:
        success_rate = success_count / resolved_count
        print(f"  Success Rate: {success_rate:.1%} ({success_count}/{resolved_count} resolved)")

    avg_confidence = sum(p.confidence for p, _ in patterns) / len(patterns)
    print(f"  Avg Confidence: {avg_confidence:.1%}")

    # Average bars to breakout for resolved patterns
    resolved_bars = [p.bars_to_breakout for p, _ in patterns if p.bars_to_breakout]
    if resolved_bars:
        avg_bars = sum(resolved_bars) / len(resolved_bars)
        print(f"  Avg Bars to Breakout: {avg_bars:.1f}")

    print()


def save_results_json(
    patterns: List[Tuple[AscendingTrianglePattern, pd.DataFrame]],
    output_dir: Path,
    ticker: str,
    timeframe: str,
    df: pd.DataFrame
):
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Build results dict
    results = {
        "scan_timestamp": timestamp,
        "ticker": ticker,
        "timeframe": timeframe,
        "total_bars": len(df),
        "date_range": {
            "start": str(df['timestamp'].iloc[0]) if 'timestamp' in df.columns else None,
            "end": str(df['timestamp'].iloc[-1]) if 'timestamp' in df.columns else None
        },
        "patterns": [p.to_dict() for p, _ in patterns],
        "daily_summary": generate_daily_summary(patterns),
        "overall_stats": {
            "total_patterns": len(patterns),
            "success_count": sum(1 for p, _ in patterns if p.breakout_status == "success"),
            "failure_count": sum(1 for p, _ in patterns if p.breakout_status == "failure"),
            "pending_count": sum(1 for p, _ in patterns if p.breakout_status == "pending"),
            "expired_count": sum(1 for p, _ in patterns if p.breakout_status == "expired"),
            "avg_confidence": sum(p.confidence for p, _ in patterns) / len(patterns) if patterns else 0
        }
    }

    # Calculate success rate
    resolved = results["overall_stats"]["success_count"] + results["overall_stats"]["failure_count"]
    if resolved > 0:
        results["overall_stats"]["success_rate"] = results["overall_stats"]["success_count"] / resolved

    # Save
    json_path = output_dir / f"{ticker}_scan_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {json_path}")
    return json_path


def generate_charts(
    patterns: List[Tuple[AscendingTrianglePattern, pd.DataFrame]],
    output_dir: Path,
    ticker: str,
    timeframe: str
):
    """Generate charts for each pattern."""
    import matplotlib.pyplot as plt

    if not patterns:
        return

    generator = PatternChartGenerator()

    print(f"\nGenerating {len(patterns)} charts...")

    for i, (pattern, df) in enumerate(patterns, 1):
        # Create unique filename with pattern date
        if pattern.end_timestamp:
            date_part = str(pattern.end_timestamp)[:10].replace("-", "")
        else:
            date_part = f"bar{pattern.end_index}"

        chart_path = output_dir / f"{ticker}_{timeframe}_{date_part}_pattern{i}.png"

        try:
            generator.generate_annotated_chart(
                df, pattern, str(chart_path),
                show_full_range=True  # Show full window for context
            )
            print(f"  Saved: {chart_path}")
        except Exception as e:
            print(f"  Error generating chart {i}: {e}")
        finally:
            # Close all figures to avoid memory warning
            plt.close('all')


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print header
    print("\n" + "=" * 70)
    print("  Historical Pattern Scan")
    print("  Ascending Triangle Detection with Sliding Window")
    print("=" * 70)
    print(f"  Ticker:      {args.ticker}")
    print(f"  Days:        {args.days} trading days")
    print(f"  Timeframe:   {args.timeframe}")
    print(f"  Window:      {args.window_size} bars")
    print(f"  Step:        {args.step_size} bars")
    print(f"  Mode:        {'INTRADAY ONLY' if args.intraday_only else 'Cross-day allowed'}")
    print("=" * 70)
    print()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print("ERROR: Could not load configuration.")
        sys.exit(1)

    # Calculate lookback bars
    bars_per_day = BARS_PER_DAY.get(args.timeframe, 390)
    lookback_bars = args.days * bars_per_day

    print(f"Fetching {lookback_bars} bars ({args.days} days × {bars_per_day} bars/day)...")

    # Fetch data
    try:
        data = fetch_bars(
            tickers=[args.ticker],
            lookback_bars=lookback_bars,
            timeframe=args.timeframe,
            config=config,
            rth_only=True
        )
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.ticker not in data:
        print(f"ERROR: No data returned for {args.ticker}")
        sys.exit(1)

    df = data[args.ticker]
    print(f"Fetched {len(df)} bars\n")

    # Scan for patterns
    print("Scanning for ascending triangle patterns...")

    if args.intraday_only:
        # Scan each day separately (no cross-day patterns)
        pattern_data = scan_intraday(
            df=df,
            ticker=args.ticker,
            timeframe=args.timeframe,
            window_size=args.window_size,
            step_size=args.step_size,
            post_context=args.post_context,
            verbose=args.verbose
        )
    else:
        # Original behavior - sliding window across all data
        pattern_data = scan_historical_data(
            df=df,
            ticker=args.ticker,
            timeframe=args.timeframe,
            window_size=args.window_size,
            step_size=args.step_size,
            post_context=args.post_context,
            verbose=args.verbose
        )

    # Print results
    print_results(pattern_data, args.ticker, df)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    if pattern_data:
        save_results_json(pattern_data, output_dir, args.ticker, args.timeframe, df)

    # Generate charts
    if not args.no_charts and pattern_data:
        generate_charts(pattern_data, output_dir, args.ticker, args.timeframe)

    # Run YOLO detection if requested
    if args.yolo_detect:
        print("\n" + "-" * 70)
        print("  YOLO Pattern Detection")
        print("-" * 70)

        # Generate daily charts for YOLO
        print(f"\nGenerating daily charts for YOLO detection...")
        chart_paths = generate_daily_charts_for_yolo(
            df=df,
            ticker=args.ticker,
            timeframe=args.timeframe,
            output_dir=output_dir,
            verbose=args.verbose
        )

        if chart_paths:
            # Run YOLO detection
            yolo_results = run_yolo_detection_on_charts(
                chart_paths=chart_paths,
                output_dir=output_dir,
                model_path=args.yolo_model,
                confidence=args.yolo_conf
            )

            # Print summary
            if yolo_results:
                print_yolo_summary(yolo_results)

                # Save YOLO results to JSON
                yolo_json_path = output_dir / f"{args.ticker}_yolo_detections.json"
                with open(yolo_json_path, 'w') as f:
                    json.dump(yolo_results, f, indent=2, default=str)
                print(f"\nYOLO results saved to: {yolo_json_path}")
        else:
            print("No daily charts generated for YOLO detection.")

    print("\n" + "=" * 70)
    print("  Scan complete!")
    print("=" * 70 + "\n")

    return 0 if pattern_data else 1


if __name__ == "__main__":
    sys.exit(main())
