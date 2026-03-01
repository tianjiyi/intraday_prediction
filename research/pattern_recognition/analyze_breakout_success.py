"""
Analyze W_Bottom patterns for successful breakouts.
Generates visualization images for manual verification.

Success Definition:
- After W_Bottom bbox range ends, within up to 120 bars (or whatever remains until market close),
  if ANY bar's HIGH reaches the breakout target (range_high + range_width), it's a SUCCESS.
"""
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Constants
CONTEXT_BARS = 150       # Bars before pattern
POST_BARS = 120          # Max bars after pattern to check
SUCCESS_MULTIPLIER = 1.0 # Range extension for success (1x = same as range width)
IMAGE_SIZE = 640         # Original chart pixel size

# Directories
BASE_DIR = Path(__file__).parent.parent
SCAN_DIR = BASE_DIR / "yolo_w_bottom_scan"
DATA_DIR = BASE_DIR / "historical_data"
OUTPUT_DIR = BASE_DIR / "breakout_analysis"
SUCCESS_DIR = OUTPUT_DIR / "successful_breakouts"
FAILED_DIR = OUTPUT_DIR / "failed_breakouts"


def bbox_to_bar_indices(bbox_str: str, num_bars: int) -> tuple:
    """Convert bbox x-coords to bar indices."""
    bbox = ast.literal_eval(bbox_str)
    x1, y1, x2, y2 = bbox
    pattern_start = int(x1 / IMAGE_SIZE * num_bars)
    pattern_end = int(x2 / IMAGE_SIZE * num_bars)
    return pattern_start, pattern_end


def bbox_to_price_range(bbox_str: str, price_min: float, price_max: float) -> tuple:
    """Convert bbox y-coords to price levels."""
    bbox = ast.literal_eval(bbox_str)
    x1, y1, x2, y2 = bbox
    # Y is inverted: y=0 is top (highest price), y=640 is bottom (lowest price)
    range_high = price_max - (y1 / IMAGE_SIZE) * (price_max - price_min)
    range_low = price_max - (y2 / IMAGE_SIZE) * (price_max - price_min)
    return range_high, range_low


def check_breakout_success(df: pd.DataFrame, pattern_end: int,
                          range_high: float, range_width: float) -> tuple:
    """
    Check if price reaches breakout target within available post-pattern bars.

    Returns:
        (success, breakout_target, post_bars_count, max_high_reached)
    """
    breakout_target = range_high + (range_width * SUCCESS_MULTIPLIER)

    # Get post-pattern bars (up to POST_BARS or whatever remains)
    post_start = pattern_end + 1
    post_end = min(post_start + POST_BARS, len(df))
    post_df = df.iloc[post_start:post_end]

    # Check if any high reaches target
    if len(post_df) == 0:
        return False, breakout_target, 0, 0.0

    max_high = post_df['high'].max()
    success = max_high >= breakout_target

    return success, breakout_target, len(post_df), max_high


def generate_breakout_chart(df: pd.DataFrame, pattern_start: int, pattern_end: int,
                           range_high: float, range_low: float,
                           breakout_target: float, success: bool,
                           ticker: str, date: str, post_bars: int,
                           max_high: float, output_path: Path):
    """Generate visualization chart with pattern bbox and levels."""
    # Calculate chart bounds
    context_start = max(0, pattern_start - CONTEXT_BARS)
    chart_end = min(len(df), pattern_end + post_bars + 1)

    chart_df = df.iloc[context_start:chart_end].copy()

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot candlesticks manually
    for i, (idx, row) in enumerate(chart_df.iterrows()):
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Body
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        if body_height < 0.001:  # Doji - make visible
            body_height = 0.001
        ax.add_patch(Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor=color))
        # Wicks
        ax.plot([i, i], [row['low'], body_bottom], color=color, linewidth=0.5)
        ax.plot([i, i], [body_bottom + body_height, row['high']], color=color, linewidth=0.5)

    # Draw pattern bbox (yellow shaded area)
    pattern_start_rel = pattern_start - context_start
    pattern_end_rel = pattern_end - context_start
    ax.axvspan(pattern_start_rel, pattern_end_rel, alpha=0.2, color='yellow', label='W_Bottom Pattern')

    # Draw horizontal lines
    ax.axhline(y=range_high, color='blue', linestyle='-', linewidth=1.5, label=f'Range High: ${range_high:.2f}')
    ax.axhline(y=range_low, color='blue', linestyle='--', linewidth=1.5, label=f'Range Low: ${range_low:.2f}')
    ax.axhline(y=breakout_target, color='green' if success else 'red',
               linestyle='--', linewidth=2, label=f'Target: ${breakout_target:.2f}')

    # Vertical line at pattern end
    ax.axvline(x=pattern_end_rel, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Pattern End')

    # Set y-axis limits with padding
    all_lows = chart_df['low'].min()
    all_highs = chart_df['high'].max()
    y_range = all_highs - all_lows
    ax.set_ylim(all_lows - y_range * 0.05, all_highs + y_range * 0.1)

    # Labels and title
    status = "SUCCESS" if success else "FAILED"
    range_width = range_high - range_low
    ax.set_title(f"{ticker} - {date} - {status}\n"
                f"Range: ${range_low:.2f} to ${range_high:.2f} (width: ${range_width:.2f}) | "
                f"Target: ${breakout_target:.2f} | Max High: ${max_high:.2f} | Post Bars: {post_bars}",
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Bars')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(-1, len(chart_df))
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main(tickers_to_run=None, skip_images=False):
    """Main analysis function."""
    print("=" * 60)
    print("W_Bottom Breakout Analysis")
    print("=" * 60)

    # Create output directories
    SUCCESS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    tickers = tickers_to_run if tickers_to_run else ['QQQ', 'IWM', 'NVDA', 'TSLA']
    results = []

    if skip_images:
        print("Mode: CSV only (skipping image generation)")

    for ticker in tickers:
        print(f"\n=== Analyzing {ticker} ===")

        # Load historical data (pick file with widest date range)
        data_files = list(DATA_DIR.glob(f"{ticker}_*_1min_rth.csv"))
        if not data_files:
            print(f"  No data file found for {ticker}, skipping...")
            continue

        # Sort by filename to pick widest range (e.g., 2020_2025 over 2020_2020)
        def get_date_range(f):
            # Extract year range from filename like "QQQ_2020_2025_1min_rth.csv"
            parts = f.stem.split('_')
            if len(parts) >= 3:
                try:
                    start_year = int(parts[1])
                    end_year = int(parts[2])
                    return end_year - start_year
                except ValueError:
                    return 0
            return 0

        data_file = max(data_files, key=get_date_range)
        print(f"  Loading: {data_file.name}")
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        # Load W_Bottom results
        for year in range(2020, 2026):
            csv_file = SCAN_DIR / f"{ticker}_{year}_w_bottom_results.csv"
            if not csv_file.exists():
                continue

            w_bottoms = pd.read_csv(csv_file)
            print(f"  {year}: {len(w_bottoms)} patterns")

            for _, row in w_bottoms.iterrows():
                date = pd.to_datetime(row['date']).date()
                day_df = df[df['date'] == date].reset_index(drop=True)

                if len(day_df) < 100:
                    continue

                num_bars = len(day_df)
                price_min = day_df['low'].min()
                price_max = day_df['high'].max()

                # Convert bbox
                pattern_start, pattern_end = bbox_to_bar_indices(row['bbox'], num_bars)
                range_high, range_low = bbox_to_price_range(row['bbox'], price_min, price_max)
                range_width = range_high - range_low

                # Skip if range is too small (likely bad detection)
                if range_width < 0.01:
                    continue

                # Check success
                success, breakout_target, post_bars, max_high = check_breakout_success(
                    day_df, pattern_end, range_high, range_width
                )

                # Generate chart (skip if skip_images=True)
                output_dir = SUCCESS_DIR if success else FAILED_DIR
                output_file = output_dir / f"{ticker}_{date}_{row['confidence']:.2f}.png"

                if not skip_images:
                    generate_breakout_chart(
                        day_df, pattern_start, pattern_end,
                        range_high, range_low, breakout_target, success,
                        ticker, str(date), post_bars, max_high, output_file
                    )

                results.append({
                    'ticker': ticker,
                    'date': str(date),
                    'confidence': row['confidence'],
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_width': range_width,
                    'breakout_target': breakout_target,
                    'max_high': max_high,
                    'success': success,
                    'post_bars': post_bars
                })

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "breakout_analysis_results.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total patterns analyzed: {len(results_df)}")
    print(f"Successful breakouts: {results_df['success'].sum()} ({results_df['success'].mean()*100:.1f}%)")
    print(f"Failed breakouts: {(~results_df['success']).sum()} ({(~results_df['success']).mean()*100:.1f}%)")
    print(f"\nImages saved to:")
    print(f"  - Successful: {SUCCESS_DIR}")
    print(f"  - Failed: {FAILED_DIR}")
    print(f"  - Results CSV: {OUTPUT_DIR / 'breakout_analysis_results.csv'}")

    # Per-ticker breakdown
    print("\nPer-ticker breakdown:")
    for ticker in tickers:
        ticker_df = results_df[results_df['ticker'] == ticker]
        if len(ticker_df) > 0:
            success_rate = ticker_df['success'].mean() * 100
            print(f"  {ticker}: {len(ticker_df)} patterns, {ticker_df['success'].sum()} successful ({success_rate:.1f}%)")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Analyze W_Bottom breakout success")
    parser.add_argument('tickers', nargs='*', help='Tickers to analyze (default: QQQ IWM NVDA TSLA)')
    parser.add_argument('--skip-images', action='store_true', help='Skip image generation, only create CSV')
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else None
    main(tickers_to_run=tickers, skip_images=args.skip_images)
