"""
Scan trading days with pre-trained YOLO to find W_Bottom patterns.
Uses locally downloaded data (no API calls).

Usage:
    python -m pattern_recognition.scan_for_w_bottoms --year 2025
    python -m pattern_recognition.scan_for_w_bottoms --year 2024 --ticker QQQ
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_recognition.yolo_inference import YOLOPatternDetector

# Configuration
DATA_DIR = Path(__file__).parent.parent / "historical_data"
OUTPUT_DIR = Path(__file__).parent.parent / "yolo_w_bottom_scan"
IMAGE_SIZE = 640


def load_local_data(ticker: str) -> pd.DataFrame:
    """Load locally downloaded historical data."""

    def get_date_range(filename):
        """Extract year range from filename like QQQ_2020_2025_1min_rth.parquet"""
        parts = filename.stem.split('_')
        try:
            start_year = int(parts[1])
            end_year = int(parts[2])
            return end_year - start_year  # Return range width
        except (IndexError, ValueError):
            return 0

    # Try parquet first (faster)
    parquet_files = list(DATA_DIR.glob(f"{ticker}_*_1min_rth.parquet"))
    if parquet_files:
        # Pick file with widest date range
        file = max(parquet_files, key=get_date_range)
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    # Try CSV
    csv_files = list(DATA_DIR.glob(f"{ticker}_*_1min_rth.csv"))
    if csv_files:
        file = max(csv_files, key=get_date_range)
        print(f"Loading: {file.name}")
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    raise FileNotFoundError(f"No data file found for {ticker} in {DATA_DIR}")


def get_day_data(df: pd.DataFrame, target_date) -> pd.DataFrame:
    """Extract data for a specific date."""
    df_day = df[df['timestamp'].dt.date == target_date].copy()
    return df_day.reset_index(drop=True)


def generate_chart_image(df: pd.DataFrame, output_path: Path) -> bool:
    """Generate a 640x640 candlestick chart image."""
    if df.empty or len(df) < 10:
        return False

    chart_df = df.copy()
    chart_df.set_index('timestamp', inplace=True)
    chart_df = chart_df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })

    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit', wick='inherit', volume='gray'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle='',
        facecolor='white', edgecolor='white'
    )

    fig, axes = mpf.plot(
        chart_df, type='candle', style=style,
        volume=False, returnfig=True,
        figsize=(6.4, 6.4), tight_layout=True
    )

    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(str(output_path), dpi=100, bbox_inches='tight',
                pad_inches=0, facecolor='white')
    plt.close(fig)

    # Resize to exact 640x640
    img = Image.open(output_path)
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        img.save(output_path)

    return True


def main():
    parser = argparse.ArgumentParser(description="Scan for W_Bottom patterns (local data)")
    parser.add_argument('--year', type=int, default=2025, help='Year to scan')
    parser.add_argument('--ticker', type=str, default='QQQ', help='Ticker symbol')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    print("=" * 60)
    print(f"W_Bottom Scanner - {args.ticker} {args.year} (Local Data)")
    print("=" * 60)

    # Load local data
    print("\nLoading local data...")
    df = load_local_data(args.ticker)
    print(f"Total bars: {len(df):,}")

    # Filter to target year
    df['date'] = df['timestamp'].dt.date
    df['year'] = df['timestamp'].dt.year
    df_year = df[df['year'] == args.year].copy()

    # Get unique trading days
    trading_days = sorted(df_year['date'].unique())
    print(f"Trading days in {args.year}: {len(trading_days)}")

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")

    # Initialize detector
    print("\nLoading YOLO model...")
    detector = YOLOPatternDetector(confidence=args.confidence)

    # Results
    results = []
    temp_chart = OUTPUT_DIR / f"temp_chart_{args.year}.png"  # Unique per year to avoid race conditions

    print(f"\nScanning {len(trading_days)} days...\n")

    for i, date in enumerate(trading_days):
        date_str = date.strftime('%Y-%m-%d')

        # Progress
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(trading_days)} ({len(results)} W_Bottoms found)...")

        # Get day data
        day_df = get_day_data(df_year, date)
        if len(day_df) < 100:
            continue

        # Generate chart
        if not generate_chart_image(day_df, temp_chart):
            continue

        # Detect patterns
        detections = detector.detect(str(temp_chart))
        w_bottoms = [d for d in detections if d['class_name'] == 'W_Bottom']

        if w_bottoms:
            # Save image with bbox
            output_path = OUTPUT_DIR / f"{args.ticker}_{date_str}_w_bottom.png"
            detector.detect_and_save(str(temp_chart), str(output_path))

            best = max(w_bottoms, key=lambda x: x['confidence'])
            bbox = best['bbox']  # [x1, y1, x2, y2] in pixels

            # Calculate bar indices from pixel bbox
            # Chart is 640x640, x-axis maps to bar indices
            n_bars = len(day_df)
            x1, y1, x2, y2 = bbox
            bar_start = int(round(x1 / IMAGE_SIZE * n_bars))
            bar_end = int(round(x2 / IMAGE_SIZE * n_bars))
            # Clamp to valid range
            bar_start = max(0, min(n_bars - 1, bar_start))
            bar_end = max(0, min(n_bars - 1, bar_end))

            results.append({
                'date': date_str,
                'confidence': best['confidence'],
                'bbox': bbox,
                'bar_start': bar_start,
                'bar_end': bar_end,
                'num_detections': len(w_bottoms)
            })
            print(f"  {date_str}: W_Bottom (conf={best['confidence']:.1%}, bars={bar_start}-{bar_end})")

    # Cleanup
    if temp_chart.exists():
        temp_chart.unlink()

    # Summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Year: {args.year}")
    print(f"Days scanned: {len(trading_days)}")
    print(f"W_Bottom found: {len(results)}")
    if len(trading_days) > 0:
        print(f"Detection rate: {len(results)/len(trading_days)*100:.1f}%")
    else:
        print("Detection rate: N/A (no trading days found)")
    print(f"Output: {OUTPUT_DIR}")

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_file = OUTPUT_DIR / f"{args.ticker}_{args.year}_w_bottom_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Results saved: {results_file}")

        # Top 10 by confidence
        print(f"\nTop 10 by confidence:")
        for r in sorted(results, key=lambda x: x['confidence'], reverse=True)[:10]:
            print(f"  {r['date']}: {r['confidence']:.1%}")


if __name__ == "__main__":
    main()
