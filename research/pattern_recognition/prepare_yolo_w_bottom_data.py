"""
Prepare YOLO training data from breakout_analysis results.

Generates clean 640x640 candlestick charts and YOLO-format labels
from the validated W_Bottom patterns in breakout_analysis folder.

Usage:
    python -m pattern_recognition.prepare_yolo_w_bottom_data
    python -m pattern_recognition.prepare_yolo_w_bottom_data --train-split 0.8
    python -m pattern_recognition.prepare_yolo_w_bottom_data --dry-run
"""

import ast
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import shutil

# Directories
BASE_DIR = Path(__file__).parent.parent
BREAKOUT_DIR = BASE_DIR / "breakout_analysis"
BREAKOUT_CSV = BREAKOUT_DIR / "breakout_analysis_results.csv"
DATA_DIR = BASE_DIR / "historical_data"
OUTPUT_DIR = BASE_DIR / "yolo_dataset_w_bottom"

IMAGE_SIZE = 640


def load_historical_data(ticker: str) -> pd.DataFrame:
    """
    Load historical OHLCV data for a ticker.

    Finds the file with the widest date range.
    """
    data_files = list(DATA_DIR.glob(f"{ticker}_*_1min_rth.csv"))
    if not data_files:
        raise FileNotFoundError(f"No historical data found for {ticker}")

    # Sort by date range (e.g., QQQ_2020_2025 > QQQ_2020_2020)
    def get_date_range(f):
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
    df['date'] = df['timestamp'].dt.date.astype(str)

    return df


def generate_clean_chart(day_df: pd.DataFrame, output_path: Path, image_size: int = 640):
    """
    Generate a clean 640x640 candlestick chart (no axes, borders, annotations).

    This produces the same visual output as the original YOLO detection images,
    ensuring bbox coordinates remain valid.
    """
    # Create figure with exact pixel size
    fig_size = image_size / 100  # Convert to inches at 100 dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)

    # Remove all decorations
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    n_bars = len(day_df)

    # Draw candlesticks
    width = 0.6
    for i, (idx, row) in enumerate(day_df.iterrows()):
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = '#26a69a' if c >= o else '#ef5350'  # Green/Red

        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.001:  # Doji - make visible
            body_height = 0.001

        ax.add_patch(Rectangle(
            (i - width/2, body_bottom), width, body_height,
            facecolor=color, edgecolor=color
        ))

        # Wicks
        ax.plot([i, i], [l, body_bottom], color=color, linewidth=1)
        ax.plot([i, i], [body_bottom + body_height, h], color=color, linewidth=1)

    # Set axis limits (matches original chart generation)
    ax.set_xlim(-1, n_bars)

    y_min = day_df['low'].min()
    y_max = day_df['high'].max()
    y_padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, facecolor='white', edgecolor='none',
                pad_inches=0, bbox_inches='tight')
    plt.close(fig)


def pixel_to_yolo(bbox_str: str, image_size: int = 640) -> str:
    """
    Convert pixel bbox [x1, y1, x2, y2] to YOLO normalized format.

    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    """
    bbox = ast.literal_eval(bbox_str)
    x1, y1, x2, y2 = bbox

    # Convert to center format and normalize
    x_center = (x1 + x2) / 2 / image_size
    y_center = (y1 + y2) / 2 / image_size
    width = (x2 - x1) / image_size
    height = (y2 - y1) / image_size

    # Clamp to [0, 1] range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.01, min(1.0, width))
    height = max(0.01, min(1.0, height))

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def price_to_yolo(day_df: pd.DataFrame, bar_start: int, bar_end: int, image_size: int = 640) -> str:
    """
    Calculate YOLO bbox from actual bar high/low prices.

    This produces accurate bounding boxes that match the chart's y-axis scaling,
    ensuring the model learns precise pattern boundaries.

    Args:
        day_df: DataFrame with OHLCV data for the day
        bar_start: Start bar index of the pattern
        bar_end: End bar index of the pattern
        image_size: Chart image size in pixels (default 640)

    Returns:
        YOLO format string: "class_id x_center y_center width height"
    """
    n_bars = len(day_df)

    # Clamp bar indices to valid range
    bar_start = max(0, min(n_bars - 1, bar_start))
    bar_end = max(0, min(n_bars - 1, bar_end))

    # Get actual pattern high/low from bar data
    pattern_bars = day_df.iloc[bar_start:bar_end + 1]
    actual_high = pattern_bars['high'].max()
    actual_low = pattern_bars['low'].min()

    # Chart y-axis scaling (matches generate_clean_chart)
    y_min = day_df['low'].min()
    y_max = day_df['high'].max()
    y_padding = (y_max - y_min) * 0.05
    chart_y_min = y_min - y_padding
    chart_y_max = y_max + y_padding
    chart_y_range = chart_y_max - chart_y_min

    # Convert price to pixel Y (inverted: high price = low pixel)
    y1_pixel = (1 - (actual_high - chart_y_min) / chart_y_range) * image_size
    y2_pixel = (1 - (actual_low - chart_y_min) / chart_y_range) * image_size

    # X coordinates from bar indices
    # Note: Chart x-axis goes from -1 to n_bars, but we use 0 to n_bars for bbox
    x1_pixel = (bar_start / n_bars) * image_size
    x2_pixel = ((bar_end + 1) / n_bars) * image_size  # +1 to include the end bar

    # YOLO normalized format (center + dimensions)
    x_center = (x1_pixel + x2_pixel) / 2 / image_size
    y_center = (y1_pixel + y2_pixel) / 2 / image_size
    width = (x2_pixel - x1_pixel) / image_size
    height = abs(y2_pixel - y1_pixel) / image_size

    # Clamp to [0, 1] range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.01, min(1.0, width))
    height = max(0.01, min(1.0, height))

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def range_to_yolo(day_df: pd.DataFrame, range_high: float, range_low: float, image_size: int = 640) -> str:
    """
    Calculate YOLO bbox from validated range_high/range_low prices.

    Uses the day's price range to calculate accurate y-axis coordinates,
    and finds the bar indices where the pattern exists for x-axis coordinates.

    Args:
        day_df: DataFrame with OHLCV data for the day
        range_high: Pattern high price from breakout_analysis
        range_low: Pattern low price from breakout_analysis
        image_size: Chart image size in pixels (default 640)

    Returns:
        YOLO format string: "class_id x_center y_center width height"
    """
    n_bars = len(day_df)

    # Chart y-axis scaling (matches generate_clean_chart)
    y_min = day_df['low'].min()
    y_max = day_df['high'].max()
    y_padding = (y_max - y_min) * 0.05
    chart_y_min = y_min - y_padding
    chart_y_max = y_max + y_padding
    chart_y_range = chart_y_max - chart_y_min

    # Convert price to pixel Y (inverted: high price = low pixel)
    y1_pixel = (1 - (range_high - chart_y_min) / chart_y_range) * image_size
    y2_pixel = (1 - (range_low - chart_y_min) / chart_y_range) * image_size

    # Find bar indices where pattern exists (bars touching the range)
    pattern_bars = day_df[(day_df['high'] >= range_low) & (day_df['low'] <= range_high)]
    if len(pattern_bars) == 0:
        # Fallback: use middle portion of the chart
        bar_start = n_bars // 4
        bar_end = n_bars * 3 // 4
    else:
        bar_start = pattern_bars.index.min()
        bar_end = pattern_bars.index.max()

    # X coordinates from bar indices
    x1_pixel = (bar_start / n_bars) * image_size
    x2_pixel = ((bar_end + 1) / n_bars) * image_size

    # YOLO normalized format (center + dimensions)
    x_center = (x1_pixel + x2_pixel) / 2 / image_size
    y_center = (y1_pixel + y2_pixel) / 2 / image_size
    width = (x2_pixel - x1_pixel) / image_size
    height = abs(y2_pixel - y1_pixel) / image_size

    # Clamp to [0, 1] range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.01, min(1.0, width))
    height = max(0.01, min(1.0, height))

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def collect_all_patterns() -> list:
    """
    Collect all W_Bottom patterns from breakout_analysis results.

    Returns list of dicts with ticker, date, range_high, range_low, confidence, success.
    Uses validated patterns from breakout_analysis_results.csv.
    """
    if not BREAKOUT_CSV.exists():
        raise FileNotFoundError(f"Breakout analysis CSV not found: {BREAKOUT_CSV}")

    df = pd.read_csv(BREAKOUT_CSV)
    print(f"   Loaded {len(df)} patterns from breakout_analysis_results.csv")

    all_patterns = []
    for _, row in df.iterrows():
        pattern = {
            'ticker': row['ticker'],
            'date': str(row['date']),
            'range_high': float(row['range_high']),
            'range_low': float(row['range_low']),
            'confidence': float(row['confidence']),
            'success': bool(row['success'])
        }
        all_patterns.append(pattern)

    # Count by ticker
    ticker_counts = {}
    for p in all_patterns:
        ticker_counts[p['ticker']] = ticker_counts.get(p['ticker'], 0) + 1

    success_count = sum(1 for p in all_patterns if p['success'])
    print(f"   Successful breakouts: {success_count}, Failed: {len(all_patterns) - success_count}")

    return all_patterns


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO training data for W_Bottom detection")
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Count patterns without generating images')
    parser.add_argument('--clean', action='store_true', help='Remove existing output before generating')
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO W_Bottom Training Data Preparation (breakout_analysis)")
    print("=" * 60)

    # Set random seed
    random.seed(args.seed)

    # Clean output directory if requested
    if args.clean and OUTPUT_DIR.exists():
        print(f"\nCleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 1. Collect all patterns from breakout_analysis
    print("\n1. Collecting W_Bottom patterns from breakout_analysis...")
    all_patterns = collect_all_patterns()
    print(f"   Total patterns found: {len(all_patterns)}")

    # Count by ticker
    ticker_counts = {}
    for p in all_patterns:
        ticker_counts[p['ticker']] = ticker_counts.get(p['ticker'], 0) + 1
    for ticker, count in sorted(ticker_counts.items()):
        print(f"     {ticker}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without generating images.")
        return

    # 2. Shuffle and split
    print(f"\n2. Shuffling and splitting (train: {args.train_split*100:.0f}%, val: {(1-args.train_split)*100:.0f}%)...")
    random.shuffle(all_patterns)
    split_idx = int(len(all_patterns) * args.train_split)
    train_patterns = all_patterns[:split_idx]
    val_patterns = all_patterns[split_idx:]
    print(f"   Train: {len(train_patterns)}, Val: {len(val_patterns)}")

    # 3. Create output directories
    print("\n3. Creating output directories...")
    (OUTPUT_DIR / "images/train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images/val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels/train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels/val").mkdir(parents=True, exist_ok=True)

    # 4. Load historical data for all tickers found in patterns
    print("\n4. Loading historical data...")
    unique_tickers = list(set(p['ticker'] for p in all_patterns))
    ticker_data = {}
    for ticker in unique_tickers:
        try:
            ticker_data[ticker] = load_historical_data(ticker)
        except FileNotFoundError as e:
            print(f"   WARNING: {e}")

    # 5. Generate images and labels
    print("\n5. Generating images and labels...")

    # Track duplicates (same ticker+date)
    generated_files = {}
    skipped = 0
    errors = 0

    for split_name, patterns in [("train", train_patterns), ("val", val_patterns)]:
        print(f"\n   Processing {split_name} set ({len(patterns)} patterns)...")

        for p in tqdm(patterns, desc=f"   {split_name}"):
            ticker = p['ticker']
            date = p['date']

            if ticker not in ticker_data:
                skipped += 1
                continue

            df = ticker_data[ticker]
            day_df = df[df['date'] == date].reset_index(drop=True)

            if len(day_df) < 50:  # Skip days with too few bars
                skipped += 1
                continue

            # Handle duplicates (same ticker+date) with suffix
            file_key = f"{ticker}_{date}"
            if file_key in generated_files:
                suffix = generated_files[file_key]
                generated_files[file_key] = suffix + 1
                file_key_with_suffix = f"{file_key}_{suffix}"
            else:
                generated_files[file_key] = 1
                file_key_with_suffix = file_key

            try:
                # Generate clean 640x640 chart
                image_path = OUTPUT_DIR / f"images/{split_name}/{file_key_with_suffix}.png"
                if not image_path.exists():  # Skip if already generated (for duplicates)
                    generate_clean_chart(day_df, image_path)

                # Generate YOLO label from validated range_high/range_low
                yolo_label = range_to_yolo(day_df, p['range_high'], p['range_low'])

                # Write/append label
                label_path = OUTPUT_DIR / f"labels/{split_name}/{file_key_with_suffix}.txt"

                # For duplicates on same day, append to existing label file
                if label_path.exists() and file_key_with_suffix == file_key:
                    with open(label_path, 'a') as f:
                        f.write(yolo_label + '\n')
                else:
                    label_path.write_text(yolo_label + '\n')

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n   ERROR processing {file_key}: {e}")

    # 6. Write data.yaml
    print("\n6. Writing data.yaml...")
    data_yaml = f"""# YOLO Dataset for W_Bottom Detection
# Auto-generated by prepare_yolo_w_bottom_data.py
# Generated: {datetime.now().isoformat()}

path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: 1

names:
  0: w_bottom
"""
    (OUTPUT_DIR / "data.yaml").write_text(data_yaml)

    # Summary
    train_images = len(list((OUTPUT_DIR / "images/train").glob("*.png")))
    val_images = len(list((OUTPUT_DIR / "images/val").glob("*.png")))
    train_labels = len(list((OUTPUT_DIR / "labels/train").glob("*.txt")))
    val_labels = len(list((OUTPUT_DIR / "labels/val").glob("*.txt")))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total patterns processed: {len(all_patterns)}")
    print(f"Skipped (missing data): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Val: {val_images} images, {val_labels} labels")
    print(f"\nNext steps:")
    print(f"  1. Train YOLO:")
    print(f"     yolo train model=yolov8n.pt data={OUTPUT_DIR}/data.yaml epochs=100 imgsz=640")
    print(f"  2. Evaluate:")
    print(f"     yolo val model=runs/detect/train/weights/best.pt data={OUTPUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
