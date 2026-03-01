"""
Prepare training data for Kronos fine-tuning on W_Bottom patterns.

Uses bbox from YOLO scan results to identify pattern location,
then creates input/target pairs for Kronos model training.

Training Data Structure:
    INPUT = 150 context bars BEFORE pattern + complete pattern bars
    TARGET = 60 bars AFTER pattern (breakout continuation)

Training Strategy:
    - Multi-ticker support (QQQ, IWM, NVDA, TSLA)
    - Filter to SUCCESSFUL breakouts only (from breakout_analysis_results.csv)
    - Goal: Train Kronos to predict upward continuation instead of mean reversion

Usage:
    python -m pattern_recognition.prepare_kronos_training_data
    python -m pattern_recognition.prepare_kronos_training_data --all-patterns  # Include failed patterns
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration
DATA_DIR = Path(__file__).parent.parent / "historical_data"
SCAN_RESULTS_DIR = Path(__file__).parent.parent / "yolo_w_bottom_scan"
BREAKOUT_RESULTS = Path(__file__).parent.parent / "breakout_analysis" / "breakout_analysis_results.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "kronos_training_data"

# Supported tickers
TICKERS = ['QQQ', 'IWM', 'NVDA', 'TSLA']

# Training parameters
CONTEXT_BARS = 150    # Bars BEFORE pattern
TARGET_BARS = 60      # Bars AFTER pattern (breakout)
CHART_BARS = 390      # Full trading day (used during scan)
REQUIRE_POSITIVE_RETURN = True  # Only include samples with positive 60-bar return


def calculate_atr(ohlcv: np.ndarray, period: int = 14) -> float:
    """
    Calculate Average True Range for volatility scaling.

    Args:
        ohlcv: numpy array with columns [open, high, low, close, volume]
        period: ATR period (default 14)

    Returns:
        ATR value (average of last `period` True Range values)
    """
    if len(ohlcv) < 2:
        return 0.0

    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    tr_list = []
    for i in range(1, len(ohlcv)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.0

    return np.mean(tr_list[-period:])


def bbox_to_bar_indices(bbox_str: str, num_bars: int) -> tuple:
    """
    Convert YOLO bbox x-coordinates to bar indices.

    Args:
        bbox_str: String like "[x1, y1, x2, y2]" in pixels (0-640 range)
        num_bars: Number of bars in the day

    Returns:
        (pattern_start_bar, pattern_end_bar)
    """
    bbox = ast.literal_eval(bbox_str)
    x1, y1, x2, y2 = bbox

    # Map x-coordinate (0-640) to bar index (0-num_bars)
    pattern_start = int(x1 / 640 * num_bars)
    pattern_end = int(x2 / 640 * num_bars)

    return pattern_start, pattern_end


def load_historical_data(ticker: str = "QQQ") -> pd.DataFrame:
    """Load the full historical data (prefer CSV for 2020-2025 range)."""

    def get_date_range(f):
        """Extract year range from filename."""
        parts = f.stem.split('_')
        try:
            return int(parts[2]) - int(parts[1])
        except (IndexError, ValueError):
            return 0

    # Try CSV first (has 2020-2025 data)
    csv_files = list(DATA_DIR.glob(f"{ticker}_*_1min_rth.csv"))
    if csv_files:
        file = max(csv_files, key=get_date_range)
        print(f"Loading: {file.name}")
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        return df

    # Fallback to parquet
    parquet_files = list(DATA_DIR.glob(f"{ticker}_*_1min_rth.parquet"))
    if parquet_files:
        file = max(parquet_files, key=get_date_range)
        print(f"Loading: {file.name}")
        df = pd.read_parquet(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        return df

    raise FileNotFoundError(f"No data file found for {ticker} in {DATA_DIR}")


def load_w_bottom_results(successful_only: bool = True) -> list:
    """
    Load W_Bottom results from breakout analysis.

    Args:
        successful_only: If True, only load successful breakout patterns.
                        If False, load all patterns.

    Returns:
        List of pattern dicts with ticker, date, confidence, bbox
    """
    all_results = []

    # Load from breakout analysis results (has success flag)
    if BREAKOUT_RESULTS.exists():
        print(f"  Loading from: {BREAKOUT_RESULTS.name}")
        breakout_df = pd.read_csv(BREAKOUT_RESULTS)

        if successful_only:
            breakout_df = breakout_df[breakout_df['success'] == True]
            print(f"  Filtering to successful breakouts only")

        # Group by ticker for reporting
        for ticker in TICKERS:
            ticker_df = breakout_df[breakout_df['ticker'] == ticker]
            print(f"    {ticker}: {len(ticker_df)} patterns")

        # Now load bbox from original scan results
        for _, row in breakout_df.iterrows():
            ticker = row['ticker']
            date_str = row['date']
            date = pd.to_datetime(date_str).date()

            # Find bbox from original scan CSV
            year = date.year
            scan_file = SCAN_RESULTS_DIR / f"{ticker}_{year}_w_bottom_results.csv"
            if scan_file.exists():
                scan_df = pd.read_csv(scan_file)
                scan_df['date'] = pd.to_datetime(scan_df['date']).dt.date
                match = scan_df[scan_df['date'] == date]
                if len(match) > 0:
                    bbox = match.iloc[0]['bbox']
                    all_results.append({
                        'ticker': ticker,
                        'date': date,
                        'confidence': row['confidence'],
                        'bbox': bbox,
                        'year': year,
                        'success': row['success']
                    })
    else:
        # Fallback: load directly from scan results (no success filter)
        print(f"  Warning: {BREAKOUT_RESULTS} not found, loading from scan results")
        for ticker in TICKERS:
            for year in range(2020, 2026):
                csv_file = SCAN_RESULTS_DIR / f"{ticker}_{year}_w_bottom_results.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    for _, row in df.iterrows():
                        all_results.append({
                            'ticker': ticker,
                            'date': pd.to_datetime(row['date']).date(),
                            'confidence': row['confidence'],
                            'bbox': row['bbox'],
                            'year': year,
                            'success': None
                        })
                    print(f"    {ticker} {year}: {len(df)} samples")

    print(f"  Total patterns loaded: {len(all_results)}")
    return all_results


def get_day_data(df: pd.DataFrame, target_date) -> pd.DataFrame:
    """Extract data for a specific date."""
    return df[df['date'] == target_date].reset_index(drop=True)


def create_training_sample(df: pd.DataFrame, day_df: pd.DataFrame,
                           bbox_str: str, date) -> dict:
    """
    Create a training sample (input/target pair) using bbox to identify pattern.

    Args:
        df: Full historical DataFrame (for getting previous days if needed)
        day_df: Current day's DataFrame
        bbox_str: Bbox string from scan CSV
        date: Date of the pattern

    Returns:
        dict with input, target, and metadata, or None if invalid
    """
    num_bars = len(day_df)

    # Map bbox to bar indices
    pattern_start, pattern_end = bbox_to_bar_indices(bbox_str, num_bars)

    # Validate: pattern_end must be within the day
    if pattern_end >= num_bars:
        return None

    # Validate: need enough bars after pattern for target
    if pattern_end + TARGET_BARS > num_bars:
        return None

    # Calculate input range
    # INPUT = CONTEXT_BARS before pattern_start + pattern bars
    bars_before_pattern = pattern_start
    bars_needed_from_prev = max(0, CONTEXT_BARS - bars_before_pattern)

    if bars_needed_from_prev > 0:
        # Need bars from previous days
        prev_data = df[df['date'] < date].tail(bars_needed_from_prev)
        if len(prev_data) < bars_needed_from_prev:
            return None  # Not enough historical data

        # Combine: prev_data + current day up to pattern_end
        input_df = pd.concat([
            prev_data[['open', 'high', 'low', 'close', 'volume']],
            day_df.iloc[:pattern_end][['open', 'high', 'low', 'close', 'volume']]
        ], ignore_index=True)
    else:
        # All context comes from current day
        input_start = pattern_start - CONTEXT_BARS
        input_df = day_df.iloc[input_start:pattern_end][['open', 'high', 'low', 'close', 'volume']]

    # TARGET: bars after pattern
    target_df = day_df.iloc[pattern_end:pattern_end + TARGET_BARS][['open', 'high', 'low', 'close', 'volume']]

    if len(target_df) < TARGET_BARS:
        return None

    # Calculate 60-bar return for filtering
    last_input_close = input_df.iloc[-1]['close']
    final_target_close = target_df.iloc[-1]['close']
    target_return = (final_target_close - last_input_close) / last_input_close * 100

    # Calculate ATR for volatility scaling (using input data only - no future leakage)
    input_ohlcv = input_df.values  # numpy array [open, high, low, close, volume]
    atr_14 = calculate_atr(input_ohlcv, period=14)
    atr_pct = (atr_14 / last_input_close) * 100 if last_input_close > 0 else 0.0

    return {
        'input': input_df.values.tolist(),
        'target': target_df.values.tolist(),
        'date': str(date),
        'pattern_start': pattern_start,
        'pattern_end': pattern_end,
        'pattern_bars': pattern_end - pattern_start,
        'input_length': len(input_df),
        'target_length': len(target_df),
        'target_return': target_return,
        'atr_14': atr_14,
        'atr_pct': atr_pct,
        'last_close': last_input_close
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Kronos training data from W_Bottom patterns")
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--all-patterns', action='store_true', help='Include failed breakouts (default: successful only)')
    parser.add_argument('--include-negative-returns', action='store_true', help='Include samples with negative 60-bar returns')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_only = not args.all_patterns

    print("=" * 60)
    print("Kronos Training Data Preparation (Multi-Ticker)")
    print("=" * 60)
    print(f"Training mode: {'SUCCESSFUL BREAKOUTS ONLY' if successful_only else 'ALL PATTERNS'}")
    print(f"Tickers: {', '.join(TICKERS)}")

    # Load historical data for all tickers
    print("\n1. Loading historical data...")
    ticker_data = {}
    for ticker in TICKERS:
        try:
            df = load_historical_data(ticker)
            ticker_data[ticker] = df
            print(f"   {ticker}: {len(df):,} bars ({df['date'].min()} to {df['date'].max()})")
        except FileNotFoundError as e:
            print(f"   {ticker}: SKIPPED - {e}")

    # Load W_Bottom results
    print("\n2. Loading W_Bottom patterns...")
    w_bottom_results = load_w_bottom_results(successful_only=successful_only)

    # Balance dataset when using all patterns (50/50 success/failed)
    if not successful_only:
        success_patterns = [p for p in w_bottom_results if p.get('success') == True]
        failed_patterns = [p for p in w_bottom_results if p.get('success') == False]
        print(f"\n   Balancing dataset:")
        print(f"     Successful patterns: {len(success_patterns)}")
        print(f"     Failed patterns: {len(failed_patterns)}")

        min_count = min(len(success_patterns), len(failed_patterns))
        if min_count > 0:
            np.random.seed(42)
            if len(success_patterns) > min_count:
                success_indices = np.random.choice(len(success_patterns), min_count, replace=False)
                success_patterns = [success_patterns[i] for i in success_indices]
            if len(failed_patterns) > min_count:
                failed_indices = np.random.choice(len(failed_patterns), min_count, replace=False)
                failed_patterns = [failed_patterns[i] for i in failed_indices]

            w_bottom_results = success_patterns + failed_patterns
            np.random.shuffle(w_bottom_results)
            print(f"     Balanced to: {len(w_bottom_results)} total ({min_count} success + {min_count} failed)")

    # Create training samples
    print("\n3. Creating training samples...")
    samples = []
    skipped = 0
    skip_reasons = {'no_data': 0, 'no_ticker_data': 0, 'pattern_end_overflow': 0, 'not_enough_target': 0, 'not_enough_context': 0}

    for i, wb in enumerate(w_bottom_results):
        ticker = wb['ticker']
        date = wb['date']

        if (i + 1) % 200 == 0:
            print(f"   Processing {i + 1}/{len(w_bottom_results)}...")

        # Get historical data for this ticker
        if ticker not in ticker_data:
            skipped += 1
            skip_reasons['no_ticker_data'] += 1
            continue

        df = ticker_data[ticker]

        # Get day data
        day_df = get_day_data(df, date)
        if len(day_df) < 100:
            skipped += 1
            skip_reasons['no_data'] += 1
            continue

        # Create training sample
        sample = create_training_sample(df, day_df, wb['bbox'], date)
        if sample is None:
            skipped += 1
            continue

        # Filter: only include samples with positive 60-bar return (unless disabled)
        require_positive = REQUIRE_POSITIVE_RETURN and not args.include_negative_returns
        if require_positive and sample['target_return'] <= 0:
            skipped += 1
            skip_reasons['negative_return'] = skip_reasons.get('negative_return', 0) + 1
            continue

        sample['ticker'] = ticker
        sample['confidence'] = wb['confidence']
        sample['year'] = wb['year']
        sample['success'] = wb.get('success', None)
        samples.append(sample)

    print(f"   Created: {len(samples)} samples")
    print(f"   Skipped: {skipped} samples")

    if len(samples) == 0:
        print("\nERROR: No valid samples created!")
        return

    # Analyze input lengths (variable due to pattern size)
    input_lengths = [s['input_length'] for s in samples]
    pattern_bars = [s['pattern_bars'] for s in samples]
    target_returns = [s['target_return'] for s in samples]

    print(f"\n   Input length stats:")
    print(f"     Min: {min(input_lengths)}, Max: {max(input_lengths)}, Mean: {np.mean(input_lengths):.1f}")
    print(f"   Pattern bars stats:")
    print(f"     Min: {min(pattern_bars)}, Max: {max(pattern_bars)}, Mean: {np.mean(pattern_bars):.1f}")
    print(f"   Target return stats (60-bar):")
    print(f"     Min: {min(target_returns):.2f}%, Max: {max(target_returns):.2f}%")
    print(f"     Mean: {np.mean(target_returns):.2f}%, Median: {np.median(target_returns):.2f}%")
    print(f"     All positive: {all(r > 0 for r in target_returns)}")

    # Split into train/val
    print("\n4. Splitting into train/val...")
    np.random.seed(42)
    np.random.shuffle(samples)

    val_size = int(len(samples) * args.val_split)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    print(f"   Train: {len(train_samples)}")
    print(f"   Val: {len(val_samples)}")

    # Save samples
    print("\n5. Saving samples...")

    # Save as JSON (for inspection and metadata)
    train_file = output_dir / "train_samples.json"
    val_file = output_dir / "val_samples.json"

    with open(train_file, 'w') as f:
        json.dump(train_samples, f)

    with open(val_file, 'w') as f:
        json.dump(val_samples, f)

    # For numpy arrays, we need to handle variable input lengths
    # Option 1: Pad to max length
    # Option 2: Save as list of arrays
    # Using Option 2 for flexibility

    # Save targets (fixed length)
    train_targets = np.array([s['target'] for s in train_samples])
    val_targets = np.array([s['target'] for s in val_samples])

    np.save(output_dir / "train_targets.npy", train_targets)
    np.save(output_dir / "val_targets.npy", val_targets)

    # Save inputs as object array (variable length)
    train_inputs = np.array([np.array(s['input']) for s in train_samples], dtype=object)
    val_inputs = np.array([np.array(s['input']) for s in val_samples], dtype=object)

    np.save(output_dir / "train_inputs.npy", train_inputs, allow_pickle=True)
    np.save(output_dir / "val_inputs.npy", val_inputs, allow_pickle=True)

    # Summary
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Context bars: {CONTEXT_BARS}")
    print(f"Target bars: {TARGET_BARS}")
    print(f"Input shape: (variable, 5) - OHLCV")
    print(f"Target shape: ({TARGET_BARS}, 5) - OHLCV")
    print(f"\nFiles saved to: {output_dir}")
    print(f"  - train_samples.json ({len(train_samples)} samples)")
    print(f"  - val_samples.json ({len(val_samples)} samples)")
    print(f"  - train_inputs.npy, train_targets.npy")
    print(f"  - val_inputs.npy, val_targets.npy")

    # Show sample distribution by ticker
    print("\nSamples by ticker:")
    ticker_counts = {}
    for s in samples:
        ticker = s.get('ticker', 'Unknown')
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    for ticker in sorted(ticker_counts.keys()):
        print(f"  {ticker}: {ticker_counts[ticker]}")

    # Show sample distribution by year
    print("\nSamples by year:")
    year_counts = {}
    for s in samples:
        year = s['year']
        year_counts[year] = year_counts.get(year, 0) + 1
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]}")

    # Show sample dates (first 5)
    print("\nSample dates (first 5):")
    for s in train_samples[:5]:
        print(f"  {s['date']}: conf={s['confidence']:.1%}, pattern_bars={s['pattern_bars']}, input_len={s['input_length']}")


if __name__ == "__main__":
    main()
