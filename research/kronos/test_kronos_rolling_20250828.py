"""
Rolling Kronos-Augmented YOLO Detection Test for 2025-08-28

This script simulates stepping through the day bar by bar:
1. At each step, use all data available up to that point
2. Run Kronos to predict 60 future bars
3. Generate augmented chart and run YOLO detection
4. Track when W_Bottom is first detected

Goal: See if Kronos augmentation enables earlier detection than 15:55 (original)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json

import pandas as pd
import numpy as np
import pytz

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Kronos"))

import mplfinance as mpf
import matplotlib.pyplot as plt

# Import our modules
from pattern_recognition.data_fetcher import load_config
from pattern_recognition.yolo_inference import YOLOPatternDetector

# Import Kronos
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_full_day_data(ticker: str, target_date: datetime):
    """
    Fetch full day data for 2025-08-28 plus previous days for context.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    config = load_config()
    client = StockHistoricalDataClient(
        config['ALPACA_KEY_ID'],
        config['ALPACA_SECRET_KEY']
    )

    eastern = pytz.timezone('US/Eastern')

    # Fetch 5 days back to ensure enough context
    start_time = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 9, 30)) - timedelta(days=5)
    end_time = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 16, 0))

    print(f"Fetching data from {start_time} to {end_time}")

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time,
        feed=DataFeed.SIP
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    if df.empty:
        print("No data returned!")
        return None

    # Convert to Eastern time
    df['timestamp'] = df['timestamp'].dt.tz_convert(eastern)

    # Filter to RTH only (9:30-15:59)
    df = df.set_index('timestamp')
    df = df.between_time('09:30', '15:59')
    df = df.reset_index()

    # Rename columns to lowercase
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })

    # Remove symbol column if present
    if 'symbol' in df.columns:
        df = df.drop(columns=['symbol'])

    return df


def initialize_models():
    """Initialize Kronos and YOLO models."""
    print("Initializing models...")

    # Kronos
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device="cuda:0",
        max_context=512
    )

    # YOLO
    yolo = YOLOPatternDetector(confidence=0.20)

    print("Models initialized.")
    return predictor, yolo


def run_kronos_prediction(predictor, real_data: pd.DataFrame, prediction_horizon: int = 60):
    """Run Kronos prediction and return predicted bars."""
    context_length = min(480, len(real_data))
    context_data = real_data.tail(context_length).copy()

    # Prepare timestamps - make timezone-naive for Kronos
    x_timestamp = pd.to_datetime(context_data['timestamp']).dt.tz_localize(None)
    x_timestamp = pd.Series(x_timestamp.values)

    last_ts = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(pd.date_range(
        start=last_ts + pd.Timedelta(minutes=1),
        periods=prediction_horizon,
        freq='1min'
    ))

    # Prepare data for Kronos
    context_df = context_data[['open', 'high', 'low', 'close', 'volume']].copy()
    context_df['amount'] = context_df['volume'] * context_df['close']

    # Run prediction
    pred_result = predictor.predict(
        df=context_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=prediction_horizon,
        sample_count=5,  # Fewer samples for speed
        T=1.0,
        top_p=0.9,
        return_samples=True,
        verbose=False
    )

    # Extract mean prediction
    predicted_df = pred_result['mean'].copy()
    predicted_df['timestamp'] = y_timestamp.values

    # Get direction consensus
    samples = pred_result['samples']
    sample_finals = [s['close'].iloc[-1] for s in samples]
    current_price = context_df['close'].iloc[-1]
    up_count = sum(1 for f in sample_finals if f > current_price)

    return predicted_df, up_count, len(samples)


def generate_chart_and_detect(real_data: pd.DataFrame, predicted_data: pd.DataFrame,
                              yolo: YOLOPatternDetector, output_dir: Path,
                              step_name: str, include_predictions: bool = True):
    """Generate chart and run YOLO detection."""

    # Prepare data
    real_copy = real_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    real_copy['timestamp'] = pd.to_datetime(real_copy['timestamp']).dt.tz_localize(None)

    if include_predictions and predicted_data is not None:
        pred_copy = predicted_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        pred_copy['timestamp'] = pd.to_datetime(pred_copy['timestamp'])
        combined_df = pd.concat([real_copy, pred_copy], ignore_index=True)
    else:
        combined_df = real_copy

    # Prepare for mplfinance
    chart_df = combined_df.copy()
    chart_df.set_index('timestamp', inplace=True)
    chart_df = chart_df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })

    # Use last 150 bars
    chart_df = chart_df.tail(150)

    # Create chart
    chart_path = output_dir / f"chart_{step_name}.png"
    fig, axes = mpf.plot(
        chart_df,
        type='candle',
        style='charles',
        volume=True,
        figsize=(14, 8),
        returnfig=True,
        tight_layout=True
    )
    fig.savefig(str(chart_path), dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Run YOLO detection
    detections = yolo.detect(str(chart_path))

    # Filter for W_Bottom
    w_bottom_detections = [d for d in detections if d['class_name'] == 'W_Bottom']

    return detections, w_bottom_detections, str(chart_path)


def main():
    print("=" * 70)
    print("ROLLING KRONOS-AUGMENTED YOLO DETECTION TEST")
    print("Date: 2025-08-28 | Original Detection: 15:55 (4-hour lag)")
    print("=" * 70)

    target_date = datetime(2025, 8, 28)
    eastern = pytz.timezone('US/Eastern')

    # Create output directory
    output_dir = Path("./kronos_rolling_test")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Fetch full day data
    print("\n[1/3] Fetching full day data...")
    full_data = fetch_full_day_data("QQQ", target_date)

    if full_data is None:
        print("ERROR: Failed to fetch data!")
        return

    # Separate target day data
    target_day_mask = full_data['timestamp'].dt.date == target_date.date()
    target_day_data = full_data[target_day_mask].copy()
    previous_days_data = full_data[~target_day_mask].copy()

    print(f"Previous days: {len(previous_days_data)} bars")
    print(f"Target day (Aug 28): {len(target_day_data)} bars")

    # Step 2: Initialize models
    print("\n[2/3] Initializing models...")
    predictor, yolo = initialize_models()

    # Step 3: Rolling simulation
    print("\n[3/3] Running rolling simulation...")
    print("-" * 70)

    # Pattern was 11:06-11:57, detected at 15:55
    # Test every 5 bars from 11:00 to 16:00

    results = []
    first_w_bottom_detection = None
    first_augmented_w_bottom = None

    # Get bar indices for the simulation
    # Start from 11:00 (30 mins before pattern), step every 5 bars
    start_time = eastern.localize(datetime(2025, 8, 28, 11, 0))

    target_day_data = target_day_data.reset_index(drop=True)

    # Find starting index
    start_idx = None
    for i, row in target_day_data.iterrows():
        if row['timestamp'] >= start_time:
            start_idx = i
            break

    if start_idx is None:
        print("Could not find start index!")
        return

    print(f"Starting simulation from bar {start_idx} ({target_day_data.iloc[start_idx]['timestamp']})")
    print(f"Pattern: 11:06-11:57 AM | Original detection: 15:55")
    print("-" * 70)

    step_interval = 10  # Every 10 bars (10 minutes)

    for current_bar in range(start_idx, len(target_day_data), step_interval):
        current_time = target_day_data.iloc[current_bar]['timestamp']
        time_str = current_time.strftime("%H:%M")

        # Build data up to current bar
        current_day_slice = target_day_data.iloc[:current_bar + 1].copy()
        available_data = pd.concat([previous_days_data, current_day_slice], ignore_index=True)

        print(f"\n{time_str} | Bar {current_bar} | {len(available_data)} total bars available")

        # Run Kronos prediction
        try:
            predicted_df, up_count, total_samples = run_kronos_prediction(
                predictor, available_data, prediction_horizon=60
            )
            direction = "UP" if up_count > total_samples / 2 else "DOWN"
            print(f"  Kronos: {up_count}/{total_samples} predict UP ({direction})")
        except Exception as e:
            print(f"  Kronos error: {e}")
            predicted_df = None
            direction = "ERROR"

        # Run YOLO on augmented chart (with Kronos predictions)
        if predicted_df is not None:
            all_det, w_det, _ = generate_chart_and_detect(
                available_data, predicted_df, yolo, output_dir,
                f"aug_{time_str.replace(':', '')}", include_predictions=True
            )
            aug_w_bottom = len(w_det) > 0
            aug_confidence = max([d['confidence'] for d in w_det]) if w_det else 0
            print(f"  YOLO (augmented): {len(all_det)} patterns, W_Bottom: {aug_w_bottom} ({aug_confidence:.1%})")

            if aug_w_bottom and first_augmented_w_bottom is None:
                first_augmented_w_bottom = {
                    'time': time_str,
                    'bar': current_bar,
                    'confidence': aug_confidence,
                    'kronos_direction': direction
                }
                print(f"  >>> FIRST AUGMENTED W_BOTTOM DETECTION!")
        else:
            aug_w_bottom = False
            aug_confidence = 0

        # Run YOLO on real-only chart (no Kronos)
        all_det_real, w_det_real, _ = generate_chart_and_detect(
            available_data, None, yolo, output_dir,
            f"real_{time_str.replace(':', '')}", include_predictions=False
        )
        real_w_bottom = len(w_det_real) > 0
        real_confidence = max([d['confidence'] for d in w_det_real]) if w_det_real else 0
        print(f"  YOLO (real-only): {len(all_det_real)} patterns, W_Bottom: {real_w_bottom} ({real_confidence:.1%})")

        if real_w_bottom and first_w_bottom_detection is None:
            first_w_bottom_detection = {
                'time': time_str,
                'bar': current_bar,
                'confidence': real_confidence
            }
            print(f"  >>> FIRST REAL-ONLY W_BOTTOM DETECTION!")

        results.append({
            'time': time_str,
            'bar': current_bar,
            'kronos_direction': direction,
            'aug_w_bottom': aug_w_bottom,
            'aug_confidence': aug_confidence,
            'real_w_bottom': real_w_bottom,
            'real_confidence': real_confidence
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nPattern formed: 11:06 - 11:57 AM")
    print(f"Original backtest detection: 15:55 (4-hour lag)")

    print(f"\n--- Rolling Simulation Results ---")

    if first_w_bottom_detection:
        print(f"\nFirst REAL-ONLY W_Bottom detection:")
        print(f"  Time: {first_w_bottom_detection['time']}")
        print(f"  Confidence: {first_w_bottom_detection['confidence']:.1%}")
    else:
        print(f"\nReal-only: NO W_Bottom detected in simulation")

    if first_augmented_w_bottom:
        print(f"\nFirst AUGMENTED W_Bottom detection:")
        print(f"  Time: {first_augmented_w_bottom['time']}")
        print(f"  Confidence: {first_augmented_w_bottom['confidence']:.1%}")
        print(f"  Kronos direction: {first_augmented_w_bottom['kronos_direction']}")
    else:
        print(f"\nAugmented: NO W_Bottom detected in simulation")

    # Kronos direction analysis
    kronos_directions = [r['kronos_direction'] for r in results if r['kronos_direction'] != 'ERROR']
    up_predictions = sum(1 for d in kronos_directions if d == 'UP')
    print(f"\nKronos direction across day:")
    print(f"  UP predictions: {up_predictions}/{len(kronos_directions)}")
    print(f"  DOWN predictions: {len(kronos_directions) - up_predictions}/{len(kronos_directions)}")

    # Save results
    results_path = output_dir / "rolling_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'date': '2025-08-28',
            'pattern_time': '11:06-11:57',
            'original_detection': '15:55',
            'first_real_detection': first_w_bottom_detection,
            'first_augmented_detection': first_augmented_w_bottom,
            'step_results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
