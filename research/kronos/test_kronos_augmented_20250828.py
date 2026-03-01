"""
Test Kronos-Augmented YOLO Detection on 2025-08-28 (worst lag case)

Goal: Detect the W_Bottom at 11:58 instead of 15:55 (4 hour improvement)

This script:
1. Fetches QQQ data for 2025-08-28 up to 11:57 AM (pattern end time)
2. Runs Kronos to predict next 60 bars
3. Generates a chart with real + predicted data
4. Runs YOLO detection on the combined chart
5. Compares with the original detection that took 4 hours
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

import pandas as pd
import numpy as np
import pytz

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Kronos"))

import mplfinance as mpf
import matplotlib.pyplot as plt

# Import our modules
from pattern_recognition.data_fetcher import fetch_single_ticker, get_alpaca_client, load_config
from pattern_recognition.yolo_inference import YOLOPatternDetector

# Import Kronos
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_historical_data_for_date(ticker: str, target_date: datetime, cutoff_time: datetime, min_context_bars: int = 480):
    """
    Fetch historical data up to a specific cutoff time on a specific date.

    Ensures we have enough context for Kronos by including previous day(s) RTH bars.
    Kronos needs ~480 bars for optimal prediction.

    Args:
        ticker: Stock symbol
        target_date: The target date
        cutoff_time: Cutoff time on target date
        min_context_bars: Minimum bars needed for Kronos context (default 480)

    Returns:
        DataFrame with OHLCV data
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

    # Calculate how many days we need to get enough context
    # RTH = 390 bars/day, so 2 days = 780 bars (plenty for 480 context)
    days_back = 5  # Fetch 5 days to ensure we have enough after weekends/holidays

    start_time = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 9, 30)) - timedelta(days=days_back)
    end_time = eastern.localize(cutoff_time)

    print(f"Fetching data from {start_time} to {end_time}")
    print(f"Target: At least {min_context_bars} bars for Kronos context")

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

    # Take last N bars to ensure we have proper context
    # Include previous day(s) RTH + current day up to cutoff
    if len(df) > min_context_bars + 100:  # Keep some buffer
        df = df.tail(min_context_bars + 100)

    print(f"Fetched {len(df)} bars (need {min_context_bars} for Kronos)")

    # Show breakdown by date
    df_temp = df.copy()
    df_temp['date'] = df_temp['timestamp'].dt.date
    date_counts = df_temp.groupby('date').size()
    print("\nBars per day:")
    for date, count in date_counts.items():
        print(f"  {date}: {count} bars")

    print(f"\nDate range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    return df


def run_kronos_prediction(real_data: pd.DataFrame, prediction_horizon: int = 60, context_length: int = 480):
    """
    Run Kronos to predict future candlesticks.

    Args:
        real_data: Historical OHLCV data
        prediction_horizon: Number of bars to predict
        context_length: Number of bars to use as context for Kronos (default 480)
    """
    print("\nInitializing Kronos model...")

    # Load Kronos
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device="cuda:0",
        max_context=512
    )

    # Use last context_length bars for Kronos input
    if len(real_data) > context_length:
        context_data = real_data.tail(context_length).copy()
        print(f"Using last {context_length} bars as context (from {len(real_data)} available)")
    else:
        context_data = real_data.copy()
        print(f"Using all {len(context_data)} bars as context (less than optimal {context_length})")

    print(f"Context period: {context_data['timestamp'].iloc[0]} to {context_data['timestamp'].iloc[-1]}")
    print(f"Kronos loaded. Predicting {prediction_horizon} future bars...")

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

    # Run prediction with return_samples=True to get full OHLCV
    pred_result = predictor.predict(
        df=context_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=prediction_horizon,
        sample_count=10,  # More samples for better mean
        T=1.0,
        top_p=0.9,
        return_samples=True,
        verbose=True
    )

    # Extract mean prediction
    predicted_df = pred_result['mean'].copy()
    predicted_df['timestamp'] = y_timestamp.values

    # Also show individual sample predictions for insight
    samples = pred_result['samples']
    sample_finals = [s['close'].iloc[-1] for s in samples]
    current_price = context_df['close'].iloc[-1]

    print(f"\nKronos prediction complete!")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted {len(predicted_df)} bars")
    print(f"Predicted price range: ${predicted_df['close'].min():.2f} - ${predicted_df['close'].max():.2f}")
    print(f"\nSample final prices (10 samples):")
    for i, final in enumerate(sample_finals):
        direction = "UP" if final > current_price else "DOWN"
        print(f"  Sample {i+1}: ${final:.2f} ({direction} ${abs(final-current_price):.2f})")

    up_count = sum(1 for f in sample_finals if f > current_price)
    print(f"\nDirection consensus: {up_count}/10 samples predict UP")

    return predicted_df, pred_result


def generate_chart(real_data: pd.DataFrame, predicted_data: pd.DataFrame, output_path: str, show_boundary: bool = True):
    """
    Generate candlestick chart with real + predicted data.
    """
    # Combine data
    real_copy = real_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    pred_copy = predicted_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Make timestamps timezone-naive for mplfinance
    real_copy['timestamp'] = pd.to_datetime(real_copy['timestamp']).dt.tz_localize(None)
    pred_copy['timestamp'] = pd.to_datetime(pred_copy['timestamp'])

    combined_df = pd.concat([real_copy, pred_copy], ignore_index=True)

    # Prepare for mplfinance
    chart_df = combined_df.copy()
    chart_df.set_index('timestamp', inplace=True)
    chart_df = chart_df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })

    # Use last 150 bars only (pattern region + predictions)
    chart_df = chart_df.tail(150)

    print(f"\nGenerating chart with {len(real_copy)} real + {len(pred_copy)} predicted bars")
    print(f"Chart shows last 150 bars")

    # Create chart
    fig, axes = mpf.plot(
        chart_df,
        type='candle',
        style='charles',
        volume=True,
        figsize=(14, 8),
        returnfig=True,
        title=f'\nQQQ 2025-08-28 | Real Data + Kronos Predictions',
        tight_layout=True
    )

    # Add vertical line at boundary if requested
    if show_boundary:
        boundary_idx = len(real_copy) - (len(combined_df) - 150)  # Adjust for tail
        if boundary_idx > 0 and boundary_idx < 150:
            # axes[0] is the price axis
            axes[0].axvline(x=boundary_idx, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            axes[0].text(boundary_idx + 1, axes[0].get_ylim()[1] * 0.99,
                        'Kronos →', color='blue', fontsize=10, verticalalignment='top')

    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Chart saved to: {output_path}")
    return output_path


def run_yolo_detection(chart_path: str):
    """
    Run YOLO detection on the chart.
    """
    print("\nInitializing YOLO detector...")
    detector = YOLOPatternDetector(confidence=0.20)  # Lower threshold for testing

    print("Running detection...")
    detections = detector.detect(chart_path)

    return detections


def main():
    print("=" * 60)
    print("KRONOS-AUGMENTED YOLO DETECTION TEST")
    print("Test Case: 2025-08-28 (Worst Detection Lag - 4 hours)")
    print("=" * 60)

    # Target: 2025-08-28, cutoff at 11:57 AM (pattern end time)
    target_date = datetime(2025, 8, 28)
    cutoff_time = datetime(2025, 8, 28, 11, 57, 0)

    # Create output directory
    output_dir = Path("./kronos_augmented_test")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Fetch historical data up to pattern end time
    print("\n" + "-" * 40)
    print("STEP 1: Fetching historical data")
    print("-" * 40)

    real_data = fetch_historical_data_for_date("QQQ", target_date, cutoff_time)

    if real_data is None or len(real_data) == 0:
        print("ERROR: Failed to fetch data!")
        return

    print(f"\nReal data: {len(real_data)} bars")
    print(f"Last timestamp: {real_data['timestamp'].iloc[-1]}")
    print(f"Last close: ${real_data['close'].iloc[-1]:.2f}")

    # Step 2: Run Kronos prediction
    print("\n" + "-" * 40)
    print("STEP 2: Running Kronos prediction")
    print("-" * 40)

    predicted_data, pred_result = run_kronos_prediction(real_data, prediction_horizon=60, context_length=480)

    # Step 3: Generate chart with real + predicted
    print("\n" + "-" * 40)
    print("STEP 3: Generating augmented chart")
    print("-" * 40)

    chart_path = output_dir / "augmented_chart_20250828.png"
    generate_chart(real_data, predicted_data, str(chart_path))

    # Also generate a chart without predictions for comparison
    chart_path_real_only = output_dir / "real_only_chart_20250828.png"

    real_only_df = real_data.copy()
    real_only_df['timestamp'] = pd.to_datetime(real_only_df['timestamp']).dt.tz_localize(None)
    chart_df_real = real_only_df.set_index('timestamp').tail(150)
    chart_df_real = chart_df_real.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    fig2, _ = mpf.plot(
        chart_df_real,
        type='candle',
        style='charles',
        volume=True,
        figsize=(14, 8),
        returnfig=True,
        title=f'\nQQQ 2025-08-28 | Real Data Only (up to 11:57 AM)',
        tight_layout=True
    )
    fig2.savefig(str(chart_path_real_only), dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Real-only chart saved to: {chart_path_real_only}")

    # Step 4: Run YOLO detection on both charts
    print("\n" + "-" * 40)
    print("STEP 4: Running YOLO detection")
    print("-" * 40)

    print("\n--- Detection on AUGMENTED chart (with Kronos predictions) ---")
    detections_augmented = run_yolo_detection(str(chart_path))

    print("\n--- Detection on REAL-ONLY chart ---")
    detections_real_only = run_yolo_detection(str(chart_path_real_only))

    # Step 5: Report results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    print("\n📊 AUGMENTED CHART (Real + Kronos Predictions):")
    if detections_augmented:
        for det in detections_augmented:
            print(f"  ✅ Pattern: {det['class_name']}")
            print(f"     Confidence: {det['confidence']:.2%}")
            print(f"     Bbox: {det['bbox']}")
    else:
        print("  ❌ No patterns detected")

    print("\n📊 REAL-ONLY CHART (No predictions):")
    if detections_real_only:
        for det in detections_real_only:
            print(f"  ✅ Pattern: {det['class_name']}")
            print(f"     Confidence: {det['confidence']:.2%}")
    else:
        print("  ❌ No patterns detected (as expected - original needed 4 more hours)")

    print("\n" + "-" * 40)
    print("TIMING ANALYSIS")
    print("-" * 40)
    print(f"  Pattern formed:           11:06 - 11:57 AM")
    print(f"  Original detection time:  15:55 (4 hours lag)")
    print(f"  Augmented detection time: 11:58 (if detected above)")

    w_bottom_found = any(d['class_name'] == 'W_Bottom' for d in detections_augmented)
    if w_bottom_found:
        print(f"\n  🎉 SUCCESS! W_Bottom detected ~4 hours earlier!")
    else:
        print(f"\n  ⚠️  W_Bottom not detected. May need parameter tuning.")

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  Augmented chart:  {chart_path}")
    print(f"  Real-only chart:  {chart_path_real_only}")

    return {
        'detections_augmented': detections_augmented,
        'detections_real_only': detections_real_only,
        'chart_path': str(chart_path)
    }


if __name__ == "__main__":
    main()
