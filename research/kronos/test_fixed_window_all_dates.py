"""
Test Fixed Window Detection on All 5 Worst Lag Dates

Compares growing window (backtest) vs fixed 150-bar window (proposed fix)
to validate the improvement before implementing.

Dates to test:
1. 2025-08-28 - 4.0 hours lag (already tested)
2. 2025-11-11 - 3.6 hours lag
3. 2025-03-28 - 3.6 hours lag
4. 2025-01-10 - 3.4 hours lag
5. 2025-03-13 - 3.1 hours lag
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

from pattern_recognition.data_fetcher import load_config
from pattern_recognition.yolo_inference import YOLOPatternDetector


# Test dates with their pattern info
TEST_DATES = [
    {
        'date': datetime(2025, 8, 28),
        'original_detection': '15:55',
        'original_lag_hours': 4.0,
        'pattern_start': '11:06',
        'pattern_end': '11:57'
    },
    {
        'date': datetime(2025, 11, 11),
        'original_detection': '15:40',
        'original_lag_hours': 3.6,
        'pattern_start': '09:57',
        'pattern_end': '12:03'
    },
    {
        'date': datetime(2025, 3, 28),
        'original_detection': '14:55',
        'original_lag_hours': 3.6,
        'pattern_start': '10:43',
        'pattern_end': '11:19'
    },
    {
        'date': datetime(2025, 1, 10),
        'original_detection': '15:45',
        'original_lag_hours': 3.4,
        'pattern_start': '10:47',
        'pattern_end': '12:21'
    },
    {
        'date': datetime(2025, 3, 13),
        'original_detection': '15:30',
        'original_lag_hours': 3.1,
        'pattern_start': '11:34',
        'pattern_end': '12:21'
    }
]


def fetch_day_data(ticker: str, target_date: datetime):
    """Fetch full day data plus context for a specific date."""
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

    # Fetch from start of target day to end
    start_time = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 9, 30))
    end_time = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 16, 0))

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
        return None

    df['timestamp'] = df['timestamp'].dt.tz_convert(eastern)
    df = df.set_index('timestamp')
    df = df.between_time('09:30', '15:59')
    df = df.reset_index()

    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })

    if 'symbol' in df.columns:
        df = df.drop(columns=['symbol'])

    return df


def run_fixed_window_detection(day_df: pd.DataFrame, yolo: YOLOPatternDetector,
                                temp_dir: Path, fixed_window: int = 150,
                                step_interval: int = 10, min_confidence: float = 0.20):
    """
    Run detection with fixed window size.

    Returns list of all W_Bottom detections with times and confidences.
    """
    detections = []

    # Start after we have enough bars for the window
    start_idx = max(fixed_window, 60)  # At least 60 bars (10:30 AM)

    for end_idx in range(start_idx, len(day_df), step_interval):
        # Fixed window: last N bars
        start_window_idx = max(0, end_idx - fixed_window + 1)
        window_df = day_df.iloc[start_window_idx:end_idx + 1].copy()

        current_time = window_df['timestamp'].iloc[-1]

        # Generate chart
        chart_df = window_df.copy()
        chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp']).dt.tz_localize(None)
        chart_df.set_index('timestamp', inplace=True)
        chart_df = chart_df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })

        chart_path = temp_dir / f"chart_{end_idx:04d}.png"

        try:
            fig, _ = mpf.plot(
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

            # Run YOLO
            dets = yolo.detect(str(chart_path))

            # Check for W_Bottom
            for det in dets:
                if det['class_name'] == 'W_Bottom' and det['confidence'] >= min_confidence:
                    detections.append({
                        'time': current_time,
                        'bar': end_idx,
                        'confidence': det['confidence']
                    })

            # Cleanup
            os.remove(chart_path)

        except Exception as e:
            print(f"  Error at bar {end_idx}: {e}")
            continue

    return detections


def test_single_date(date_info: dict, yolo: YOLOPatternDetector):
    """Test a single date and return results."""
    target_date = date_info['date']
    date_str = target_date.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"Testing {date_str}")
    print(f"Original detection: {date_info['original_detection']} ({date_info['original_lag_hours']}h lag)")
    print(f"Pattern: {date_info['pattern_start']} - {date_info['pattern_end']}")
    print(f"{'='*60}")

    # Fetch data
    print("Fetching data...")
    day_df = fetch_day_data("QQQ", target_date)

    if day_df is None or len(day_df) == 0:
        print(f"ERROR: No data for {date_str}")
        return None

    print(f"Got {len(day_df)} bars")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Run fixed window detection
        print("Running fixed window (150 bars) detection...")
        detections = run_fixed_window_detection(
            day_df, yolo, temp_dir,
            fixed_window=150, step_interval=5, min_confidence=0.20
        )

        # Find first detection above 40% (backtest threshold)
        first_40pct = None
        for det in detections:
            if det['confidence'] >= 0.40:
                first_40pct = det
                break

        # Find first detection above 25%
        first_25pct = detections[0] if detections else None

        # Find peak confidence
        peak = max(detections, key=lambda x: x['confidence']) if detections else None

        # Calculate improvements
        original_time_str = date_info['original_detection']
        original_hour, original_min = map(int, original_time_str.split(':'))

        result = {
            'date': date_str,
            'original_detection': original_time_str,
            'original_lag_hours': date_info['original_lag_hours'],
            'pattern_end': date_info['pattern_end'],
            'total_detections': len(detections),
            'first_25pct': None,
            'first_40pct': None,
            'peak': None,
            'improvement_hours': None
        }

        if first_25pct:
            time_str = first_25pct['time'].strftime('%H:%M')
            result['first_25pct'] = {
                'time': time_str,
                'confidence': first_25pct['confidence']
            }
            print(f"  First 25%+ detection: {time_str} ({first_25pct['confidence']:.1%})")

        if first_40pct:
            time_str = first_40pct['time'].strftime('%H:%M')
            result['first_40pct'] = {
                'time': time_str,
                'confidence': first_40pct['confidence']
            }
            print(f"  First 40%+ detection: {time_str} ({first_40pct['confidence']:.1%})")

            # Calculate improvement
            det_hour, det_min = map(int, time_str.split(':'))
            det_minutes = det_hour * 60 + det_min
            orig_minutes = original_hour * 60 + original_min
            improvement = (orig_minutes - det_minutes) / 60
            result['improvement_hours'] = improvement
            print(f"  Improvement vs original: {improvement:.1f} hours earlier")

        if peak:
            time_str = peak['time'].strftime('%H:%M')
            result['peak'] = {
                'time': time_str,
                'confidence': peak['confidence']
            }
            print(f"  Peak confidence: {time_str} ({peak['confidence']:.1%})")

        if not detections:
            print("  NO W_Bottom detections found!")

        return result

    finally:
        # Cleanup
        try:
            for f in temp_dir.glob('*'):
                f.unlink()
            temp_dir.rmdir()
        except:
            pass


def main():
    print("=" * 70)
    print("FIXED WINDOW VALIDATION TEST")
    print("Testing 150-bar fixed window on 5 worst detection lag dates")
    print("=" * 70)

    # Initialize YOLO once
    print("\nInitializing YOLO detector...")
    yolo = YOLOPatternDetector(confidence=0.20)

    results = []

    for date_info in TEST_DATES:
        result = test_single_date(date_info, yolo)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Date | Original | Fixed Window (40%) | Improvement |")
    print("|------|----------|-------------------|-------------|")

    total_improvement = 0
    improved_count = 0

    for r in results:
        orig = r['original_detection']
        if r['first_40pct']:
            fixed = r['first_40pct']['time']
            imp = f"+{r['improvement_hours']:.1f}h"
            total_improvement += r['improvement_hours']
            improved_count += 1
        else:
            fixed = "N/A"
            imp = "N/A"

        print(f"| {r['date']} | {orig} | {fixed} | {imp} |")

    if improved_count > 0:
        avg_improvement = total_improvement / improved_count
        print(f"\nAverage improvement: {avg_improvement:.1f} hours earlier")
        print(f"Dates improved: {improved_count}/{len(results)}")

    # Save results
    output_path = Path("./fixed_window_validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
