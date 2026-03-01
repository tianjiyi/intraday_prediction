"""
Analyze HMM State Distribution by Hour

Goal: Understand how regimes evolve throughout each day
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market_regime.feature_pipeline import FeaturePipeline
from market_regime.regime_detector import RegimeDetector

logging.basicConfig(level=logging.WARNING)


def main():
    symbol = 'QQQ'
    start_date = date(2025, 2, 20)
    end_date = date(2025, 3, 4)
    volume_bar_size = 50000
    model_path = Path(__file__).parent / 'market_regime' / 'models' / f'hmm_regime_volume_{symbol}_{volume_bar_size}.pkl'

    # Load model and features
    detector = RegimeDetector.load(str(model_path))
    state_labels = detector.get_model_info()['state_labels']

    pipeline = FeaturePipeline(
        symbol=symbol, bucket_volume=volume_bar_size, vpin_buckets=50,
        bar_type='volume', volume_bar_size=volume_bar_size
    )

    features = pipeline.compute_features(
        start_date=start_date, end_date=end_date,
        timeframe='volume', rth_only=True, use_cache=True
    )

    # Get predictions
    hmm_features = pipeline.normalize_for_hmm(features)
    predictions = detector.predict(hmm_features)

    features['hmm_state'] = predictions
    features['state_label'] = [state_labels[s] for s in predictions]
    features['date'] = features.index.date
    features['hour'] = features.index.hour

    print("=" * 80)
    print("HOURLY HMM STATE DISTRIBUTION")
    print("=" * 80)
    print(f"\nState Labels: {state_labels}")

    trading_days = sorted(features['date'].unique())

    # Create figure
    fig, axes = plt.subplots(len(trading_days), 1, figsize=(16, 3*len(trading_days)))

    for idx, day in enumerate(trading_days):
        ax = axes[idx]
        day_data = features[features['date'] == day].copy()
        day_name = day.strftime('%a %m/%d')

        # Price change
        price_change = (day_data['close'].iloc[-1] - day_data['close'].iloc[0]) / day_data['close'].iloc[0] * 100

        print(f"\n{day_name} ({price_change:+.2f}%):")
        print("-" * 60)

        # Overall distribution for the day
        state_counts = day_data['state_label'].value_counts()
        total = len(day_data)
        print("  Overall distribution:")
        for state, count in state_counts.items():
            print(f"    {state}: {count/total*100:.1f}%")

        # Plot price with colored background by state
        x = np.arange(len(day_data))
        ax.plot(x, day_data['close'].values, 'k-', linewidth=1, zorder=10)

        # Color background by state
        state_colors = {
            'Bull Trend': 'green',
            'Bear Trend': 'red',
            'Range': 'gray',
            'Stress/Reversal': 'orange'
        }

        current_state = None
        start_idx = 0
        for i, state in enumerate(day_data['state_label'].values):
            if state != current_state:
                if current_state is not None:
                    ax.axvspan(start_idx, i, alpha=0.3, color=state_colors.get(current_state, 'white'))
                current_state = state
                start_idx = i
        # Last segment
        ax.axvspan(start_idx, len(day_data), alpha=0.3, color=state_colors.get(current_state, 'white'))

        ax.set_ylabel(f'{day_name}\n{price_change:+.2f}%', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add legend on first plot
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=c, alpha=0.3, label=s) for s, c in state_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Hourly breakdown
        print("  Hourly breakdown:")
        for hour in sorted(day_data['hour'].unique()):
            hour_data = day_data[day_data['hour'] == hour]
            if len(hour_data) == 0:
                continue

            hour_counts = hour_data['state_label'].value_counts()
            dominant_state = hour_counts.index[0]
            dominant_pct = hour_counts.iloc[0] / len(hour_data) * 100

            # Calculate price change for this hour
            hour_price_change = (hour_data['close'].iloc[-1] - hour_data['close'].iloc[0]) / hour_data['close'].iloc[0] * 100

            # State summary string
            state_str = ", ".join([f"{s[:4]}:{c/len(hour_data)*100:.0f}%" for s, c in hour_counts.head(2).items()])
            print(f"    {hour}:00 - {state_str} | Price: {hour_price_change:+.2f}%")

    axes[-1].set_xlabel('Volume Bar Index', fontsize=11)
    plt.suptitle(f'QQQ HMM State Distribution by Day ({start_date} to {end_date})', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = Path(__file__).parent / f'hmm_rth_{start_date.strftime("%m%d")}_{end_date.strftime("%m%d")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
