"""
Analyze Regime using HMM Probabilities (Soft Classification)

Instead of hard state assignments, use the actual probabilities:
- Net_Bearish = P(Bear) + P(Stress) - P(Bull) - P(Range)
- Apply large rolling window for smoothing
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
    start_date = date(2025, 11, 17)
    end_date = date(2025, 11, 21)
    volume_bar_size = 10000
    model_path = Path(__file__).parent / 'market_regime' / 'models' / f'hmm_regime_volume_{symbol}_{volume_bar_size}.pkl'

    # Load model and features
    print("Loading model and features...")
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
    print(f"Loaded {len(features)} volume bars")

    # Get probabilities (soft classification)
    hmm_features = pipeline.normalize_for_hmm(features)
    probabilities = detector.predict_proba(hmm_features)

    # Find state indices
    bear_idx = bull_idx = stress_idx = range_idx = None
    for state_id, label in state_labels.items():
        if 'bear' in label.lower():
            bear_idx = state_id
        elif 'bull' in label.lower():
            bull_idx = state_id
        elif 'stress' in label.lower():
            stress_idx = state_id
        elif 'range' in label.lower():
            range_idx = state_id

    print(f"State indices - Bear:{bear_idx}, Bull:{bull_idx}, Stress:{stress_idx}, Range:{range_idx}")

    # Compute Net Bearish Score using probabilities
    # Net = (P(Bear) + P(Stress)) - (P(Bull) + P(Range))
    p_bearish = probabilities[:, bear_idx] + probabilities[:, stress_idx]
    p_bullish = probabilities[:, bull_idx] + probabilities[:, range_idx]
    net_bearish = p_bearish - p_bullish  # Range: -1 to +1

    features['p_bear'] = probabilities[:, bear_idx]
    features['p_bull'] = probabilities[:, bull_idx]
    features['p_stress'] = probabilities[:, stress_idx]
    features['p_range'] = probabilities[:, range_idx]
    features['net_bearish'] = net_bearish
    features['date'] = features.index.date

    # Apply different EMA spans for smoothing
    ema_spans = [20, 50, 100, 200, 500]
    for span in ema_spans:
        features[f'net_bearish_ema{span}'] = pd.Series(net_bearish).ewm(span=span).mean().values

    # Create visualization
    create_probability_chart(features, ema_spans, start_date, end_date)

    # Print daily summary
    print_daily_summary(features, ema_spans)


def create_probability_chart(features, ema_spans, start_date, end_date):
    """Create chart showing probability-based regime indicator."""

    n_panels = len(ema_spans) + 2  # +1 price, +1 raw probs
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 2.5 * n_panels), sharex=True)

    x = np.arange(len(features))

    # Day boundaries
    day_starts = []
    current_day = None
    for i, d in enumerate(features['date']):
        if d != current_day:
            day_starts.append((i, d))
            current_day = d

    # Panel 0: Price
    ax0 = axes[0]
    ax0.plot(x, features['close'].values, 'k-', linewidth=1)
    for start_idx, day in day_starts:
        ax0.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
        ax0.text(start_idx + 100, features['close'].max(), day.strftime('%a'),
                fontsize=10, color='blue', fontweight='bold')
    ax0.set_ylabel('Price', fontsize=10)
    ax0.set_title(f'QQQ Probability-Based Regime Analysis: {start_date} to {end_date}', fontsize=14)
    ax0.grid(True, alpha=0.3)

    # Panel 1: Raw state probabilities
    ax1 = axes[1]
    ax1.fill_between(x, 0, features['p_bear'].values, alpha=0.4, color='red', label='P(Bear)')
    ax1.fill_between(x, features['p_bear'].values,
                     features['p_bear'].values + features['p_stress'].values,
                     alpha=0.4, color='orange', label='P(Stress)')
    ax1.fill_between(x, features['p_bear'].values + features['p_stress'].values,
                     features['p_bear'].values + features['p_stress'].values + features['p_range'].values,
                     alpha=0.4, color='gray', label='P(Range)')
    ax1.fill_between(x, features['p_bear'].values + features['p_stress'].values + features['p_range'].values,
                     1, alpha=0.4, color='green', label='P(Bull)')
    ax1.set_ylabel('Raw Probs', fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', fontsize=8)
    for start_idx, _ in day_starts:
        ax1.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
    ax1.grid(True, alpha=0.3)

    # Panels 2+: EMA smoothed Net Bearish
    for idx, span in enumerate(ema_spans):
        ax = axes[idx + 2]
        net_ema = features[f'net_bearish_ema{span}'].values

        # Color based on value
        ax.fill_between(x, 0, net_ema, where=net_ema > 0, alpha=0.5, color='red', label='Bearish')
        ax.fill_between(x, 0, net_ema, where=net_ema <= 0, alpha=0.5, color='green', label='Bullish')
        ax.plot(x, net_ema, 'k-', linewidth=0.5)

        # Thresholds
        ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.8)
        ax.axhline(y=-0.3, color='green', linestyle='--', linewidth=1, alpha=0.6)

        for start_idx, _ in day_starts:
            ax.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)

        # Stats
        time_bearish = np.mean(net_ema > 0.3) * 100
        time_bullish = np.mean(net_ema < -0.3) * 100

        ax.set_ylabel(f'EMA({span})', fontsize=10)
        ax.set_ylim(-1, 1)
        ax.text(0.02, 0.95, f'>0.3: {time_bearish:.0f}% | <-0.3: {time_bullish:.0f}%',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Volume Bar Index', fontsize=11)
    plt.tight_layout()

    save_path = Path(__file__).parent / 'nov17_21_regime_probs.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {save_path}")
    plt.close()


def print_daily_summary(features, ema_spans):
    """Print daily regime summary."""

    print("\n" + "=" * 80)
    print("DAILY NET BEARISH SUMMARY")
    print("=" * 80)
    print("\nNet Bearish = P(Bear)+P(Stress) - P(Bull)-P(Range)")
    print("Range: -1 (very bullish) to +1 (very bearish)")

    trading_days = sorted(features['date'].unique())

    for span in [100, 200, 500]:
        print(f"\n--- EMA({span}) ---")
        for day in trading_days:
            day_data = features[features['date'] == day]
            net_ema = day_data[f'net_bearish_ema{span}']

            price_change = (day_data['close'].iloc[-1] - day_data['close'].iloc[0]) / day_data['close'].iloc[0] * 100

            avg_net = net_ema.mean()
            max_net = net_ema.max()
            min_net = net_ema.min()
            time_bearish = (net_ema > 0.3).mean() * 100
            time_bullish = (net_ema < -0.3).mean() * 100

            day_name = day.strftime('%a %m/%d')
            print(f"  {day_name}: Avg={avg_net:+.2f}, Max={max_net:+.2f}, Min={min_net:+.2f} | "
                  f">0.3: {time_bearish:.0f}%, <-0.3: {time_bullish:.0f}% | Price: {price_change:+.2f}%")


if __name__ == "__main__":
    main()
