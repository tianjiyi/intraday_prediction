"""
Analyze Sell Put HMM Regime Detection

Goal: Validate VPIN-based toxicity detection on Nov 17-21 data
Focus: Detect Toxic states BEFORE major crashes, identify Safe/Reversal entry points

Usage:
    python analyze_sell_put_regime.py [--start-date 2025-11-17] [--end-date 2025-11-21]
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import date
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from market_regime.feature_pipeline import FeaturePipeline
from market_regime.regime_detector import RegimeDetector

logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='Analyze Sell Put HMM Regime Detection')
    parser.add_argument('--start-date', type=str, default='2025-11-17', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-21', help='End date (YYYY-MM-DD)')
    parser.add_argument('--volume-bar-size', type=int, default=250000, help='Volume bar size')
    args = parser.parse_args()

    symbol = 'QQQ'
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    volume_bar_size = args.volume_bar_size

    model_path = Path(__file__).parent / 'market_regime' / 'models' / f'hmm_regime_volume_{symbol}_{volume_bar_size}.pkl'

    print("=" * 80)
    print("SELL PUT REGIME ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Volume bar size: {volume_bar_size:,}")
    print(f"Model: {model_path}")

    # Load model
    print("\nLoading model...")
    detector = RegimeDetector.load(str(model_path))
    state_labels = detector.get_model_info()['state_labels']
    print(f"State labels: {state_labels}")

    # Initialize pipeline
    pipeline = FeaturePipeline(
        symbol=symbol, bucket_volume=volume_bar_size, vpin_buckets=50,
        bar_type='volume', volume_bar_size=volume_bar_size
    )

    # Compute features
    print("\nComputing features...")
    features = pipeline.compute_features(
        start_date=start_date, end_date=end_date,
        timeframe='volume', rth_only=True, use_cache=True
    )

    if features.empty:
        print("ERROR: No features computed!")
        return

    print(f"Total volume bars: {len(features)}")

    # Get predictions
    hmm_features = pipeline.normalize_for_hmm(features)
    predictions = detector.predict(hmm_features)
    probabilities = detector.predict_proba(hmm_features)

    features['hmm_state'] = predictions
    features['state_label'] = [state_labels[s] for s in predictions]
    features['state_prob'] = [probabilities[i, predictions[i]] for i in range(len(predictions))]
    features['date'] = features.index.date
    features['hour'] = features.index.hour

    # Add VPIN-based toxicity (threshold approach)
    vpin_p25 = features['vpin'].quantile(0.25)
    vpin_p50 = features['vpin'].quantile(0.50)
    vpin_p75 = features['vpin'].quantile(0.75)
    vpin_p90 = features['vpin'].quantile(0.90)

    def vpin_toxicity(vpin):
        if vpin > vpin_p90:
            return 'DANGER'
        elif vpin > vpin_p75:
            return 'CAUTION'
        elif vpin < vpin_p25:
            return 'SAFE'
        else:
            return 'NEUTRAL'

    features['vpin_toxicity'] = features['vpin'].apply(vpin_toxicity)

    print("\n" + "=" * 60)
    print("VPIN PERCENTILE THRESHOLDS")
    print("=" * 60)
    print(f"  P25 (Safe threshold):     {vpin_p25:.4f}")
    print(f"  P50 (Median):             {vpin_p50:.4f}")
    print(f"  P75 (Caution threshold):  {vpin_p75:.4f}")
    print(f"  P90 (Danger threshold):   {vpin_p90:.4f}")

    # Overall statistics
    print("\n" + "=" * 60)
    print("VPIN TOXICITY DISTRIBUTION (Threshold-Based)")
    print("=" * 60)

    total = len(features)
    tox_counts = Counter(features['vpin_toxicity'])
    for level in ['SAFE', 'NEUTRAL', 'CAUTION', 'DANGER']:
        count = tox_counts.get(level, 0)
        pct = count / total * 100
        print(f"  {level:8s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("HMM STATE DISTRIBUTION")
    print("=" * 60)

    state_counts = Counter(features['state_label'])
    total = len(features)
    for state, count in state_counts.items():
        pct = count / total * 100
        avg_vpin = features[features['state_label'] == state]['vpin'].mean()
        avg_ofi = features[features['state_label'] == state]['ofi'].mean()
        print(f"  {state:10s}: {count:4d} ({pct:5.1f}%) | VPIN: {avg_vpin:.3f} | OFI: {avg_ofi:+.0f}")

    # State colors for visualization
    state_colors = {
        'Safe': 'green',
        'Toxic': 'red',
        'Reversal': 'orange'
    }

    # Get trading days
    trading_days = sorted(features['date'].unique())

    # Create figure with 2 rows per day: price + VPIN
    n_days = len(trading_days)
    fig, axes = plt.subplots(n_days * 2, 1, figsize=(18, 4 * n_days),
                              gridspec_kw={'height_ratios': [2, 1] * n_days})

    print("\n" + "=" * 60)
    print("DAILY BREAKDOWN")
    print("=" * 60)

    for idx, day in enumerate(trading_days):
        ax_price = axes[idx * 2]
        ax_vpin = axes[idx * 2 + 1]

        day_data = features[features['date'] == day].copy()
        day_name = day.strftime('%a %m/%d')

        # Price change
        price_open = day_data['open'].iloc[0]
        price_close = day_data['close'].iloc[-1]
        price_change = (price_close - price_open) / price_open * 100

        print(f"\n{day_name} ({price_change:+.2f}%):")
        print("-" * 50)

        # Day distribution
        day_counts = Counter(day_data['state_label'])
        day_total = len(day_data)
        for state, count in day_counts.items():
            pct = count / day_total * 100
            print(f"  {state:10s}: {count:4d} ({pct:5.1f}%)")

        # Find toxic periods
        toxic_periods = []
        in_toxic = False
        toxic_start = None
        for i, (ts, row) in enumerate(day_data.iterrows()):
            if row['state_label'] == 'Toxic' and not in_toxic:
                in_toxic = True
                toxic_start = ts
            elif row['state_label'] != 'Toxic' and in_toxic:
                in_toxic = False
                toxic_periods.append((toxic_start, ts))
                toxic_start = None
        if in_toxic:
            toxic_periods.append((toxic_start, day_data.index[-1]))

        if toxic_periods:
            print(f"  Toxic periods: {len(toxic_periods)}")
            for start, end in toxic_periods[:3]:  # Show first 3
                duration = (end - start).total_seconds() / 60
                print(f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')} ({duration:.0f} min)")

        # Plot price with regime background
        x = np.arange(len(day_data))
        ax_price.plot(x, day_data['close'].values, 'k-', linewidth=1.5, zorder=10)

        # Color background by state
        current_state = None
        start_idx = 0
        for i, state in enumerate(day_data['state_label'].values):
            if state != current_state:
                if current_state is not None:
                    ax_price.axvspan(start_idx, i, alpha=0.3, color=state_colors.get(current_state, 'white'))
                current_state = state
                start_idx = i
        ax_price.axvspan(start_idx, len(day_data), alpha=0.3, color=state_colors.get(current_state, 'white'))

        ax_price.set_ylabel(f'{day_name}\n{price_change:+.2f}%', fontsize=11)
        ax_price.grid(True, alpha=0.3)
        ax_price.set_xlim(0, len(day_data))

        # Add legend on first plot
        if idx == 0:
            legend_elements = [Patch(facecolor=c, alpha=0.3, label=s) for s, c in state_colors.items()]
            ax_price.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Plot VPIN with toxicity zones
        vpin_max = day_data['vpin'].max()
        ax_vpin.set_ylim(0, max(vpin_max * 1.1, vpin_p90 * 1.5))  # Set y-limits first

        ax_vpin.plot(x, day_data['vpin'].values, 'b-', linewidth=1.5, label='VPIN', zorder=10)

        # Color zones first (background)
        ax_vpin.axhspan(vpin_p90, ax_vpin.get_ylim()[1], alpha=0.2, color='red', label='_')
        ax_vpin.axhspan(vpin_p75, vpin_p90, alpha=0.1, color='orange', label='_')
        ax_vpin.axhspan(0, vpin_p25, alpha=0.1, color='green', label='_')

        # Add toxicity threshold lines (using overall percentiles)
        ax_vpin.axhline(y=vpin_p90, color='darkred', linestyle='-', linewidth=2, alpha=0.7, label=f'P90: {vpin_p90:.3f}')
        ax_vpin.axhline(y=vpin_p75, color='red', linestyle='--', alpha=0.5, label=f'P75: {vpin_p75:.3f}')
        ax_vpin.axhline(y=vpin_p25, color='green', linestyle='--', alpha=0.5, label=f'P25: {vpin_p25:.3f}')

        ax_vpin.set_ylabel('VPIN', fontsize=10)
        ax_vpin.grid(True, alpha=0.3)
        ax_vpin.set_xlim(0, len(day_data))
        ax_vpin.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Volume Bar Index', fontsize=11)
    plt.suptitle(f'QQQ Sell Put Regime Analysis ({start_date} to {end_date})\n'
                 f'Green=Safe, Orange=Reversal, Red=Toxic', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = Path(__file__).parent / f'sell_put_regime_{start_date.strftime("%m%d")}_{end_date.strftime("%m%d")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {save_path}")
    plt.close()

    # Print key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR SELL PUT")
    print("=" * 60)

    # Find largest price drops and check VPIN toxicity before
    for day in trading_days:
        day_data = features[features['date'] == day].copy()

        # Calculate rolling price change (looking for crashes)
        day_data['price_change'] = day_data['close'].pct_change(10) * 100  # 10-bar change

        crashes = day_data[day_data['price_change'] < -0.5]  # >0.5% drop in 10 bars

        if len(crashes) > 0:
            print(f"\n{day.strftime('%a %m/%d')} - Found {len(crashes)} significant drops:")
            for ts, crash in crashes.head(5).iterrows():
                # Look back 10 bars before crash
                crash_idx = day_data.index.get_loc(ts)
                if crash_idx >= 10:
                    lookback = day_data.iloc[crash_idx-10:crash_idx]
                    avg_vpin_before = lookback['vpin'].mean()
                    max_vpin_before = lookback['vpin'].max()
                    danger_pct = (lookback['vpin_toxicity'] == 'DANGER').sum() / len(lookback) * 100
                    caution_pct = ((lookback['vpin_toxicity'] == 'CAUTION') | (lookback['vpin_toxicity'] == 'DANGER')).sum() / len(lookback) * 100

                    # Rating based on VPIN
                    if max_vpin_before > vpin_p90:
                        warning = "[!!!] HIGH VPIN WARNING"
                    elif max_vpin_before > vpin_p75:
                        warning = "[!] Elevated VPIN"
                    else:
                        warning = "[OK] Low VPIN (surprise)"

                    print(f"  {ts.strftime('%H:%M')} drop {crash['price_change']:.2f}%: "
                          f"VPIN avg:{avg_vpin_before:.3f} max:{max_vpin_before:.3f} "
                          f"DANGER:{danger_pct:.0f}% {warning}")


if __name__ == "__main__":
    main()
