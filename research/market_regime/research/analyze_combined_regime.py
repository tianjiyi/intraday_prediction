"""
Combined Regime Indicator: HMM Net Bearish + VPIN + OFI

Hypothesis: The HMM alone is too noisy. Combine with:
1. VPIN > 0.5 = elevated informed trading
2. OFI cumulative trend = sustained selling pressure
3. HMM Net Bearish = regime confirmation
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

    # Get HMM probabilities
    hmm_features = pipeline.normalize_for_hmm(features)
    probabilities = detector.predict_proba(hmm_features)

    # Find state indices
    bear_idx = bull_idx = stress_idx = range_idx = None
    for state_id, label in state_labels.items():
        if 'bear' in label.lower(): bear_idx = state_id
        elif 'bull' in label.lower(): bull_idx = state_id
        elif 'stress' in label.lower(): stress_idx = state_id
        elif 'range' in label.lower(): range_idx = state_id

    # Compute Net Bearish
    net_bearish = (probabilities[:, bear_idx] + probabilities[:, stress_idx]) - \
                  (probabilities[:, bull_idx] + probabilities[:, range_idx])

    features['net_bearish'] = net_bearish
    features['net_bearish_ema200'] = pd.Series(net_bearish).ewm(span=200).mean().values
    features['date'] = features.index.date

    # VPIN smoothing
    features['vpin_ema50'] = features['vpin'].ewm(span=50).mean()

    # OFI cumulative (rolling sum)
    features['ofi_cum100'] = features['ofi'].rolling(100).sum()
    features['ofi_cum100_norm'] = features['ofi_cum100'] / features['ofi_cum100'].abs().rolling(500).mean()

    # Combined score:
    # 1. HMM Net Bearish (EMA200): range -1 to +1, map to 0-1
    # 2. VPIN: already 0-1, high = bearish
    # 3. OFI cumulative: negative = bearish

    # Normalize components to 0-1 scale where 1 = most bearish
    hmm_component = (features['net_bearish_ema200'] + 1) / 2  # -1,+1 -> 0,1
    vpin_component = features['vpin_ema50']  # Already 0-1
    ofi_component = 1 - (features['ofi_cum100_norm'].clip(-2, 2) + 2) / 4  # -2,+2 -> 1,0

    # Fill NaN
    hmm_component = hmm_component.fillna(0.5)
    vpin_component = vpin_component.fillna(0.5)
    ofi_component = ofi_component.fillna(0.5)

    # Combined score (weighted average)
    features['combined_bearish'] = (
        0.4 * hmm_component +
        0.3 * vpin_component +
        0.3 * ofi_component
    )

    # Smooth the combined score
    features['combined_bearish_ema100'] = features['combined_bearish'].ewm(span=100).mean()

    # Create visualization
    create_combined_chart(features, start_date, end_date)

    # Print daily summary
    print_daily_summary(features)


def create_combined_chart(features, start_date, end_date):
    """Create chart showing combined regime indicator."""

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

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
    ax0.set_title(f'QQQ Combined Regime Analysis: {start_date} to {end_date}', fontsize=14)
    ax0.grid(True, alpha=0.3)

    # Panel 1: HMM Net Bearish
    ax1 = axes[1]
    net_ema = features['net_bearish_ema200'].values
    ax1.fill_between(x, 0, net_ema, where=net_ema > 0, alpha=0.5, color='red')
    ax1.fill_between(x, 0, net_ema, where=net_ema <= 0, alpha=0.5, color='green')
    ax1.axhline(y=0, color='gray', linewidth=1)
    for start_idx, _ in day_starts:
        ax1.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
    ax1.set_ylabel('HMM Net Bearish\n(EMA200)', fontsize=10)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)

    # Panel 2: VPIN
    ax2 = axes[2]
    vpin = features['vpin_ema50'].values
    ax2.fill_between(x, 0.5, vpin, where=vpin > 0.5, alpha=0.5, color='red')
    ax2.fill_between(x, 0.5, vpin, where=vpin <= 0.5, alpha=0.5, color='green')
    ax2.plot(x, vpin, 'k-', linewidth=0.5)
    ax2.axhline(y=0.5, color='gray', linewidth=1)
    for start_idx, _ in day_starts:
        ax2.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
    ax2.set_ylabel('VPIN\n(EMA50)', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Panel 3: OFI Cumulative
    ax3 = axes[3]
    ofi_cum = features['ofi_cum100'].values
    ax3.fill_between(x, 0, ofi_cum, where=ofi_cum < 0, alpha=0.5, color='red')
    ax3.fill_between(x, 0, ofi_cum, where=ofi_cum >= 0, alpha=0.5, color='green')
    ax3.axhline(y=0, color='gray', linewidth=1)
    for start_idx, _ in day_starts:
        ax3.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
    ax3.set_ylabel('OFI\n(Sum 100 bars)', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Combined Score
    ax4 = axes[4]
    combined = features['combined_bearish_ema100'].values
    ax4.fill_between(x, 0.5, combined, where=combined > 0.5, alpha=0.6, color='red', label='Bearish')
    ax4.fill_between(x, 0.5, combined, where=combined <= 0.5, alpha=0.6, color='green', label='Bullish')
    ax4.plot(x, combined, 'k-', linewidth=1)
    ax4.axhline(y=0.5, color='gray', linewidth=1)
    ax4.axhline(y=0.6, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax4.axhline(y=0.4, color='green', linestyle='--', linewidth=1, alpha=0.7)
    for start_idx, _ in day_starts:
        ax4.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
    ax4.set_ylabel('Combined\nBearish Score', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Volume Bar Index', fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(__file__).parent / 'nov17_21_combined_regime.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {save_path}")
    plt.close()


def print_daily_summary(features):
    """Print daily summary."""

    print("\n" + "=" * 80)
    print("DAILY COMBINED REGIME SUMMARY")
    print("=" * 80)

    trading_days = sorted(features['date'].unique())

    for day in trading_days:
        day_data = features[features['date'] == day]
        combined = day_data['combined_bearish_ema100']

        price_change = (day_data['close'].iloc[-1] - day_data['close'].iloc[0]) / day_data['close'].iloc[0] * 100
        avg_vpin = day_data['vpin_ema50'].mean()
        avg_ofi = day_data['ofi_cum100'].mean()
        avg_hmm = day_data['net_bearish_ema200'].mean()
        avg_combined = combined.mean()

        time_bearish = (combined > 0.6).mean() * 100
        max_combined = combined.max()

        day_name = day.strftime('%a %m/%d')
        print(f"\n{day_name} (Price: {price_change:+.2f}%):")
        print(f"  HMM Net Bearish: {avg_hmm:+.2f}")
        print(f"  VPIN (EMA50):    {avg_vpin:.3f}")
        print(f"  OFI (Sum100):    {avg_ofi:+.0f}")
        print(f"  Combined Score:  {avg_combined:.2f} (Max: {max_combined:.2f})")
        print(f"  Time >0.6:       {time_bearish:.0f}%")


if __name__ == "__main__":
    main()
