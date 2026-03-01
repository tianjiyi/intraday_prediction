"""
Analyze OFI Divergence for Nov 17-21, 2025

Goal: Detect bearish divergence before Thursday's crash.
Key hypothesis: Price making new highs on Wed while OFI declining = bearish divergence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from market_regime.feature_pipeline import FeaturePipeline
from market_regime.ofi_divergence import OFIDivergenceDetector, Divergence

logging.basicConfig(level=logging.WARNING)


def main():
    symbol = 'QQQ'
    start_date = date(2025, 11, 17)
    end_date = date(2025, 11, 21)
    volume_bar_size = 10000

    # Load features
    print("Loading volume bar data...")
    pipeline = FeaturePipeline(
        symbol=symbol, bucket_volume=volume_bar_size, vpin_buckets=50,
        bar_type='volume', volume_bar_size=volume_bar_size
    )

    features = pipeline.compute_features(
        start_date=start_date, end_date=end_date,
        timeframe='volume', rth_only=True, use_cache=True
    )
    print(f"Loaded {len(features)} volume bars")

    # Add date column
    features['date'] = features.index.date

    # Initialize divergence detector with balanced parameters
    # Goal: Fewer, higher-quality signals
    detector = OFIDivergenceDetector(
        peak_window=40,           # Smoother peak detection
        ofi_cum_window=100,       # 100-bar rolling sum for cumulative OFI
        min_peak_distance=80,     # At least 80 bars between peaks to compare
        divergence_threshold=0.10, # OFI must decline by 10% of its range
        min_price_change_pct=0.15 # Price must move at least 0.15% between peaks
    )

    print(f"\nParameters:")
    print(f"  peak_window: {detector.peak_window}")
    print(f"  min_peak_distance: {detector.min_peak_distance}")
    print(f"  divergence_threshold: {detector.divergence_threshold}")
    print(f"  min_price_change_pct: {detector.min_price_change_pct}%")

    # Detect divergences
    price = features['close'].values
    ofi = features['ofi'].values

    print("\nDetecting divergences...")
    divergences, ofi_cum = detector.detect_all_divergences(price, ofi)
    features['ofi_cum'] = ofi_cum

    print(f"Found {len(divergences)} divergences")

    # Print divergence details
    print_divergence_details(features, divergences)

    # Compute continuous divergence score
    div_score = detector.compute_divergence_score(price, ofi, window=100)
    features['div_score'] = div_score

    # Create visualization
    create_divergence_chart(features, divergences, start_date, end_date)

    # Analyze divergence leading time
    analyze_leading_time(features, divergences)


def print_divergence_details(features, divergences):
    """Print details of detected divergences."""

    print("\n" + "=" * 80)
    print("DETECTED DIVERGENCES")
    print("=" * 80)

    bearish = [d for d in divergences if d.divergence_type == 'bearish']
    bullish = [d for d in divergences if d.divergence_type == 'bullish']

    print(f"\nBearish Divergences: {len(bearish)}")
    print(f"Bullish Divergences: {len(bullish)}")

    if bearish:
        print("\n--- Bearish Divergences (Price up, OFI down) ---")
        for d in bearish:
            timestamp = features.index[d.bar_idx]
            day = timestamp.strftime('%a %m/%d %H:%M')
            price_change = (d.price_peak2_val - d.price_peak1_val) / d.price_peak1_val * 100
            ofi_change = d.ofi_peak2_val - d.ofi_peak1_val
            print(f"  {day}: Price +{price_change:.2f}%, OFI {ofi_change:+.0f} | "
                  f"Strength: {d.strength:.2f} | "
                  f"Peak gap: {d.price_peak2_idx - d.price_peak1_idx} bars")

    if bullish:
        print("\n--- Bullish Divergences (Price down, OFI up) ---")
        for d in bullish:
            timestamp = features.index[d.bar_idx]
            day = timestamp.strftime('%a %m/%d %H:%M')
            price_change = (d.price_peak2_val - d.price_peak1_val) / d.price_peak1_val * 100
            ofi_change = d.ofi_peak2_val - d.ofi_peak1_val
            print(f"  {day}: Price {price_change:.2f}%, OFI {ofi_change:+.0f} | "
                  f"Strength: {d.strength:.2f}")


def create_divergence_chart(features, divergences, start_date, end_date):
    """Create chart showing price, OFI, and divergences."""

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    x = np.arange(len(features))
    features['date'] = features.index.date

    # Day boundaries
    day_starts = []
    current_day = None
    for i, d in enumerate(features['date']):
        if d != current_day:
            day_starts.append((i, d))
            current_day = d

    # Panel 0: Price with divergence markers
    ax0 = axes[0]
    ax0.plot(x, features['close'].values, 'k-', linewidth=1, label='Price')

    # Mark peaks and troughs
    from scipy.signal import find_peaks
    price = features['close'].values
    peaks, _ = find_peaks(price, distance=20)
    troughs, _ = find_peaks(-price, distance=20)

    ax0.scatter(peaks, price[peaks], c='red', marker='v', s=30, alpha=0.5, zorder=5, label='Peaks')
    ax0.scatter(troughs, price[troughs], c='green', marker='^', s=30, alpha=0.5, zorder=5, label='Troughs')

    # Mark divergence points
    for d in divergences:
        if d.divergence_type == 'bearish':
            ax0.scatter(d.bar_idx, price[d.bar_idx], c='darkred', marker='v', s=150, zorder=10)
            ax0.annotate('BEAR DIV', (d.bar_idx, price[d.bar_idx]),
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=8, color='darkred', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkred'))
        else:
            ax0.scatter(d.bar_idx, price[d.bar_idx], c='darkgreen', marker='^', s=150, zorder=10)
            ax0.annotate('BULL DIV', (d.bar_idx, price[d.bar_idx]),
                        xytext=(10, -30), textcoords='offset points',
                        fontsize=8, color='darkgreen', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkgreen'))

    for start_idx, day in day_starts:
        ax0.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)
        ax0.text(start_idx + 50, features['close'].max(), day.strftime('%a'),
                fontsize=10, color='blue', fontweight='bold')

    ax0.set_ylabel('Price', fontsize=10)
    ax0.set_title(f'QQQ OFI Divergence Analysis: {start_date} to {end_date}', fontsize=14)
    ax0.legend(loc='upper right', fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Panel 1: OFI Cumulative with peaks marked
    ax1 = axes[1]
    ofi_cum = features['ofi_cum'].values

    # Color by positive/negative
    ax1.fill_between(x, 0, ofi_cum, where=ofi_cum >= 0, alpha=0.4, color='green', label='Buying Pressure')
    ax1.fill_between(x, 0, ofi_cum, where=ofi_cum < 0, alpha=0.4, color='red', label='Selling Pressure')
    ax1.plot(x, ofi_cum, 'k-', linewidth=1)

    # Mark OFI at divergence points
    for d in divergences:
        color = 'darkred' if d.divergence_type == 'bearish' else 'darkgreen'
        # Draw line connecting the two peaks/troughs
        ax1.plot([d.price_peak1_idx, d.price_peak2_idx],
                [d.ofi_peak1_val, d.ofi_peak2_val],
                '--', color=color, linewidth=2, alpha=0.7)
        ax1.scatter([d.price_peak1_idx, d.price_peak2_idx],
                   [d.ofi_peak1_val, d.ofi_peak2_val],
                   c=color, s=80, zorder=10)

    ax1.axhline(y=0, color='gray', linewidth=1)
    for start_idx, _ in day_starts:
        ax1.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)

    ax1.set_ylabel('OFI Cumulative\n(Sum 100 bars)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Divergence Score (continuous)
    ax2 = axes[2]
    div_score = features['div_score'].values

    ax2.fill_between(x, 0, div_score, where=div_score > 0, alpha=0.5, color='red', label='Bearish Div')
    ax2.fill_between(x, 0, div_score, where=div_score < 0, alpha=0.5, color='green', label='Bullish Div')
    ax2.plot(x, div_score, 'k-', linewidth=0.5)

    ax2.axhline(y=0, color='gray', linewidth=1)
    ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Strong Bearish')
    ax2.axhline(y=-0.3, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Strong Bullish')

    for start_idx, _ in day_starts:
        ax2.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)

    ax2.set_ylabel('Divergence Score', fontsize=10)
    ax2.set_ylim(-1, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Raw OFI for comparison
    ax3 = axes[3]
    ofi_ema = features['ofi'].ewm(span=50).mean().values

    ax3.fill_between(x, 0, ofi_ema, where=ofi_ema >= 0, alpha=0.4, color='green')
    ax3.fill_between(x, 0, ofi_ema, where=ofi_ema < 0, alpha=0.4, color='red')
    ax3.plot(x, ofi_ema, 'k-', linewidth=0.5)
    ax3.axhline(y=0, color='gray', linewidth=1)

    for start_idx, _ in day_starts:
        ax3.axvline(x=start_idx, color='blue', alpha=0.5, linewidth=1)

    ax3.set_ylabel('Raw OFI\n(EMA50)', fontsize=10)
    ax3.set_xlabel('Volume Bar Index', fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(__file__).parent / 'nov17_21_ofi_divergence.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {save_path}")
    plt.close()


def analyze_leading_time(features, divergences):
    """Analyze how far ahead divergences predicted reversals."""

    print("\n" + "=" * 80)
    print("DIVERGENCE LEADING TIME ANALYSIS")
    print("=" * 80)

    # For each divergence, calculate price change in next N bars
    price = features['close'].values
    horizons = [50, 100, 200, 300]

    for d in divergences:
        timestamp = features.index[d.bar_idx]
        day = timestamp.strftime('%a %m/%d %H:%M')
        print(f"\n{d.divergence_type.upper()} at {day} (bar {d.bar_idx}):")

        current_price = price[d.bar_idx]

        for horizon in horizons:
            end_idx = min(d.bar_idx + horizon, len(price) - 1)
            if end_idx <= d.bar_idx:
                continue

            # Get price change over horizon
            future_prices = price[d.bar_idx:end_idx+1]
            min_price = np.min(future_prices)
            max_price = np.max(future_prices)
            end_price = price[end_idx]

            pct_change = (end_price - current_price) / current_price * 100
            max_gain = (max_price - current_price) / current_price * 100
            max_loss = (min_price - current_price) / current_price * 100

            if d.divergence_type == 'bearish':
                # For bearish, we expect price to drop
                success = "YES" if pct_change < -0.5 else "NO"
            else:
                # For bullish, we expect price to rise
                success = "YES" if pct_change > 0.5 else "NO"

            print(f"  +{horizon} bars: {pct_change:+.2f}% | "
                  f"Max: {max_gain:+.2f}% / Min: {max_loss:+.2f}% | "
                  f"Correct: {success}")


def print_daily_summary(features, divergences):
    """Print daily summary of divergence activity."""

    print("\n" + "=" * 80)
    print("DAILY OFI DIVERGENCE SUMMARY")
    print("=" * 80)

    trading_days = sorted(features['date'].unique())

    for day in trading_days:
        day_data = features[features['date'] == day]
        start_idx = day_data.index[0]
        end_idx = day_data.index[-1]

        # Get bar range for this day
        day_bar_start = features.index.get_loc(start_idx)
        day_bar_end = features.index.get_loc(end_idx)

        # Count divergences on this day
        day_divergences = [d for d in divergences
                         if day_bar_start <= d.bar_idx <= day_bar_end]

        bearish_count = len([d for d in day_divergences if d.divergence_type == 'bearish'])
        bullish_count = len([d for d in day_divergences if d.divergence_type == 'bullish'])

        # Price change
        price_change = (day_data['close'].iloc[-1] - day_data['close'].iloc[0]) / day_data['close'].iloc[0] * 100

        # OFI summary
        avg_ofi = day_data['ofi'].mean()
        ofi_cum_end = day_data['ofi_cum'].iloc[-1] if 'ofi_cum' in day_data.columns else 0

        day_name = day.strftime('%a %m/%d')
        print(f"\n{day_name} (Price: {price_change:+.2f}%):")
        print(f"  Bearish Divergences: {bearish_count}")
        print(f"  Bullish Divergences: {bullish_count}")
        print(f"  Avg OFI: {avg_ofi:+.0f}")
        print(f"  OFI Cumulative (EOD): {ofi_cum_end:+.0f}")


if __name__ == "__main__":
    main()
