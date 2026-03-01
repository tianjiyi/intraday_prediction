"""
Analyze Regime Stability with Different Window Sizes

Goal: Find the right window size for stable regime detection that:
1. Shows clear "Strong Bear" on Thursday (crash day)
2. Shows "Choppy/Range" on Monday/Tuesday
3. Doesn't flip-flop constantly
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
from market_regime.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Parameters
    symbol = 'QQQ'
    start_date = date(2025, 11, 17)
    end_date = date(2025, 11, 21)
    volume_bar_size = 10000
    model_path = Path(__file__).parent / 'market_regime' / 'models' / f'hmm_regime_volume_{symbol}_{volume_bar_size}.pkl'

    # Different window sizes to compare
    windows = [20, 50, 100, 200, 500]

    print("=" * 70)
    print("Regime Stability Analysis: Nov 17-21, 2025")
    print("=" * 70)

    # Load model
    print(f"\nLoading HMM model...")
    detector = RegimeDetector.load(str(model_path))
    state_labels = detector.get_model_info()['state_labels']
    print(f"State labels: {state_labels}")

    # Initialize pipeline
    print(f"\nLoading features...")
    pipeline = FeaturePipeline(
        symbol=symbol,
        bucket_volume=volume_bar_size,
        vpin_buckets=50,
        bar_type='volume',
        volume_bar_size=volume_bar_size
    )

    features = pipeline.compute_features(
        start_date=start_date,
        end_date=end_date,
        timeframe='volume',
        rth_only=True,
        use_cache=True
    )
    print(f"Loaded {len(features)} volume bars")

    # Get HMM predictions
    print("\nRunning HMM inference...")
    hmm_features = pipeline.normalize_for_hmm(features)
    predictions = detector.predict(hmm_features)

    # Initialize signal generator
    generator = SignalGenerator()

    # Compute regime dominance for each window size
    print("\nComputing regime dominance for different windows...")
    dominance_results = {}
    for w in windows:
        dom = generator.compute_regime_dominance(predictions, state_labels, window=w)
        dominance_results[w] = dom
        print(f"  Window {w}: Bear ratio range [{dom['bearish_ratio'].min():.2f}, {dom['bearish_ratio'].max():.2f}]")

    # Add to features
    features['date'] = features.index.date
    features['hmm_state'] = predictions

    # Create visualization
    create_stability_chart(features, dominance_results, windows, start_date, end_date)


def create_stability_chart(features, dominance_results, windows, start_date, end_date):
    """Create multi-panel chart comparing different window sizes."""

    n_panels = len(windows) + 1  # +1 for price
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 3 * n_panels), sharex=True)

    # Get day boundaries
    trading_days = features['date'].unique()
    day_starts = []
    current_day = None
    for i, d in enumerate(features['date']):
        if d != current_day:
            day_starts.append((i, d))
            current_day = d

    x = np.arange(len(features))

    # Panel 0: Price
    ax0 = axes[0]
    ax0.plot(x, features['close'].values, 'k-', linewidth=1)

    # Color background by day
    colors = ['#f0f0f0', '#ffffff']
    for i, (start_idx, day) in enumerate(day_starts):
        end_idx = day_starts[i+1][0] if i+1 < len(day_starts) else len(x)
        ax0.axvspan(start_idx, end_idx, alpha=0.3, color=colors[i % 2])
        day_name = day.strftime('%a %m/%d')
        ax0.text(start_idx + (end_idx - start_idx) / 2, features['close'].max(),
                day_name, fontsize=11, ha='center', va='bottom', fontweight='bold')

    ax0.set_ylabel('Price ($)', fontsize=10)
    ax0.set_title(f'QQQ Regime Stability Analysis: {start_date} to {end_date}', fontsize=14)
    ax0.grid(True, alpha=0.3)

    # Panels 1+: Different window sizes
    for idx, w in enumerate(windows):
        ax = axes[idx + 1]
        dom = dominance_results[w]

        bearish = dom['bearish_ratio']
        bullish = dom['bull_ratio']

        # Plot bearish ratio
        ax.fill_between(x, 0, bearish, alpha=0.4, color='red', label='Bearish (Bear+Stress)')
        ax.plot(x, bearish, 'darkred', linewidth=1)

        # Plot bullish ratio
        ax.plot(x, bullish, 'green', linewidth=1, alpha=0.7, label='Bull')

        # Thresholds
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=0.6, color='orange', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.6)

        # Day separators
        for start_idx, _ in day_starts:
            ax.axvline(x=start_idx, color='blue', alpha=0.5, linestyle='-', linewidth=1)

        # Calculate stability metrics
        transitions = np.sum(np.abs(np.diff(bearish > 0.6)))
        time_above_60 = np.mean(bearish > 0.6) * 100
        time_above_70 = np.mean(bearish > 0.7) * 100

        ax.set_ylabel(f'Window={w}\n({w*10/1000:.0f}K vol)', fontsize=9)
        ax.set_ylim(0, 1)
        ax.text(0.02, 0.95, f'Crossings(0.6): {transitions} | >60%: {time_above_60:.1f}% | >70%: {time_above_70:.1f}%',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Volume Bar Index', fontsize=11)

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent / 'nov17_21_regime_stability.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {save_path}")

    # Also create a daily breakdown
    create_daily_breakdown(features, dominance_results, windows)

    plt.show()


def create_daily_breakdown(features, dominance_results, windows):
    """Print daily regime statistics."""

    print("\n" + "=" * 70)
    print("DAILY REGIME BREAKDOWN")
    print("=" * 70)

    trading_days = features['date'].unique()

    for w in [100, 200, 500]:
        print(f"\n--- Window = {w} buckets ---")
        dom = dominance_results[w]
        bearish = dom['bearish_ratio']

        for day in trading_days:
            mask = features['date'] == day
            day_bearish = bearish[mask]

            avg_bearish = np.mean(day_bearish)
            max_bearish = np.max(day_bearish)
            time_above_60 = np.mean(day_bearish > 0.6) * 100
            time_above_70 = np.mean(day_bearish > 0.7) * 100

            # Price change
            day_prices = features.loc[mask, 'close']
            price_change = (day_prices.iloc[-1] - day_prices.iloc[0]) / day_prices.iloc[0] * 100

            day_name = day.strftime('%a %m/%d')
            print(f"  {day_name}: Avg={avg_bearish:.2f}, Max={max_bearish:.2f}, "
                  f">60%={time_above_60:.0f}%, >70%={time_above_70:.0f}%, "
                  f"Price={price_change:+.2f}%")


if __name__ == "__main__":
    main()
