"""
Analyze Trading Signals for Nov 17-21, 2025 using HMM + Signal Generator

This script:
1. Loads the volume-bar HMM model
2. Computes features for Nov 17-21
3. Generates trading signals using SignalGenerator
4. Creates visualization with Short/Cover arrows
5. Computes trade statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from market_regime.feature_pipeline import FeaturePipeline
from market_regime.regime_detector import RegimeDetector
from market_regime.signal_generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    # Parameters
    symbol = 'QQQ'
    start_date = date(2025, 11, 17)
    end_date = date(2025, 11, 21)
    volume_bar_size = 10000
    model_path = Path(__file__).parent / 'market_regime' / 'models' / f'hmm_regime_volume_{symbol}_{volume_bar_size}.pkl'

    print("=" * 70)
    print("HMM Signal Analysis: Nov 17-21, 2025")
    print("=" * 70)

    # Load model
    print(f"\nLoading HMM model from: {model_path}")
    detector = RegimeDetector.load(str(model_path))
    model_info = detector.get_model_info()
    state_labels = model_info['state_labels']
    print(f"State labels: {state_labels}")

    # Initialize pipeline
    print(f"\nInitializing feature pipeline for {symbol}...")
    pipeline = FeaturePipeline(
        symbol=symbol,
        bucket_volume=volume_bar_size,
        vpin_buckets=50,
        bar_type='volume',
        volume_bar_size=volume_bar_size
    )

    # Compute features
    print(f"Computing volume-bar features for {start_date} to {end_date}...")
    features = pipeline.compute_features(
        start_date=start_date,
        end_date=end_date,
        timeframe='volume',
        rth_only=True,
        use_cache=True
    )

    print(f"Loaded {len(features)} volume bars")

    # Normalize and predict
    print("\nRunning HMM inference...")
    hmm_features = pipeline.normalize_for_hmm(features)
    predictions = detector.predict(hmm_features)
    probabilities = detector.predict_proba(hmm_features)

    # Add predictions to features
    features['hmm_state'] = predictions
    features['hmm_state_label'] = [state_labels[s] for s in predictions]

    # Add probabilities for each state
    for state_id, label in state_labels.items():
        features[f'prob_{label.replace("/", "_").replace(" ", "_")}'] = probabilities[:, state_id]

    # Initialize Signal Generator
    print("\nInitializing Signal Generator...")
    generator = SignalGenerator(
        ema_span=5,
        entry_threshold=0.7,
        exit_threshold=0.3,
        ofi_zscore_threshold=-1.0,
        ofi_zscore_window=50
    )

    # Generate signals
    print("Generating trading signals...")
    signals = generator.generate_signals(
        probabilities=probabilities,
        state_labels=state_labels,
        ofi=features['ofi'],
        prices=features['close']
    )

    # Add signals to features
    features['bear_score'] = signals['bear_score'].values
    features['bear_score_ema'] = signals['bear_score_ema'].values
    features['ofi_zscore'] = signals['ofi_zscore'].values
    features['position'] = signals['position'].values
    features['signal'] = signals['signal'].values

    # Print signal summary
    n_shorts = (signals['signal'] == 'SHORT').sum()
    n_covers = (signals['signal'] == 'COVER').sum()
    print(f"\nSignal Summary:")
    print(f"  SHORT signals: {n_shorts}")
    print(f"  COVER signals: {n_covers}")

    # Compute trade statistics
    print("\nComputing trade statistics...")
    stats = generator.compute_trade_stats(signals, features['close'])
    print(f"\nTrade Statistics:")
    print(f"  Total trades: {stats['n_trades']}")
    if stats['n_trades'] > 0:
        print(f"  Win rate: {stats['win_rate']:.1f}%")
        print(f"  Average P&L: {stats['avg_pnl_pct']:.2f}%")
        print(f"  Max win: {stats['max_win_pct']:.2f}%")
        print(f"  Max loss: {stats['max_loss_pct']:.2f}%")
        print(f"  Total P&L (price points): {stats['total_pnl']:.2f}")

        print("\nIndividual Trades:")
        for i, trade in enumerate(stats['trades']):
            print(f"  Trade {i+1}:")
            print(f"    Entry: Bar {trade['entry_idx']}, Price: ${trade['entry_price']:.2f}")
            print(f"    Exit:  Bar {trade['exit_idx']}, Price: ${trade['exit_price']:.2f}")
            print(f"    P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")

    # Create visualization
    print("\nCreating visualization...")
    create_signal_visualization(features, signals, stats, start_date, end_date)

    print("\nAnalysis complete!")


def create_signal_visualization(features, signals, stats, start_date, end_date):
    """Create a multi-panel chart showing signals and indicators."""

    # Filter to RTH only and reset index for continuous plotting
    features_rth = features.copy()

    # Create figure with 4 panels
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # Get unique trading days
    features_rth['date'] = features_rth.index.date
    trading_days = features_rth['date'].unique()

    # Create continuous index
    x = np.arange(len(features_rth))

    # Find day boundaries for vertical lines
    day_starts = []
    current_day = None
    for i, d in enumerate(features_rth['date']):
        if d != current_day:
            day_starts.append((i, d))
            current_day = d

    # Panel 1: Price with trading signals
    ax1 = axes[0]
    ax1.plot(x, features_rth['close'].values, 'k-', linewidth=1, label='Price')

    # Add SHORT arrows (red down triangle)
    short_mask = signals['signal'] == 'SHORT'
    if short_mask.any():
        short_idx = np.where(short_mask)[0]
        short_prices = features_rth['close'].iloc[short_idx].values
        ax1.scatter(short_idx, short_prices, marker='v', s=200, c='red',
                   edgecolors='darkred', linewidth=2, label='SHORT', zorder=5)
        # Add vertical line
        for idx in short_idx:
            ax1.axvline(x=idx, color='red', alpha=0.3, linestyle='--', linewidth=1)

    # Add COVER arrows (green up triangle)
    cover_mask = signals['signal'] == 'COVER'
    if cover_mask.any():
        cover_idx = np.where(cover_mask)[0]
        cover_prices = features_rth['close'].iloc[cover_idx].values
        ax1.scatter(cover_idx, cover_prices, marker='^', s=200, c='lime',
                   edgecolors='darkgreen', linewidth=2, label='COVER', zorder=5)
        # Add vertical line
        for idx in cover_idx:
            ax1.axvline(x=idx, color='green', alpha=0.3, linestyle='--', linewidth=1)

    # Shade short positions
    position = signals['position'].values
    in_short = False
    short_start = None
    for i in range(len(position)):
        if position[i] == 1 and not in_short:
            in_short = True
            short_start = i
        elif position[i] == 0 and in_short:
            ax1.axvspan(short_start, i, alpha=0.15, color='red')
            in_short = False
    if in_short:
        ax1.axvspan(short_start, len(position), alpha=0.15, color='red')

    # Add day separators
    for start_idx, day in day_starts:
        ax1.axvline(x=start_idx, color='gray', alpha=0.5, linestyle='-', linewidth=1)
        # Add day label at top
        ax1.text(start_idx + 5, ax1.get_ylim()[1], day.strftime('%m/%d'),
                fontsize=9, color='gray', va='top')

    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title(f'QQQ Trading Signals: {start_date} to {end_date}\n'
                  f'Trades: {stats["n_trades"]}, Win Rate: {stats.get("win_rate", 0):.1f}%, '
                  f'Avg P&L: {stats.get("avg_pnl_pct", 0):.2f}%', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Bear Score with thresholds
    ax2 = axes[1]
    ax2.fill_between(x, 0, signals['bear_score'].values, alpha=0.3, color='gray', label='Raw Bear Score')
    ax2.plot(x, signals['bear_score_ema'].values, 'purple', linewidth=2, label='EMA(5) Bear Score')
    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Entry threshold (0.7)')
    ax2.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Exit threshold (0.3)')
    ax2.fill_between(x, 0.7, 1.0, alpha=0.15, color='red')
    ax2.fill_between(x, 0.0, 0.3, alpha=0.15, color='green')

    # Add day separators
    for start_idx, _ in day_starts:
        ax2.axvline(x=start_idx, color='gray', alpha=0.5, linestyle='-', linewidth=1)

    ax2.set_ylabel('Bear Score\n(P(Bear) + P(Stress))')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: OFI Z-Score
    ax3 = axes[2]
    ax3.fill_between(x, 0, signals['ofi_zscore'].values,
                     where=signals['ofi_zscore'].values >= 0,
                     alpha=0.4, color='green', label='Positive OFI')
    ax3.fill_between(x, 0, signals['ofi_zscore'].values,
                     where=signals['ofi_zscore'].values < 0,
                     alpha=0.4, color='red', label='Negative OFI')
    ax3.axhline(y=-1.0, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='Entry threshold (-1.0)')
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add day separators
    for start_idx, _ in day_starts:
        ax3.axvline(x=start_idx, color='gray', alpha=0.5, linestyle='-', linewidth=1)

    ax3.set_ylabel('OFI Z-Score')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: VPIN
    ax4 = axes[3]
    ax4.plot(x, features_rth['vpin'].values, 'orange', linewidth=1.5, label='VPIN')
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add day separators
    for start_idx, _ in day_starts:
        ax4.axvline(x=start_idx, color='gray', alpha=0.5, linestyle='-', linewidth=1)

    ax4.set_ylabel('VPIN')
    ax4.set_xlabel('Volume Bar Index')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent / 'nov17_21_trading_signals.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
