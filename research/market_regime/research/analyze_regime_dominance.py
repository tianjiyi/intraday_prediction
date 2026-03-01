"""
Analyze Trading Signals using Regime Dominance Factor

Regime Dominance: Rolling distribution of HMM states over past N buckets.
- Much more stable than raw probabilities
- Filters out noise during choppy periods
- Only trades when there's clear regime dominance

Trading Rule:
- Open Short: Bearish_Ratio > 0.6 AND OFI_ZScore < -1.0
- Close Short: Bearish_Ratio < 0.3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    dominance_window = 100  # Look at past 100 volume buckets (~10min of high activity)
    entry_threshold = 0.7   # 70% bearish dominance required
    exit_threshold = 0.4    # Exit when drops below 40%
    persistence_bars = 10   # Must stay above threshold for 10 bars
    model_path = Path(__file__).parent / 'market_regime' / 'models' / f'hmm_regime_volume_{symbol}_{volume_bar_size}.pkl'

    print("=" * 70)
    print("Regime Dominance Analysis: Nov 17-21, 2025")
    print("=" * 70)
    print(f"Dominance Window: {dominance_window} volume buckets")
    print(f"Entry Threshold: {entry_threshold} (70% bearish)")
    print(f"Exit Threshold: {exit_threshold} (40% bearish)")
    print(f"Persistence Required: {persistence_bars} consecutive bars")

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

    # Initialize Signal Generator
    print("\nComputing Regime Dominance...")
    generator = SignalGenerator(
        ema_span=5,
        entry_threshold=0.6,  # Bearish ratio > 60%
        exit_threshold=0.3,   # Bearish ratio < 30%
        ofi_zscore_threshold=-1.0,
        ofi_zscore_window=50
    )

    # Compute regime dominance
    dominance = generator.compute_regime_dominance(
        predictions=predictions,
        state_labels=state_labels,
        window=dominance_window
    )

    # Add to features
    features['bear_ratio'] = dominance['bear_ratio']
    features['bull_ratio'] = dominance['bull_ratio']
    features['stress_ratio'] = dominance['stress_ratio']
    features['bearish_ratio'] = dominance['bearish_ratio']
    features['dominance'] = dominance['dominance']

    # Compute OFI Z-Score
    ofi_zscore = generator.compute_ofi_zscore(features['ofi'])
    features['ofi_zscore'] = ofi_zscore.values

    # Generate trading signals based on regime dominance
    print("\nGenerating trading signals based on Regime Dominance...")
    signals = generate_dominance_signals(
        features=features,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        ofi_threshold=-1.0,
        persistence_bars=persistence_bars
    )

    features['position'] = signals['position']
    features['signal'] = signals['signal']

    # Print signal summary
    n_shorts = (signals['signal'] == 'SHORT').sum()
    n_covers = (signals['signal'] == 'COVER').sum()
    print(f"\nSignal Summary:")
    print(f"  SHORT signals: {n_shorts}")
    print(f"  COVER signals: {n_covers}")

    # Compute trade statistics
    print("\nComputing trade statistics...")
    stats = compute_trade_stats(signals, features['close'])
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
            entry_time = features.index[trade['entry_idx']]
            exit_time = features.index[trade['exit_idx']]
            print(f"  Trade {i+1}:")
            print(f"    Entry: {entry_time} @ ${trade['entry_price']:.2f}")
            print(f"    Exit:  {exit_time} @ ${trade['exit_price']:.2f}")
            print(f"    P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")

    # Create visualization
    print("\nCreating visualization...")
    create_dominance_chart(features, signals, stats, start_date, end_date, dominance_window)

    print("\nAnalysis complete!")


def generate_dominance_signals(
    features: pd.DataFrame,
    entry_threshold: float = 0.7,
    exit_threshold: float = 0.4,
    ofi_threshold: float = -1.0,
    persistence_bars: int = 10
) -> pd.DataFrame:
    """
    Generate trading signals based on regime dominance with persistence requirement.

    Persistence requirement: The bearish_ratio must stay above entry_threshold
    for `persistence_bars` consecutive bars before triggering a SHORT signal.
    This filters out brief spikes.
    """

    n = len(features)
    positions = np.zeros(n, dtype=int)
    signals = [None] * n

    in_position = False
    bars_above_threshold = 0

    for i in range(n):
        bearish_ratio = features['bearish_ratio'].iloc[i]
        ofi_z = features['ofi_zscore'].iloc[i] if not pd.isna(features['ofi_zscore'].iloc[i]) else 0

        if not in_position:
            # Check if above entry threshold
            if bearish_ratio > entry_threshold:
                bars_above_threshold += 1
            else:
                bars_above_threshold = 0  # Reset counter

            # Entry: Must be above threshold for persistence_bars AND OFI confirmation
            if bars_above_threshold >= persistence_bars and ofi_z < ofi_threshold:
                in_position = True
                signals[i] = 'SHORT'
                bars_above_threshold = 0  # Reset
                logger.debug(f"Bar {i}: SHORT - Bearish={bearish_ratio:.2f}, OFI_Z={ofi_z:.2f}")
        else:
            # Exit: Bearish dominance drops below exit threshold
            if bearish_ratio < exit_threshold:
                in_position = False
                signals[i] = 'COVER'
                logger.debug(f"Bar {i}: COVER - Bearish={bearish_ratio:.2f}")

        positions[i] = 1 if in_position else 0

    return pd.DataFrame({
        'position': positions,
        'signal': signals
    })


def compute_trade_stats(signals: pd.DataFrame, prices: pd.Series) -> dict:
    """Compute trade statistics."""
    trades = []
    entry_price = None
    entry_idx = None

    for i in range(len(signals)):
        if signals['signal'].iloc[i] == 'SHORT':
            entry_price = prices.iloc[i]
            entry_idx = i
        elif signals['signal'].iloc[i] == 'COVER' and entry_price is not None:
            exit_price = prices.iloc[i]
            pnl = entry_price - exit_price  # Short profit
            pnl_pct = (pnl / entry_price) * 100
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            entry_price = None
            entry_idx = None

    if not trades:
        return {'n_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl_pct': 0,
                'max_win_pct': 0, 'max_loss_pct': 0, 'trades': []}

    wins = sum(1 for t in trades if t['pnl'] > 0)
    return {
        'n_trades': len(trades),
        'total_pnl': sum(t['pnl'] for t in trades),
        'win_rate': wins / len(trades) * 100,
        'avg_pnl_pct': np.mean([t['pnl_pct'] for t in trades]),
        'max_win_pct': max(t['pnl_pct'] for t in trades),
        'max_loss_pct': min(t['pnl_pct'] for t in trades),
        'trades': trades
    }


def create_dominance_chart(features, signals, stats, start_date, end_date, window):
    """Create visualization with regime dominance."""

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # Get unique trading days
    features['date'] = features.index.date
    trading_days = features['date'].unique()

    # Create continuous index
    x = np.arange(len(features))

    # Find day boundaries
    day_starts = []
    current_day = None
    for i, d in enumerate(features['date']):
        if d != current_day:
            day_starts.append((i, d))
            current_day = d

    # Panel 1: Price with signals
    ax1 = axes[0]
    ax1.plot(x, features['close'].values, 'k-', linewidth=1, label='Price')

    # Add SHORT arrows
    short_mask = signals['signal'] == 'SHORT'
    if short_mask.any():
        short_idx = np.where(short_mask)[0]
        short_prices = features['close'].iloc[short_idx].values
        ax1.scatter(short_idx, short_prices, marker='v', s=250, c='red',
                   edgecolors='darkred', linewidth=2, label='SHORT', zorder=5)
        for idx in short_idx:
            ax1.axvline(x=idx, color='red', alpha=0.4, linestyle='--', linewidth=1)

    # Add COVER arrows
    cover_mask = signals['signal'] == 'COVER'
    if cover_mask.any():
        cover_idx = np.where(cover_mask)[0]
        cover_prices = features['close'].iloc[cover_idx].values
        ax1.scatter(cover_idx, cover_prices, marker='^', s=250, c='lime',
                   edgecolors='darkgreen', linewidth=2, label='COVER', zorder=5)
        for idx in cover_idx:
            ax1.axvline(x=idx, color='green', alpha=0.4, linestyle='--', linewidth=1)

    # Shade short positions
    position = signals['position'].values
    in_short = False
    short_start = None
    for i in range(len(position)):
        if position[i] == 1 and not in_short:
            in_short = True
            short_start = i
        elif position[i] == 0 and in_short:
            ax1.axvspan(short_start, i, alpha=0.2, color='red')
            in_short = False
    if in_short:
        ax1.axvspan(short_start, len(position), alpha=0.2, color='red')

    # Add day separators and labels
    for start_idx, day in day_starts:
        ax1.axvline(x=start_idx, color='blue', alpha=0.6, linestyle='-', linewidth=1.5)
        day_name = day.strftime('%a %m/%d')
        ax1.text(start_idx + 50, ax1.get_ylim()[1] if ax1.get_ylim()[1] else features['close'].max(),
                day_name, fontsize=10, color='blue', va='top', fontweight='bold')

    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title(f'QQQ Regime Dominance Trading: {start_date} to {end_date}\n'
                  f'Window={window} buckets | Trades: {stats["n_trades"]}, '
                  f'Win Rate: {stats.get("win_rate", 0):.1f}%, '
                  f'Total P&L: ${stats.get("total_pnl", 0):.2f}', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Regime Dominance (Bearish Ratio)
    ax2 = axes[1]

    # Plot bearish ratio (Bear + Stress)
    ax2.fill_between(x, 0, features['bearish_ratio'].values, alpha=0.4, color='red',
                     label='Bearish (Bear+Stress)')
    ax2.plot(x, features['bearish_ratio'].values, 'darkred', linewidth=1.5)

    # Plot bullish ratio
    ax2.plot(x, features['bull_ratio'].values, 'green', linewidth=1.5, alpha=0.7,
             label='Bull Ratio')

    # Thresholds
    ax2.axhline(y=0.6, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label='Entry threshold (0.6)')
    ax2.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.8,
                label='Exit threshold (0.3)')

    # Shade entry zone
    ax2.fill_between(x, 0.6, 1.0, alpha=0.1, color='red')
    ax2.fill_between(x, 0.0, 0.3, alpha=0.1, color='green')

    # Day separators
    for start_idx, _ in day_starts:
        ax2.axvline(x=start_idx, color='blue', alpha=0.6, linestyle='-', linewidth=1.5)

    ax2.set_ylabel(f'Regime Dominance\n(Rolling {window} buckets)', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: OFI Z-Score
    ax3 = axes[2]
    ax3.fill_between(x, 0, features['ofi_zscore'].values,
                     where=features['ofi_zscore'].values >= 0,
                     alpha=0.5, color='green', label='Positive OFI')
    ax3.fill_between(x, 0, features['ofi_zscore'].values,
                     where=features['ofi_zscore'].values < 0,
                     alpha=0.5, color='red', label='Negative OFI')
    ax3.axhline(y=-1.0, color='darkred', linestyle='--', linewidth=2, alpha=0.8,
                label='Entry threshold (-1.0)')
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    for start_idx, _ in day_starts:
        ax3.axvline(x=start_idx, color='blue', alpha=0.6, linestyle='-', linewidth=1.5)

    ax3.set_ylabel('OFI Z-Score', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: VPIN
    ax4 = axes[3]
    ax4.plot(x, features['vpin'].values, 'orange', linewidth=1.5, label='VPIN')
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    for start_idx, _ in day_starts:
        ax4.axvline(x=start_idx, color='blue', alpha=0.6, linestyle='-', linewidth=1.5)

    ax4.set_ylabel('VPIN', fontsize=11)
    ax4.set_xlabel('Volume Bar Index', fontsize=11)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent / 'nov17_21_regime_dominance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
