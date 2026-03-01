"""
Signal Generation Layer for HMM-Based Trading Signals

This module implements a signal smoothing and state machine layer that converts
noisy HMM regime probabilities into clean, actionable trading signals.

Key Components:
1. Bear Score: P(Bear) + P(Stress) - aggregated bearish probability
2. EMA Smoothing: Reduces noise in probability signals
3. Hysteresis State Machine: Wide entry (0.7), strict exit (0.3) to prevent whipsaws
4. OFI Z-Score Filter: Additional confirmation for short entries
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Position(Enum):
    """Trading position state."""
    FLAT = 0
    SHORT = 1


class SignalGenerator:
    """
    Signal Generation Layer with Hysteresis State Machine.

    Converts HMM regime probabilities into trading signals using:
    - Bear Score = P(Bear) + P(Stress)
    - EMA smoothing for noise reduction
    - Hysteresis logic: easy entry (0.7), hard exit (0.3)
    - OFI Z-Score filter for confirmation

    Usage:
        generator = SignalGenerator(
            ema_span=5,
            entry_threshold=0.7,
            exit_threshold=0.3,
            ofi_zscore_threshold=-1.0
        )
        signals = generator.generate_signals(probabilities_df, ofi_series)
    """

    def __init__(
        self,
        ema_span: int = 5,
        entry_threshold: float = 0.7,
        exit_threshold: float = 0.3,
        ofi_zscore_threshold: float = -1.0,
        ofi_zscore_window: int = 50
    ):
        """
        Initialize Signal Generator.

        Args:
            ema_span: Span for EMA smoothing of Bear Score
            entry_threshold: Bear Score threshold to enter short (must exceed)
            exit_threshold: Bear Score threshold to exit short (must fall below)
            ofi_zscore_threshold: OFI Z-Score threshold for entry confirmation
            ofi_zscore_window: Rolling window for OFI Z-Score calculation
        """
        self.ema_span = ema_span
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.ofi_zscore_threshold = ofi_zscore_threshold
        self.ofi_zscore_window = ofi_zscore_window

        # State machine
        self._position = Position.FLAT
        self._entry_bar = None
        self._entry_price = None

    def reset(self):
        """Reset state machine to flat position."""
        self._position = Position.FLAT
        self._entry_bar = None
        self._entry_price = None

    def compute_bear_score(
        self,
        probabilities: np.ndarray,
        state_labels: Dict[int, str],
        use_net_score: bool = True
    ) -> np.ndarray:
        """
        Compute Bear Score from HMM probabilities.

        Two modes:
        1. Net Score (default): [P(Bear) + P(Stress)] - [P(Bull) + P(Range)]
           - Range: -1.0 (very bullish) to +1.0 (very bearish)
           - Centered at 0 for neutral markets

        2. Raw Score: P(Bear) + P(Stress)
           - Range: 0.0 to 1.0
           - Problem: Often 0.5+ even in neutral markets

        Args:
            probabilities: (N, n_states) probability matrix from HMM
            state_labels: Mapping from state_id to label name
            use_net_score: If True, compute net score (bear - bull)

        Returns:
            (N,) array of Bear Scores
        """
        # Find state indices
        bear_idx = None
        stress_idx = None
        bull_idx = None
        range_idx = None

        for state_id, label in state_labels.items():
            label_lower = label.lower()
            if 'bear' in label_lower:
                bear_idx = state_id
            elif 'stress' in label_lower or 'reversal' in label_lower:
                stress_idx = state_id
            elif 'bull' in label_lower:
                bull_idx = state_id
            elif 'range' in label_lower:
                range_idx = state_id

        if bear_idx is None or stress_idx is None:
            raise ValueError(
                f"Could not find Bear and Stress states in labels: {state_labels}"
            )

        logger.info(f"State indices - Bear: {bear_idx}, Stress: {stress_idx}, "
                   f"Bull: {bull_idx}, Range: {range_idx}")

        # Compute bearish probability
        p_bearish = probabilities[:, bear_idx] + probabilities[:, stress_idx]

        if use_net_score and bull_idx is not None:
            # Net score: bearish - bullish
            p_bullish = probabilities[:, bull_idx]
            if range_idx is not None:
                p_bullish = p_bullish + probabilities[:, range_idx]

            bear_score = p_bearish - p_bullish
            logger.info(f"Using NET Bear Score: range [{bear_score.min():.2f}, {bear_score.max():.2f}]")
        else:
            # Raw score (original method)
            bear_score = p_bearish
            logger.info(f"Using RAW Bear Score: range [{bear_score.min():.2f}, {bear_score.max():.2f}]")

        return bear_score

    def smooth_bear_score(self, bear_score: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to Bear Score.

        Args:
            bear_score: Raw Bear Score array

        Returns:
            EMA-smoothed Bear Score
        """
        series = pd.Series(bear_score)
        smoothed = series.ewm(span=self.ema_span, adjust=False).mean()
        return smoothed.values

    def compute_regime_dominance(
        self,
        predictions: np.ndarray,
        state_labels: Dict[int, str],
        window: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Compute Regime Dominance - rolling state distribution over past N buckets.

        Instead of looking at current state probabilities (noisy), look at
        the actual state distribution over the past N volume buckets.

        This is much more stable and filters out noise effectively.

        Args:
            predictions: (N,) array of predicted state IDs
            state_labels: Mapping from state_id to label name
            window: Number of past buckets to consider (default: 20)

        Returns:
            Dict with:
            - 'bear_ratio': Rolling % of buckets in Bear state
            - 'bull_ratio': Rolling % of buckets in Bull state
            - 'stress_ratio': Rolling % of buckets in Stress state
            - 'range_ratio': Rolling % of buckets in Range state
            - 'dominance': 'Strong Bear', 'Strong Bull', or 'Choppy'
        """
        n = len(predictions)

        # Find state indices
        bear_idx = None
        bull_idx = None
        stress_idx = None
        range_idx = None

        for state_id, label in state_labels.items():
            label_lower = label.lower()
            if 'bear' in label_lower:
                bear_idx = state_id
            elif 'bull' in label_lower:
                bull_idx = state_id
            elif 'stress' in label_lower or 'reversal' in label_lower:
                stress_idx = state_id
            elif 'range' in label_lower:
                range_idx = state_id

        logger.info(f"Regime Dominance - Window: {window} buckets")
        logger.info(f"State indices - Bear: {bear_idx}, Bull: {bull_idx}, "
                   f"Stress: {stress_idx}, Range: {range_idx}")

        # Initialize arrays
        bear_ratio = np.zeros(n)
        bull_ratio = np.zeros(n)
        stress_ratio = np.zeros(n)
        range_ratio = np.zeros(n)
        dominance = ['Choppy'] * n

        # Compute rolling ratios
        for i in range(n):
            # Get window of past states
            start_idx = max(0, i - window + 1)
            window_states = predictions[start_idx:i+1]
            window_size = len(window_states)

            # Count states in window
            if bear_idx is not None:
                bear_ratio[i] = np.sum(window_states == bear_idx) / window_size
            if bull_idx is not None:
                bull_ratio[i] = np.sum(window_states == bull_idx) / window_size
            if stress_idx is not None:
                stress_ratio[i] = np.sum(window_states == stress_idx) / window_size
            if range_idx is not None:
                range_ratio[i] = np.sum(window_states == range_idx) / window_size

            # Determine dominance
            # Combine Bear + Stress for bearish dominance
            bearish_ratio = bear_ratio[i] + stress_ratio[i]
            bullish_ratio = bull_ratio[i]

            if bearish_ratio > 0.6:
                dominance[i] = 'Strong Bear'
            elif bullish_ratio > 0.6:
                dominance[i] = 'Strong Bull'
            else:
                dominance[i] = 'Choppy'

        # Log summary
        strong_bear_pct = sum(1 for d in dominance if d == 'Strong Bear') / n * 100
        strong_bull_pct = sum(1 for d in dominance if d == 'Strong Bull') / n * 100
        choppy_pct = sum(1 for d in dominance if d == 'Choppy') / n * 100

        logger.info(f"Regime Distribution: Strong Bear={strong_bear_pct:.1f}%, "
                   f"Strong Bull={strong_bull_pct:.1f}%, Choppy={choppy_pct:.1f}%")

        return {
            'bear_ratio': bear_ratio,
            'bull_ratio': bull_ratio,
            'stress_ratio': stress_ratio,
            'range_ratio': range_ratio,
            'bearish_ratio': bear_ratio + stress_ratio,  # Combined bearish
            'dominance': dominance
        }

    def compute_ofi_zscore(self, ofi: pd.Series) -> pd.Series:
        """
        Compute rolling Z-Score of OFI.

        Args:
            ofi: OFI series

        Returns:
            Z-Score series
        """
        rolling_mean = ofi.rolling(window=self.ofi_zscore_window, min_periods=1).mean()
        rolling_std = ofi.rolling(window=self.ofi_zscore_window, min_periods=1).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-10)

        zscore = (ofi - rolling_mean) / rolling_std
        return zscore

    def generate_signals(
        self,
        probabilities: np.ndarray,
        state_labels: Dict[int, str],
        ofi: pd.Series,
        prices: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate trading signals from HMM probabilities and OFI.

        Args:
            probabilities: (N, n_states) probability matrix from HMM
            state_labels: Mapping from state_id to label name
            ofi: OFI series (same length as probabilities)
            prices: Optional price series for P&L tracking

        Returns:
            DataFrame with columns:
            - bear_score: Raw P(Bear) + P(Stress)
            - bear_score_ema: Smoothed Bear Score
            - ofi_zscore: OFI Z-Score
            - position: Current position (0=flat, 1=short)
            - signal: 'SHORT', 'COVER', or None
        """
        self.reset()

        n_bars = len(probabilities)

        # Compute indicators
        bear_score = self.compute_bear_score(probabilities, state_labels)
        bear_score_ema = self.smooth_bear_score(bear_score)
        ofi_zscore = self.compute_ofi_zscore(ofi)

        # Initialize result arrays
        positions = np.zeros(n_bars, dtype=int)
        signals = [None] * n_bars

        # Run state machine
        for i in range(n_bars):
            current_bear = bear_score_ema[i]
            current_ofi_z = ofi_zscore.iloc[i] if not pd.isna(ofi_zscore.iloc[i]) else 0

            if self._position == Position.FLAT:
                # Check entry conditions: Bear Score > 0.7 AND OFI Z-Score < -1.0
                if (current_bear > self.entry_threshold and
                    current_ofi_z < self.ofi_zscore_threshold):
                    self._position = Position.SHORT
                    self._entry_bar = i
                    if prices is not None:
                        self._entry_price = prices.iloc[i]
                    signals[i] = 'SHORT'
                    logger.debug(
                        f"Bar {i}: SHORT signal - Bear={current_bear:.3f}, OFI_Z={current_ofi_z:.2f}"
                    )

            elif self._position == Position.SHORT:
                # Check exit condition: Bear Score < 0.3
                if current_bear < self.exit_threshold:
                    self._position = Position.FLAT
                    signals[i] = 'COVER'
                    logger.debug(
                        f"Bar {i}: COVER signal - Bear={current_bear:.3f}"
                    )
                    self._entry_bar = None
                    self._entry_price = None

            positions[i] = self._position.value

        # Build result DataFrame
        result = pd.DataFrame({
            'bear_score': bear_score,
            'bear_score_ema': bear_score_ema,
            'ofi_zscore': ofi_zscore.values,
            'position': positions,
            'signal': signals
        })

        if prices is not None:
            result['price'] = prices.values

        # Log summary
        n_shorts = sum(1 for s in signals if s == 'SHORT')
        n_covers = sum(1 for s in signals if s == 'COVER')
        logger.info(
            f"Generated {n_shorts} SHORT and {n_covers} COVER signals "
            f"(entry={self.entry_threshold}, exit={self.exit_threshold})"
        )

        return result

    def compute_trade_stats(
        self,
        signals_df: pd.DataFrame,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute trade statistics from signals.

        Args:
            signals_df: DataFrame from generate_signals()
            prices: Price series

        Returns:
            Dict with trade statistics
        """
        trades = []
        entry_price = None
        entry_idx = None

        for i, row in signals_df.iterrows():
            if row['signal'] == 'SHORT':
                entry_price = prices.iloc[i] if isinstance(i, int) else prices.loc[i]
                entry_idx = i
            elif row['signal'] == 'COVER' and entry_price is not None:
                exit_price = prices.iloc[i] if isinstance(i, int) else prices.loc[i]
                # Short profit = entry - exit (profit when price goes down)
                pnl = entry_price - exit_price
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
            return {
                'n_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'trades': []
            }

        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades])

        return {
            'n_trades': len(trades),
            'total_pnl': total_pnl,
            'win_rate': wins / len(trades) * 100,
            'avg_pnl_pct': avg_pnl_pct,
            'max_win_pct': max(t['pnl_pct'] for t in trades),
            'max_loss_pct': min(t['pnl_pct'] for t in trades),
            'trades': trades
        }


def create_signal_chart(
    features_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    title: str = "HMM Trading Signals",
    save_path: str = None
):
    """
    Create a chart showing price with trading signals.

    Args:
        features_df: DataFrame with OHLC and features
        signals_df: DataFrame from SignalGenerator.generate_signals()
        title: Chart title
        save_path: Path to save the chart
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Ensure same index
    if len(features_df) != len(signals_df):
        logger.warning(
            f"Length mismatch: features={len(features_df)}, signals={len(signals_df)}"
        )
        min_len = min(len(features_df), len(signals_df))
        features_df = features_df.iloc[:min_len]
        signals_df = signals_df.iloc[:min_len]

    # Use sequential index for plotting (to avoid gaps)
    x = np.arange(len(features_df))

    # Panel 1: Price with signals
    ax1 = axes[0]
    ax1.plot(x, features_df['close'].values, 'k-', linewidth=1, label='Price')

    # Add SHORT arrows (red down)
    short_mask = signals_df['signal'] == 'SHORT'
    if short_mask.any():
        short_idx = np.where(short_mask)[0]
        short_prices = features_df['close'].iloc[short_idx].values
        ax1.scatter(short_idx, short_prices, marker='v', s=150, c='red',
                   edgecolors='darkred', linewidth=1, label='SHORT', zorder=5)

    # Add COVER arrows (green up)
    cover_mask = signals_df['signal'] == 'COVER'
    if cover_mask.any():
        cover_idx = np.where(cover_mask)[0]
        cover_prices = features_df['close'].iloc[cover_idx].values
        ax1.scatter(cover_idx, cover_prices, marker='^', s=150, c='lime',
                   edgecolors='darkgreen', linewidth=1, label='COVER', zorder=5)

    # Shade short positions
    position = signals_df['position'].values
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

    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Bear Score with thresholds
    ax2 = axes[1]
    ax2.plot(x, signals_df['bear_score'].values, 'gray', alpha=0.5,
             linewidth=0.5, label='Raw Bear Score')
    ax2.plot(x, signals_df['bear_score_ema'].values, 'purple',
             linewidth=1.5, label=f'EMA({5}) Bear Score')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Entry (0.7)')
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Exit (0.3)')
    ax2.fill_between(x, 0.7, 1.0, alpha=0.1, color='red')
    ax2.fill_between(x, 0.0, 0.3, alpha=0.1, color='green')
    ax2.set_ylabel('Bear Score')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: OFI Z-Score with threshold
    ax3 = axes[2]
    ax3.plot(x, signals_df['ofi_zscore'].values, 'blue', linewidth=1)
    ax3.axhline(y=-1.0, color='red', linestyle='--', alpha=0.7, label='Entry threshold (-1.0)')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.fill_between(x, -1.0, signals_df['ofi_zscore'].values,
                     where=signals_df['ofi_zscore'].values < -1.0,
                     alpha=0.3, color='red')
    ax3.set_ylabel('OFI Z-Score')
    ax3.set_xlabel('Bar Index')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Chart saved to: {save_path}")

    return fig, axes


if __name__ == "__main__":
    # Test with synthetic data
    import logging
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n = 500

    # Simulate HMM probabilities (4 states)
    probs = np.random.dirichlet([1, 1, 1, 1], n)

    # Add some structure - trending bear in middle
    probs[200:300, 1] += 0.4  # Boost Bear state
    probs[200:300, 2] += 0.2  # Boost Stress state
    probs = probs / probs.sum(axis=1, keepdims=True)  # Renormalize

    state_labels = {0: 'Range', 1: 'Bear Trend', 2: 'Stress/Reversal', 3: 'Bull Trend'}

    # Simulate OFI (negative during bear trend)
    ofi = pd.Series(np.random.randn(n) * 1000)
    ofi[200:300] -= 2000  # Negative during bear

    # Simulate price (declining during bear)
    price = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.1))
    price[200:300] -= np.linspace(0, 5, 100)  # Decline

    # Test signal generator
    generator = SignalGenerator()
    signals = generator.generate_signals(probs, state_labels, ofi, price)

    print("\nSignals Summary:")
    print(f"  Total bars: {len(signals)}")
    print(f"  SHORT signals: {(signals['signal'] == 'SHORT').sum()}")
    print(f"  COVER signals: {(signals['signal'] == 'COVER').sum()}")

    # Compute trade stats
    stats = generator.compute_trade_stats(signals, price)
    print(f"\nTrade Statistics:")
    print(f"  Trades: {stats['n_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Avg P&L: {stats['avg_pnl_pct']:.2f}%")
