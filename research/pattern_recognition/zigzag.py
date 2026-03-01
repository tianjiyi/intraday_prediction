"""
High-performance ZigZag algorithm using Numba JIT compilation.

The ZigZag indicator filters out market noise by identifying significant
price swings. It produces a series of pivot points (peaks and valleys)
that represent major price reversals.

Performance: O(N) time complexity, bypasses Python GIL via Numba.

Reference: PDF Section 3.2 - Numba/Cython implementation path.
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Optional
import logging

from .models import Pivot, PivotType

logger = logging.getLogger(__name__)


# Deviation presets for different timeframes (from PDF Section 3.1)
# Smaller timeframes need smaller deviation to capture more swings
DEVIATION_PRESETS = {
    "1min": 0.005,    # 0.5%
    "5min": 0.008,    # 0.8%
    "15min": 0.01,    # 1.0%
    "30min": 0.012,   # 1.2%
    "1hour": 0.015,   # 1.5%
    "4hour": 0.02,    # 2.0%
    "daily": 0.03,    # 3.0%
    "weekly": 0.05,   # 5.0%
}


@jit(nopython=True)
def _zigzag_core(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    deviation: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated ZigZag core algorithm.

    This function uses a state machine to track price swings:
    - State 0: Looking for initial direction
    - State 1: In uptrend (looking for peak confirmation)
    - State -1: In downtrend (looking for valley confirmation)

    A pivot is confirmed when price reverses by more than the deviation threshold.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        deviation: Percentage threshold (e.g., 0.01 = 1%)

    Returns:
        pivot_indices: Array of pivot bar indices
        pivot_values: Array of pivot prices
        pivot_types: Array of pivot types (1=Peak, -1=Valley)
    """
    n = len(closes)

    # Pre-allocate arrays (max possible pivots = n)
    pivot_indices = np.empty(n, dtype=np.int32)
    pivot_values = np.empty(n, dtype=np.float64)
    pivot_types = np.empty(n, dtype=np.int8)

    if n < 2:
        return pivot_indices[:0], pivot_values[:0], pivot_types[:0]

    count = 0

    # State: 0=looking for initial direction, 1=in uptrend, -1=in downtrend
    trend = 0

    # Track extremes in current swing
    swing_high_idx = 0
    swing_high_val = highs[0]
    swing_low_idx = 0
    swing_low_val = lows[0]

    for i in range(1, n):
        current_high = highs[i]
        current_low = lows[i]

        if trend == 0:  # Looking for initial direction
            # Update extremes while searching
            if current_high > swing_high_val:
                swing_high_idx = i
                swing_high_val = current_high
            if current_low < swing_low_val:
                swing_low_idx = i
                swing_low_val = current_low

            # Check for uptrend start (rally from low exceeds deviation)
            if current_high > swing_low_val * (1 + deviation):
                # First pivot is a valley (the low before the rally)
                pivot_indices[count] = swing_low_idx
                pivot_values[count] = swing_low_val
                pivot_types[count] = -1  # Valley
                count += 1

                trend = 1  # Now in uptrend
                swing_high_idx = i
                swing_high_val = current_high

            # Check for downtrend start (drop from high exceeds deviation)
            elif current_low < swing_high_val * (1 - deviation):
                # First pivot is a peak (the high before the drop)
                pivot_indices[count] = swing_high_idx
                pivot_values[count] = swing_high_val
                pivot_types[count] = 1  # Peak
                count += 1

                trend = -1  # Now in downtrend
                swing_low_idx = i
                swing_low_val = current_low

        elif trend == 1:  # In uptrend - looking for peak confirmation
            # New high - extend the swing
            if current_high > swing_high_val:
                swing_high_idx = i
                swing_high_val = current_high

            # Check for reversal (drop exceeds deviation from swing high)
            elif current_low < swing_high_val * (1 - deviation):
                # Confirm peak at swing_high
                pivot_indices[count] = swing_high_idx
                pivot_values[count] = swing_high_val
                pivot_types[count] = 1  # Peak
                count += 1

                # Start downtrend from current bar
                trend = -1
                swing_low_idx = i
                swing_low_val = current_low

        elif trend == -1:  # In downtrend - looking for valley confirmation
            # New low - extend the swing
            if current_low < swing_low_val:
                swing_low_idx = i
                swing_low_val = current_low

            # Check for reversal (rally exceeds deviation from swing low)
            elif current_high > swing_low_val * (1 + deviation):
                # Confirm valley at swing_low
                pivot_indices[count] = swing_low_idx
                pivot_values[count] = swing_low_val
                pivot_types[count] = -1  # Valley
                count += 1

                # Start uptrend from current bar
                trend = 1
                swing_high_idx = i
                swing_high_val = current_high

    return pivot_indices[:count], pivot_values[:count], pivot_types[:count]


def compute_zigzag(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    deviation: float = 0.01,
    timestamps: Optional[np.ndarray] = None
) -> List[Pivot]:
    """
    Compute ZigZag pivots from OHLC data.

    Args:
        highs: High prices array
        lows: Low prices array
        closes: Close prices array
        deviation: Deviation threshold (default 1%)
        timestamps: Optional array of timestamps for pivot labeling

    Returns:
        List of Pivot objects

    Deviation Guidelines (from PDF Section 3.1):
        - Intraday (1min-30min): 0.5% - 1.5% (0.005 - 0.015)
        - Daily: 3% - 5% (0.03 - 0.05)

    Example:
        >>> pivots = compute_zigzag(df['high'].values, df['low'].values, df['close'].values)
        >>> for p in pivots:
        ...     print(f"Index {p.index}: {p.pivot_type.name} at ${p.price:.2f}")
    """
    # Ensure arrays are float64 for Numba
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)

    # Run Numba-optimized core
    indices, values, types = _zigzag_core(highs, lows, closes, deviation)

    # Convert to Pivot objects
    pivots = []
    for i in range(len(indices)):
        ts = None
        if timestamps is not None and indices[i] < len(timestamps):
            ts = str(timestamps[indices[i]])

        pivot = Pivot(
            index=int(indices[i]),
            price=float(values[i]),
            pivot_type=PivotType.PEAK if types[i] == 1 else PivotType.VALLEY,
            timestamp=ts
        )
        pivots.append(pivot)

    logger.debug(f"ZigZag found {len(pivots)} pivots with deviation={deviation:.3%}")
    return pivots


def compute_zigzag_for_timeframe(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    timeframe: str = "5min",
    timestamps: Optional[np.ndarray] = None
) -> List[Pivot]:
    """
    Compute ZigZag with automatic deviation based on timeframe.

    Args:
        highs: High prices array
        lows: Low prices array
        closes: Close prices array
        timeframe: Timeframe string (e.g., "1min", "5min", "15min", "30min", "daily")
        timestamps: Optional array of timestamps

    Returns:
        List of Pivot objects
    """
    deviation = DEVIATION_PRESETS.get(timeframe.lower(), 0.01)
    logger.info(f"Using ZigZag deviation {deviation:.2%} for {timeframe} timeframe")
    return compute_zigzag(highs, lows, closes, deviation, timestamps)


def get_zigzag_segments(pivots: List[Pivot]) -> List[Tuple[Pivot, Pivot]]:
    """
    Get consecutive pivot pairs as line segments.

    Useful for drawing ZigZag lines on charts.

    Args:
        pivots: List of Pivot objects

    Returns:
        List of (start_pivot, end_pivot) tuples
    """
    if len(pivots) < 2:
        return []

    segments = []
    for i in range(len(pivots) - 1):
        segments.append((pivots[i], pivots[i + 1]))

    return segments


def filter_pivots_by_range(
    pivots: List[Pivot],
    start_index: int,
    end_index: int
) -> List[Pivot]:
    """
    Filter pivots to those within a specific index range.

    Args:
        pivots: List of Pivot objects
        start_index: Start of range (inclusive)
        end_index: End of range (inclusive)

    Returns:
        Filtered list of pivots
    """
    return [p for p in pivots if start_index <= p.index <= end_index]


def get_last_n_pivots(
    pivots: List[Pivot],
    n_peaks: int = 3,
    n_valleys: int = 3
) -> Tuple[List[Pivot], List[Pivot]]:
    """
    Get the last N peaks and last N valleys.

    Useful for pattern detection that requires recent pivots.

    Args:
        pivots: List of Pivot objects
        n_peaks: Number of recent peaks to return
        n_valleys: Number of recent valleys to return

    Returns:
        (recent_peaks, recent_valleys)
    """
    peaks = [p for p in pivots if p.pivot_type == PivotType.PEAK]
    valleys = [p for p in pivots if p.pivot_type == PivotType.VALLEY]

    recent_peaks = peaks[-n_peaks:] if len(peaks) >= n_peaks else peaks
    recent_valleys = valleys[-n_valleys:] if len(valleys) >= n_valleys else valleys

    return recent_peaks, recent_valleys
