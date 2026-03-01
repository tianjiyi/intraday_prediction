"""
Pivot point classification for market structure analysis.

Classifies ZigZag pivots into market structure types:
- HH (Higher High): Current peak > Previous peak
- LH (Lower High): Current peak < Previous peak
- HL (Higher Low): Current valley > Previous valley
- LL (Lower Low): Current valley < Previous valley

This classification helps identify trend direction and pattern structures.
"""

from typing import List, Tuple, Optional
import numpy as np
import logging

from .models import Pivot, PivotType, PivotClass, TrendLine

logger = logging.getLogger(__name__)


def classify_pivots(pivots: List[Pivot]) -> List[Pivot]:
    """
    Classify pivots into market structure types (HH, HL, LH, LL).

    Compares each pivot to the previous pivot of the same type:
    - Peaks: HH if higher than previous peak, LH if lower
    - Valleys: HL if higher than previous valley, LL if lower

    Args:
        pivots: List of Pivot objects from ZigZag algorithm

    Returns:
        Same list with pivot_class field populated
    """
    if len(pivots) < 2:
        return pivots

    # Separate peaks and valleys for comparison
    peaks = [p for p in pivots if p.pivot_type == PivotType.PEAK]
    valleys = [p for p in pivots if p.pivot_type == PivotType.VALLEY]

    # Classify peaks
    for i in range(1, len(peaks)):
        if peaks[i].price > peaks[i-1].price:
            peaks[i].pivot_class = PivotClass.HH
        else:
            peaks[i].pivot_class = PivotClass.LH

    # Classify valleys
    for i in range(1, len(valleys)):
        if valleys[i].price > valleys[i-1].price:
            valleys[i].pivot_class = PivotClass.HL
        else:
            valleys[i].pivot_class = PivotClass.LL

    logger.debug(f"Classified {len(peaks)} peaks and {len(valleys)} valleys")
    return pivots


def get_recent_pivots(
    pivots: List[Pivot],
    n_peaks: int = 2,
    n_valleys: int = 2
) -> Tuple[List[Pivot], List[Pivot]]:
    """
    Get the most recent N peaks and N valleys.

    Required for ascending triangle detection:
    - Minimum 2 peaks for flat resistance
    - Minimum 2 valleys for ascending support

    Args:
        pivots: List of Pivot objects
        n_peaks: Number of recent peaks to return
        n_valleys: Number of recent valleys to return

    Returns:
        (recent_peaks, recent_valleys) tuples
    """
    peaks = [p for p in pivots if p.pivot_type == PivotType.PEAK]
    valleys = [p for p in pivots if p.pivot_type == PivotType.VALLEY]

    recent_peaks = peaks[-n_peaks:] if len(peaks) >= n_peaks else peaks
    recent_valleys = valleys[-n_valleys:] if len(valleys) >= n_valleys else valleys

    return recent_peaks, recent_valleys


def calculate_trendline(pivots: List[Pivot]) -> Tuple[float, float]:
    """
    Calculate linear regression trendline through pivot points.

    Uses least squares regression to find the best-fit line.

    Args:
        pivots: List of Pivot objects to fit line through

    Returns:
        (slope, intercept) of the trendline
        slope > 0 indicates upward trend
        slope < 0 indicates downward trend
    """
    if len(pivots) < 2:
        return 0.0, pivots[0].price if pivots else 0.0

    x = np.array([p.index for p in pivots], dtype=np.float64)
    y = np.array([p.price for p in pivots], dtype=np.float64)

    # Linear regression: y = slope * x + intercept
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    denominator = n * sum_x2 - sum_x ** 2
    if abs(denominator) < 1e-10:
        # Near-vertical line, return zero slope
        return 0.0, np.mean(y)

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    return float(slope), float(intercept)


def create_trendline(pivots: List[Pivot]) -> Optional[TrendLine]:
    """
    Create a TrendLine object from pivot points.

    Args:
        pivots: List of Pivot objects

    Returns:
        TrendLine object or None if insufficient pivots
    """
    if len(pivots) < 2:
        return None

    slope, intercept = calculate_trendline(pivots)

    # Sort by index to get start and end
    sorted_pivots = sorted(pivots, key=lambda p: p.index)
    start_pivot = sorted_pivots[0]
    end_pivot = sorted_pivots[-1]

    return TrendLine(
        start_index=start_pivot.index,
        end_index=end_pivot.index,
        start_price=start_pivot.price,
        end_price=end_pivot.price,
        slope=slope,
        intercept=intercept,
        pivots_used=[p.index for p in pivots]
    )


def calculate_average_price(pivots: List[Pivot]) -> float:
    """
    Calculate average price of pivot points.

    Useful for determining resistance level from multiple peaks.

    Args:
        pivots: List of Pivot objects

    Returns:
        Average price
    """
    if not pivots:
        return 0.0

    return sum(p.price for p in pivots) / len(pivots)


def calculate_price_deviation(pivots: List[Pivot], reference_price: float) -> float:
    """
    Calculate maximum deviation from a reference price.

    Used to check if peaks form a "flat" resistance within tolerance.

    Args:
        pivots: List of Pivot objects
        reference_price: Reference price (e.g., average resistance)

    Returns:
        Maximum deviation as a percentage (0.01 = 1%)
    """
    if not pivots or reference_price <= 0:
        return 0.0

    max_deviation = 0.0
    for p in pivots:
        deviation = abs(p.price - reference_price) / reference_price
        max_deviation = max(max_deviation, deviation)

    return max_deviation


def check_higher_lows(valleys: List[Pivot]) -> bool:
    """
    Check if valleys form a series of higher lows.

    Required condition for ascending triangle support.

    Args:
        valleys: List of valley Pivot objects

    Returns:
        True if each valley is higher than the previous one
    """
    if len(valleys) < 2:
        return False

    # Sort by index to ensure chronological order
    sorted_valleys = sorted(valleys, key=lambda v: v.index)

    for i in range(1, len(sorted_valleys)):
        if sorted_valleys[i].price <= sorted_valleys[i-1].price:
            return False

    return True


def check_lower_highs(peaks: List[Pivot]) -> bool:
    """
    Check if peaks form a series of lower highs.

    Required condition for descending triangle resistance.

    Args:
        peaks: List of peak Pivot objects

    Returns:
        True if each peak is lower than the previous one
    """
    if len(peaks) < 2:
        return False

    sorted_peaks = sorted(peaks, key=lambda p: p.index)

    for i in range(1, len(sorted_peaks)):
        if sorted_peaks[i].price >= sorted_peaks[i-1].price:
            return False

    return True


def check_flat_tops(peaks: List[Pivot], tolerance: float = 0.015) -> Tuple[bool, float]:
    """
    Check if peaks form a flat resistance zone.

    Ascending triangle characteristic: peaks at similar price levels.

    Args:
        peaks: List of peak Pivot objects
        tolerance: Maximum deviation allowed (default 1.5%)

    Returns:
        (is_flat, average_resistance_price)
    """
    if len(peaks) < 2:
        return False, 0.0

    avg_price = calculate_average_price(peaks)
    max_deviation = calculate_price_deviation(peaks, avg_price)

    is_flat = max_deviation <= tolerance
    return is_flat, avg_price


def check_flat_bottoms(valleys: List[Pivot], tolerance: float = 0.015) -> Tuple[bool, float]:
    """
    Check if valleys form a flat support zone.

    Descending triangle characteristic: valleys at similar price levels.

    Args:
        valleys: List of valley Pivot objects
        tolerance: Maximum deviation allowed (default 1.5%)

    Returns:
        (is_flat, average_support_price)
    """
    if len(valleys) < 2:
        return False, 0.0

    avg_price = calculate_average_price(valleys)
    max_deviation = calculate_price_deviation(valleys, avg_price)

    is_flat = max_deviation <= tolerance
    return is_flat, avg_price


def get_pivot_sequence(pivots: List[Pivot]) -> str:
    """
    Get the sequence of pivot classifications as a string.

    Useful for pattern matching and debugging.

    Args:
        pivots: List of Pivot objects (should be classified first)

    Returns:
        String like "HL-LH-HL-LH" representing the pivot sequence
    """
    if not pivots:
        return ""

    # Sort by index
    sorted_pivots = sorted(pivots, key=lambda p: p.index)

    sequence = []
    for p in sorted_pivots:
        if p.pivot_class != PivotClass.UNKNOWN:
            sequence.append(p.pivot_class.value)
        else:
            sequence.append("P" if p.pivot_type == PivotType.PEAK else "V")

    return "-".join(sequence)


def get_market_structure(pivots: List[Pivot]) -> str:
    """
    Determine overall market structure from pivot sequence.

    Args:
        pivots: List of classified Pivot objects

    Returns:
        "UPTREND" if HH and HL dominate
        "DOWNTREND" if LH and LL dominate
        "CONSOLIDATION" if mixed
    """
    if len(pivots) < 4:
        return "INSUFFICIENT_DATA"

    # Count classifications
    hh_count = sum(1 for p in pivots if p.pivot_class == PivotClass.HH)
    hl_count = sum(1 for p in pivots if p.pivot_class == PivotClass.HL)
    lh_count = sum(1 for p in pivots if p.pivot_class == PivotClass.LH)
    ll_count = sum(1 for p in pivots if p.pivot_class == PivotClass.LL)

    bullish = hh_count + hl_count
    bearish = lh_count + ll_count

    if bullish > bearish * 1.5:
        return "UPTREND"
    elif bearish > bullish * 1.5:
        return "DOWNTREND"
    else:
        return "CONSOLIDATION"
