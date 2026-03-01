"""
Ascending Triangle Pattern Detection

An ascending triangle is a bullish continuation/breakout pattern characterized by:
1. Flat horizontal resistance (peaks at similar price levels)
2. Rising support line (higher lows)
3. Compression (price range narrowing as pattern develops)

Geometric Definition (from PDF Section 3.4):
- Flat Resistance: |Peak_i - Peak_{i-2}| / Peak_i < tolerance (1.0-1.5%)
- Rising Support: Valley_i > Valley_{i-2} AND slope > 0
- Minimum: 2 peaks + 2 valleys required
- Compression: Price range narrowing over time

Trading Implication:
- Bullish breakout expected when price breaks above resistance
- Pattern failure if price breaks below support trendline
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging

from .models import (
    Pivot, PivotType, PivotClass,
    AscendingTrianglePattern, TrendLine, BoundingBox,
    InsufficientDataError, NoPivotsFoundError
)
from .zigzag import compute_zigzag, compute_zigzag_for_timeframe, DEVIATION_PRESETS
from .pivots import (
    classify_pivots, get_recent_pivots, calculate_trendline,
    create_trendline, calculate_average_price, calculate_price_deviation,
    check_higher_lows, check_flat_tops
)

logger = logging.getLogger(__name__)

# Minimum bars presets by timeframe (auto-scaled like ZigZag deviation)
# Higher timeframes need fewer bars since each bar represents more time
MIN_BARS_PRESETS = {
    "1min": 30,     # 30 minutes minimum
    "5min": 25,     # ~2 hours minimum
    "15min": 20,    # ~5 hours minimum
    "30min": 15,    # ~7.5 hours minimum
    "1hour": 12,    # ~12 hours minimum
    "4hour": 10,    # ~40 hours minimum
    "daily": 10,    # ~2 weeks minimum
    "weekly": 6,    # ~6 weeks minimum
}

# Resistance tolerance presets by timeframe
# Tighter tolerance for intraday (smaller price swings, more precise resistance)
RESISTANCE_TOLERANCE_PRESETS = {
    "1min": 0.005,    # 0.5% - very tight for scalping
    "5min": 0.007,    # 0.7% - tight for intraday
    "15min": 0.010,   # 1.0% - moderate for swing
    "30min": 0.012,   # 1.2%
    "1hour": 0.015,   # 1.5%
    "4hour": 0.015,   # 1.5%
    "daily": 0.015,   # 1.5% - original default
    "weekly": 0.020,  # 2.0% - looser for weekly patterns
}


class AscendingTriangleDetector:
    """
    Detects ascending triangle patterns from OHLC data.

    Uses ZigZag algorithm to identify pivot points, then checks for:
    - Flat resistance zone (horizontal line through peaks)
    - Rising support line (positive slope through valleys)
    - Compression (narrowing price range)

    Example:
        >>> detector = AscendingTriangleDetector()
        >>> pattern = detector.detect(
        ...     highs=df['high'].values,
        ...     lows=df['low'].values,
        ...     closes=df['close'].values,
        ...     ticker="QQQ",
        ...     timeframe="5min"
        ... )
        >>> if pattern:
        ...     print(f"Pattern found: resistance={pattern.resistance_level:.2f}")
    """

    def __init__(
        self,
        resistance_tolerance: float = 0.015,  # 1.5% tolerance for flat top
        min_support_slope: float = 0.00001,   # Minimum positive slope
        min_peaks: int = 2,
        min_valleys: int = 2,
        zigzag_deviation: Optional[float] = None,  # Auto if None
        min_compression: float = 0.0,         # Minimum compression ratio (0 = disabled)
        min_bars: Optional[int] = None,       # Minimum bars (None = auto based on timeframe)
        max_bars: int = 500                   # Maximum bars for pattern
    ):
        """
        Initialize the detector with configurable parameters.

        Args:
            resistance_tolerance: Max % deviation for flat resistance (1.0-1.5% typical)
            min_support_slope: Minimum slope for ascending support (slope > 0 required)
            min_peaks: Minimum number of peaks required (default 2)
            min_valleys: Minimum number of valleys required (default 2)
            zigzag_deviation: ZigZag deviation threshold (None = auto based on timeframe)
            min_compression: Minimum compression ratio (0-1, disabled if 0)
            min_bars: Minimum bars the pattern should span (None = auto based on timeframe)
            max_bars: Maximum bars the pattern can span
        """
        self.resistance_tolerance = resistance_tolerance
        self.min_support_slope = min_support_slope
        self.min_peaks = min_peaks
        self.min_valleys = min_valleys
        self.zigzag_deviation = zigzag_deviation
        self.min_compression = min_compression
        self.min_bars = min_bars
        self.max_bars = max_bars

    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        ticker: str = "UNKNOWN",
        timeframe: str = "5min",
        timestamps: Optional[np.ndarray] = None,
        return_pivots: bool = False,
        pre_context_bars: int = 20,
        post_context_bars: int = 20,
        analyze_breakout: bool = True
    ) -> Optional[AscendingTrianglePattern]:
        """
        Detect ascending triangle pattern in OHLC data.

        Args:
            highs: High prices array
            lows: Low prices array
            closes: Close prices array
            ticker: Stock ticker symbol
            timeframe: Timeframe string (e.g., "1min", "5min", "15min")
            timestamps: Optional timestamp array for pivot labeling
            return_pivots: If True, includes all pivots in pattern (for debugging)
            pre_context_bars: Number of bars before pattern for context
            post_context_bars: Number of bars after pattern for breakout analysis
            analyze_breakout: If True, analyze post-pattern bars for breakout

        Returns:
            AscendingTrianglePattern if detected, None otherwise

        Raises:
            InsufficientDataError: If data array is too short
        """
        n = len(closes)

        # Resolve min_bars (auto-scale based on timeframe if not specified)
        if self.min_bars is not None:
            min_bars = self.min_bars
        else:
            min_bars = MIN_BARS_PRESETS.get(timeframe.lower(), 20)
            logger.debug(f"Auto min_bars={min_bars} for {timeframe} timeframe")

        # Resolve resistance_tolerance (auto-scale based on timeframe if using default)
        if self.resistance_tolerance == 0.015:  # default value
            effective_tolerance = RESISTANCE_TOLERANCE_PRESETS.get(timeframe.lower(), 0.015)
            logger.debug(f"Auto resistance_tolerance={effective_tolerance:.3%} for {timeframe} timeframe")
        else:
            effective_tolerance = self.resistance_tolerance

        if n < min_bars:
            logger.debug(f"Insufficient data: {n} bars < {min_bars} minimum")
            return None

        # Step 1: Compute ZigZag pivots
        if self.zigzag_deviation is not None:
            deviation = self.zigzag_deviation
        else:
            deviation = DEVIATION_PRESETS.get(timeframe.lower(), 0.01)

        pivots = compute_zigzag(highs, lows, closes, deviation, timestamps)

        if len(pivots) < (self.min_peaks + self.min_valleys):
            logger.debug(f"Insufficient pivots: {len(pivots)} found, "
                        f"need {self.min_peaks + self.min_valleys}")
            return None

        # Step 2: Classify pivots (HH, HL, LH, LL)
        pivots = classify_pivots(pivots)

        # Step 3: Get recent peaks and valleys
        peaks, valleys = get_recent_pivots(pivots, self.min_peaks, self.min_valleys)

        if len(peaks) < self.min_peaks:
            logger.debug(f"Not enough peaks: {len(peaks)} < {self.min_peaks}")
            return None

        if len(valleys) < self.min_valleys:
            logger.debug(f"Not enough valleys: {len(valleys)} < {self.min_valleys}")
            return None

        # Step 4: Check flat resistance condition
        is_flat, resistance_level = self._check_flat_resistance(peaks, effective_tolerance)
        if not is_flat:
            logger.debug(f"Resistance not flat enough (tolerance={effective_tolerance:.2%})")
            return None

        # Step 5: Check ascending support condition
        is_ascending, support_slope, support_intercept = self._check_ascending_support(valleys)
        if not is_ascending:
            logger.debug(f"Support not ascending: slope={support_slope:.8f}")
            return None

        # Step 6: Check pattern span
        all_pattern_pivots = peaks + valleys
        start_index = min(p.index for p in all_pattern_pivots)
        end_index = max(p.index for p in all_pattern_pivots)
        pattern_span = end_index - start_index

        if pattern_span < min_bars:
            logger.debug(f"Pattern too short: {pattern_span} < {min_bars}")
            return None

        if pattern_span > self.max_bars:
            logger.debug(f"Pattern too long: {pattern_span} > {self.max_bars}")
            return None

        # Step 7: Check compression (optional)
        compression_ratio = self._calculate_compression(peaks, valleys)
        if self.min_compression > 0 and compression_ratio < self.min_compression:
            logger.debug(f"Insufficient compression: {compression_ratio:.2%} < {self.min_compression:.2%}")
            return None

        # Step 8: Build pattern result
        support_trendline = create_trendline(valleys)

        pattern = AscendingTrianglePattern(
            ticker=ticker,
            timeframe=timeframe,
            start_index=start_index,
            end_index=end_index,
            peaks=peaks,
            valleys=valleys,
            resistance_level=resistance_level,
            resistance_tolerance=calculate_price_deviation(peaks, resistance_level),
            support_slope=support_slope,
            support_intercept=support_intercept,
            support_trendline=support_trendline,
            compression_ratio=compression_ratio,
            pattern_height=resistance_level - min(v.price for v in valleys),
            confidence=self._calculate_confidence(peaks, valleys, resistance_level, support_slope, effective_tolerance),
            start_timestamp=peaks[0].timestamp if peaks[0].timestamp else None,
            end_timestamp=peaks[-1].timestamp if peaks[-1].timestamp else None,
            pre_context_bars=pre_context_bars,
            post_context_bars=post_context_bars
        )

        # Step 9: Analyze breakout if requested
        if analyze_breakout:
            pattern = self._analyze_breakout(
                pattern=pattern,
                closes=closes,
                highs=highs,
                lows=lows,
                timestamps=timestamps,
                lookforward_bars=post_context_bars
            )

        logger.info(f"Ascending triangle detected for {ticker}: "
                   f"resistance={resistance_level:.2f}, slope={support_slope:.8f}, "
                   f"confidence={pattern.confidence:.2%}")

        return pattern

    def _check_flat_resistance(self, peaks: List[Pivot], tolerance: float) -> Tuple[bool, float]:
        """
        Check if peaks form a flat resistance zone.

        Condition: All peaks within tolerance of average price.
        Formula: |Peak_i - avg| / avg < tolerance for all peaks
        """
        return check_flat_tops(peaks, tolerance)

    def _check_ascending_support(
        self,
        valleys: List[Pivot]
    ) -> Tuple[bool, float, float]:
        """
        Check if valleys form an ascending support line.

        Conditions:
        1. Each valley higher than the previous (Higher Lows)
        2. Trendline slope > min_support_slope
        """
        if len(valleys) < 2:
            return False, 0.0, 0.0

        # Check Higher Lows condition
        if not check_higher_lows(valleys):
            return False, 0.0, 0.0

        # Calculate trendline slope
        slope, intercept = calculate_trendline(valleys)

        if slope < self.min_support_slope:
            return False, slope, intercept

        return True, slope, intercept

    def _calculate_compression(
        self,
        peaks: List[Pivot],
        valleys: List[Pivot]
    ) -> float:
        """
        Calculate how much the price range has compressed.

        Compression = 1 - (final_range / initial_range)

        A higher value indicates more compression (price coiling tighter).
        """
        if len(peaks) < 2 or len(valleys) < 2:
            return 0.0

        # Sort by index
        sorted_peaks = sorted(peaks, key=lambda p: p.index)
        sorted_valleys = sorted(valleys, key=lambda v: v.index)

        # Initial range (first peak - first valley, absolute)
        initial_range = abs(sorted_peaks[0].price - sorted_valleys[0].price)

        # Final range (last peak - last valley, absolute)
        final_range = abs(sorted_peaks[-1].price - sorted_valleys[-1].price)

        if initial_range <= 0:
            return 0.0

        compression = 1 - (final_range / initial_range)
        return max(0.0, compression)

    def _calculate_confidence(
        self,
        peaks: List[Pivot],
        valleys: List[Pivot],
        resistance_level: float,
        support_slope: float,
        tolerance: float
    ) -> float:
        """
        Calculate pattern confidence score (0-1).

        Factors:
        - Number of touches on resistance (more = higher confidence)
        - Number of touches on support (more = higher confidence)
        - Flatness of resistance (flatter = higher confidence)
        - Slope strength of support
        - Compression quality
        """
        # Pivot count score (more pivots = higher confidence)
        total_pivots = len(peaks) + len(valleys)
        pivot_score = min(1.0, total_pivots / 8)  # Max score at 8 pivots

        # Resistance flatness score
        resistance_deviation = calculate_price_deviation(peaks, resistance_level)
        flatness_score = max(0.0, 1.0 - (resistance_deviation / tolerance))

        # Support slope score (stronger slope = higher confidence, up to a point)
        slope_score = min(1.0, support_slope / 0.0005)

        # Compression score
        compression_score = self._calculate_compression(peaks, valleys)

        # Weighted average
        confidence = (
            pivot_score * 0.30 +
            flatness_score * 0.30 +
            slope_score * 0.20 +
            compression_score * 0.20
        )

        return confidence

    def _analyze_breakout(
        self,
        pattern: AscendingTrianglePattern,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        lookforward_bars: int = 20
    ) -> AscendingTrianglePattern:
        """
        Analyze post-pattern bars for breakout detection.

        Breakout Logic:
        - SUCCESS: Close > resistance_level within lookforward_bars
        - FAILURE: Close < support trendline (at that bar's index) within lookforward_bars
        - EXPIRED: lookforward_bars passed without either condition
        - PENDING: Not enough data after pattern end

        Args:
            pattern: Detected AscendingTrianglePattern
            closes: Full close prices array
            highs: Full high prices array
            lows: Full low prices array
            timestamps: Optional timestamps array
            lookforward_bars: Number of bars after pattern end to analyze

        Returns:
            Pattern with breakout fields populated
        """
        n = len(closes)
        end_idx = pattern.end_index

        # Calculate support level at pattern end
        pattern.support_at_end = pattern.support_slope * end_idx + pattern.support_intercept

        # Check if we have enough post-pattern data
        available_post_bars = n - end_idx - 1
        pattern.post_context_bars = min(lookforward_bars, available_post_bars)

        if available_post_bars <= 0:
            pattern.breakout_status = "pending"
            logger.debug(f"No post-pattern bars available for {pattern.ticker}")
            return pattern

        # Analyze post-pattern bars
        analyze_end = min(end_idx + lookforward_bars + 1, n)

        post_highs = highs[end_idx + 1:analyze_end]
        post_lows = lows[end_idx + 1:analyze_end]
        post_closes = closes[end_idx + 1:analyze_end]

        if len(post_closes) > 0:
            pattern.post_pattern_high = float(np.max(post_highs))
            pattern.post_pattern_low = float(np.min(post_lows))
            pattern.post_pattern_close = float(post_closes[-1])

        # Check each bar for breakout conditions
        for i, (close, high, low) in enumerate(zip(post_closes, post_highs, post_lows)):
            bar_index = end_idx + 1 + i

            # Calculate support level at this bar's index (support line continues)
            support_at_bar = pattern.support_slope * bar_index + pattern.support_intercept

            # SUCCESS: Close above resistance
            if close > pattern.resistance_level:
                pattern.breakout_status = "success"
                pattern.breakout_index = bar_index
                pattern.breakout_price = float(close)
                pattern.bars_to_breakout = i + 1
                if timestamps is not None and bar_index < len(timestamps):
                    pattern.breakout_timestamp = str(timestamps[bar_index])
                logger.info(f"{pattern.ticker}: Bullish breakout at bar {bar_index}, "
                           f"close={close:.2f} > resistance={pattern.resistance_level:.2f}")
                return pattern

            # FAILURE: Close below support trendline
            if close < support_at_bar:
                pattern.breakout_status = "failure"
                pattern.breakout_index = bar_index
                pattern.breakout_price = float(close)
                pattern.bars_to_breakout = i + 1
                if timestamps is not None and bar_index < len(timestamps):
                    pattern.breakout_timestamp = str(timestamps[bar_index])
                logger.info(f"{pattern.ticker}: Pattern failure at bar {bar_index}, "
                           f"close={close:.2f} < support={support_at_bar:.2f}")
                return pattern

        # Reached end of lookforward period without breakout
        if len(post_closes) >= lookforward_bars:
            pattern.breakout_status = "expired"
            logger.debug(f"{pattern.ticker}: Pattern expired after {lookforward_bars} bars")
        else:
            pattern.breakout_status = "pending"
            logger.debug(f"{pattern.ticker}: Only {len(post_closes)} post-pattern bars, "
                        f"need {lookforward_bars}")

        return pattern


def scan_for_ascending_triangles(
    data_dict: Dict[str, Any],
    lookback_bars: int = 200,
    timeframe: str = "5min",
    **detector_kwargs
) -> List[AscendingTrianglePattern]:
    """
    Scan multiple tickers for ascending triangle patterns.

    Args:
        data_dict: Dict mapping ticker to DataFrame with OHLCV data
        lookback_bars: Number of recent bars to analyze
        timeframe: Timeframe string for ZigZag deviation
        **detector_kwargs: Additional arguments for AscendingTriangleDetector

    Returns:
        List of detected AscendingTrianglePattern objects
    """
    detector = AscendingTriangleDetector(**detector_kwargs)
    patterns = []

    for ticker, df in data_dict.items():
        try:
            # Get last N bars
            df_recent = df.tail(lookback_bars)

            if len(df_recent) < 50:  # Minimum data requirement
                logger.debug(f"Skipping {ticker}: only {len(df_recent)} bars")
                continue

            # Get column names (handle both lowercase and capitalized)
            high_col = 'high' if 'high' in df_recent.columns else 'High'
            low_col = 'low' if 'low' in df_recent.columns else 'Low'
            close_col = 'close' if 'close' in df_recent.columns else 'Close'

            pattern = detector.detect(
                highs=df_recent[high_col].values,
                lows=df_recent[low_col].values,
                closes=df_recent[close_col].values,
                ticker=ticker,
                timeframe=timeframe,
                timestamps=df_recent.index.values if hasattr(df_recent.index, 'values') else None
            )

            if pattern is not None:
                patterns.append(pattern)
                logger.info(f"[FOUND] {ticker}: Ascending triangle detected")
            else:
                logger.debug(f"[----] {ticker}: No pattern")

        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
            continue

    logger.info(f"Scan complete: {len(patterns)} patterns found in {len(data_dict)} tickers")
    return patterns


def get_pattern_summary(patterns: List[AscendingTrianglePattern]) -> Dict[str, Any]:
    """
    Generate summary statistics for detected patterns.

    Args:
        patterns: List of detected patterns

    Returns:
        Dictionary with summary statistics
    """
    if not patterns:
        return {"count": 0}

    confidences = [p.confidence for p in patterns]
    heights = [p.pattern_height for p in patterns]
    compressions = [p.compression_ratio for p in patterns]

    # Breakout statistics
    breakout_stats = {
        "success": sum(1 for p in patterns if p.breakout_status == "success"),
        "failure": sum(1 for p in patterns if p.breakout_status == "failure"),
        "pending": sum(1 for p in patterns if p.breakout_status == "pending"),
        "expired": sum(1 for p in patterns if p.breakout_status == "expired"),
    }

    # Average bars to breakout (for completed patterns)
    completed_patterns = [p for p in patterns if p.bars_to_breakout is not None]
    avg_bars_to_breakout = (
        sum(p.bars_to_breakout for p in completed_patterns) / len(completed_patterns)
        if completed_patterns else None
    )

    # Success rate (only count success + failure, not pending/expired)
    resolved_count = breakout_stats["success"] + breakout_stats["failure"]
    success_rate = (
        breakout_stats["success"] / resolved_count
        if resolved_count > 0 else None
    )

    return {
        "count": len(patterns),
        "tickers": [p.ticker for p in patterns],
        "avg_confidence": sum(confidences) / len(confidences),
        "max_confidence": max(confidences),
        "avg_height": sum(heights) / len(heights),
        "avg_compression": sum(compressions) / len(compressions),
        "highest_confidence_ticker": max(patterns, key=lambda p: p.confidence).ticker,
        # Breakout statistics
        "breakout_stats": breakout_stats,
        "success_rate": success_rate,
        "avg_bars_to_breakout": avg_bars_to_breakout
    }
