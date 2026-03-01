"""
Data models for pattern recognition module.

Defines dataclasses for ZigZag pivots, trendlines, bounding boxes,
and pattern detection results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


class PivotType(Enum):
    """Pivot point type classification."""
    PEAK = 1      # Local maximum (resistance)
    VALLEY = -1   # Local minimum (support)


class PivotClass(Enum):
    """Market structure classification for pivots."""
    HH = "HH"      # Higher High
    LH = "LH"      # Lower High
    HL = "HL"      # Higher Low
    LL = "LL"      # Lower Low
    UNKNOWN = "UNKNOWN"


class BreakoutStatus(Enum):
    """Pattern breakout status classification."""
    PENDING = "pending"   # Not enough post-pattern data or pattern still forming
    SUCCESS = "success"   # Price closed above resistance within lookforward period
    FAILURE = "failure"   # Price closed below support trendline within lookforward period
    EXPIRED = "expired"   # Lookforward period passed without breakout in either direction


@dataclass
class Pivot:
    """
    Represents a ZigZag pivot point.

    Attributes:
        index: Bar index in the data array
        price: Price at the pivot point
        pivot_type: PEAK (local max) or VALLEY (local min)
        timestamp: Optional timestamp string
        pivot_class: Market structure classification (HH, HL, LH, LL)
    """
    index: int
    price: float
    pivot_type: PivotType
    timestamp: Optional[str] = None
    pivot_class: PivotClass = PivotClass.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "index": self.index,
            "price": self.price,
            "type": self.pivot_type.name,
            "timestamp": self.timestamp,
            "class": self.pivot_class.value
        }


@dataclass
class TrendLine:
    """
    Represents a trendline through pivot points.

    Attributes:
        start_index: Starting bar index
        end_index: Ending bar index
        start_price: Price at start
        end_price: Price at end
        slope: Linear regression slope
        intercept: Linear regression intercept
        pivots_used: List of pivot indices used for calculation
    """
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    slope: float
    intercept: float
    pivots_used: List[int] = field(default_factory=list)

    def price_at_index(self, index: int) -> float:
        """Calculate price on trendline at given index."""
        return self.slope * index + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_price": self.start_price,
            "end_price": self.end_price,
            "slope": self.slope,
            "intercept": self.intercept,
            "pivots_used": self.pivots_used
        }


@dataclass
class BoundingBox:
    """
    YOLO-format bounding box with normalized coordinates (0-1).

    Attributes:
        x_center: Center X coordinate (normalized)
        y_center: Center Y coordinate (normalized)
        width: Box width (normalized)
        height: Box height (normalized)
        class_id: Pattern class ID (0 = ascending_triangle)
    """
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int = 0  # 0 = ascending_triangle

    def to_yolo_string(self) -> str:
        """Convert to YOLO annotation format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height
        }


@dataclass
class AscendingTrianglePattern:
    """
    Detected ascending triangle pattern.

    An ascending triangle has:
    - Flat horizontal resistance (peaks at similar levels)
    - Rising support line (higher lows)
    - Compression (price range narrowing over time)

    Attributes:
        ticker: Stock ticker symbol
        timeframe: Timeframe string (e.g., "5min", "15min")
        start_index: First bar index of pattern
        end_index: Last bar index of pattern
        peaks: List of peak pivots forming resistance
        valleys: List of valley pivots forming support
        resistance_level: Average resistance price
        resistance_tolerance: % deviation in resistance
        support_slope: Slope of support trendline
        support_intercept: Intercept of support trendline
        support_trendline: TrendLine object for support
        compression_ratio: How much price range compressed (0-1)
        pattern_height: Max high - Min low of pattern
        confidence: Detection confidence score (0-1)
        bounding_box: YOLO bounding box (if calculated)
    """
    ticker: str
    timeframe: str
    start_index: int
    end_index: int

    # Pivot points
    peaks: List[Pivot] = field(default_factory=list)
    valleys: List[Pivot] = field(default_factory=list)

    # Resistance line (flat)
    resistance_level: float = 0.0
    resistance_tolerance: float = 0.0

    # Support line (ascending)
    support_slope: float = 0.0
    support_intercept: float = 0.0
    support_trendline: Optional[TrendLine] = None

    # Pattern metrics
    compression_ratio: float = 0.0
    pattern_height: float = 0.0
    confidence: float = 0.0

    # YOLO bounding box
    bounding_box: Optional[BoundingBox] = None

    # Optional metadata
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None

    # Context tracking
    pre_context_bars: int = 20
    post_context_bars: int = 20

    # Breakout detection
    breakout_status: str = "pending"
    breakout_index: Optional[int] = None
    breakout_price: Optional[float] = None
    breakout_timestamp: Optional[str] = None
    bars_to_breakout: Optional[int] = None
    support_at_end: Optional[float] = None

    # Post-pattern statistics
    post_pattern_high: Optional[float] = None
    post_pattern_low: Optional[float] = None
    post_pattern_close: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "peaks": [p.to_dict() for p in self.peaks],
            "valleys": [v.to_dict() for v in self.valleys],
            "resistance_level": round(self.resistance_level, 4),
            "resistance_tolerance": round(self.resistance_tolerance, 6),
            "support_slope": round(self.support_slope, 8),
            "support_intercept": round(self.support_intercept, 4),
            "support_trendline": self.support_trendline.to_dict() if self.support_trendline else None,
            "compression_ratio": round(self.compression_ratio, 4),
            "pattern_height": round(self.pattern_height, 4),
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            # Context tracking
            "pre_context_bars": self.pre_context_bars,
            "post_context_bars": self.post_context_bars,
            # Breakout detection
            "breakout_status": self.breakout_status,
            "breakout_index": self.breakout_index,
            "breakout_price": round(self.breakout_price, 4) if self.breakout_price else None,
            "breakout_timestamp": self.breakout_timestamp,
            "bars_to_breakout": self.bars_to_breakout,
            "support_at_end": round(self.support_at_end, 4) if self.support_at_end else None,
            # Post-pattern statistics
            "post_pattern_high": round(self.post_pattern_high, 4) if self.post_pattern_high else None,
            "post_pattern_low": round(self.post_pattern_low, 4) if self.post_pattern_low else None,
            "post_pattern_close": round(self.post_pattern_close, 4) if self.post_pattern_close else None,
        }

    @property
    def pattern_bars(self) -> int:
        """Number of bars in the pattern."""
        return self.end_index - self.start_index + 1

    @property
    def num_peaks(self) -> int:
        """Number of peaks in resistance."""
        return len(self.peaks)

    @property
    def num_valleys(self) -> int:
        """Number of valleys in support."""
        return len(self.valleys)


# Pattern class IDs for YOLO
PATTERN_CLASSES = {
    "ascending_triangle": 0,
    "descending_triangle": 1,
    "symmetrical_triangle": 2,
    "head_shoulders": 3,
    "inverse_head_shoulders": 4,
    "double_top": 5,
    "double_bottom": 6,
    "wedge_rising": 7,
    "wedge_falling": 8,
    "flag_bull": 9,
    "flag_bear": 10,
}


class PatternRecognitionError(Exception):
    """Base exception for pattern recognition module."""
    pass


class InsufficientDataError(PatternRecognitionError):
    """Raised when not enough data for pattern detection."""
    pass


class NoPivotsFoundError(PatternRecognitionError):
    """Raised when ZigZag finds no significant pivots."""
    pass


class DataFetchError(PatternRecognitionError):
    """Raised when data fetching fails."""
    pass
