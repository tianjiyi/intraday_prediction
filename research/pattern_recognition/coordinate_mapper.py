"""
Coordinate Mapper - Convert YOLO pixel coordinates to tradable price/time levels.

Maps bounding box pixel coordinates from YOLO detection to actual price levels
and timestamps for generating tradable signals.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ChartCoordinateMapper:
    """
    Maps YOLO pixel coordinates to price/time for tradable signals.

    Captures matplotlib axes metadata during chart generation to enable
    accurate conversion of YOLO bounding boxes to price levels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fig,
        axes,
        chart_path: str,
        dpi: int = 100
    ):
        """
        Initialize mapper with chart metadata.

        Args:
            df: DataFrame with OHLCV data used to generate the chart
            fig: Matplotlib figure object
            axes: List of axes from mplfinance (axes[0] is price axis)
            chart_path: Path where chart will be saved
            dpi: DPI used when saving the chart
        """
        self.chart_path = chart_path
        self.dpi = dpi
        self.num_bars = len(df)

        # Store price range from data
        self.min_price = float(df['Low'].min())
        self.max_price = float(df['High'].max())

        # Get the price (main) axes
        price_ax = axes[0]

        # Get figure dimensions in pixels
        fig_width_inches, fig_height_inches = fig.get_size_inches()
        self.fig_width_px = fig_width_inches * dpi
        self.fig_height_px = fig_height_inches * dpi

        # Get axes bounding box in figure coordinates (0-1)
        # We need to render the figure first to get accurate positions
        fig.canvas.draw()
        bbox = price_ax.get_position()

        # Convert to pixel coordinates
        self.ax_left = bbox.x0 * self.fig_width_px
        self.ax_right = bbox.x1 * self.fig_width_px
        self.ax_bottom = bbox.y0 * self.fig_height_px
        self.ax_top = bbox.y1 * self.fig_height_px
        self.ax_width = self.ax_right - self.ax_left
        self.ax_height = self.ax_top - self.ax_bottom

        # Y-axis limits (price range displayed - includes padding)
        self.ylim = price_ax.get_ylim()  # (min_price, max_price)

        # X-axis limits (bar indices)
        self.xlim = price_ax.get_xlim()  # (min_idx, max_idx)

        # Store timestamps for bar index conversion
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            self.timestamps = df.index.tolist()
        elif 'timestamp' in df.columns:
            self.timestamps = pd.to_datetime(df['timestamp']).tolist()
        else:
            self.timestamps = list(range(len(df)))

        logger.debug(
            f"ChartCoordinateMapper initialized: "
            f"fig={self.fig_width_px:.0f}x{self.fig_height_px:.0f}px, "
            f"ax_left={self.ax_left:.1f}, ax_bottom={self.ax_bottom:.1f}, "
            f"ax_width={self.ax_width:.1f}, ax_height={self.ax_height:.1f}, "
            f"ylim={self.ylim}, xlim={self.xlim}"
        )

    def pixel_to_price(self, y_pixel: float) -> float:
        """
        Convert pixel Y coordinate to price level.

        Note: In image coordinates, Y=0 is at the TOP (highest price),
        and Y increases downward (toward lower prices).

        Args:
            y_pixel: Y coordinate in pixels (from top of image)

        Returns:
            Price level at that Y coordinate
        """
        # Image Y is inverted: pixel 0 = top = high price
        # ax_top is at the top of the axes (in image coords, this is lower Y value)
        # ax_bottom is at the bottom of the axes (higher Y value)

        # y_pixel is from top of image, so:
        # - y_pixel = ax_top corresponds to ylim[1] (high price)
        # - y_pixel = ax_bottom corresponds to ylim[0] (low price)

        # But wait - in mplfinance, ax_top is actually stored as a fraction from bottom
        # After conversion to pixels, ax_top is measured from bottom of figure

        # Let's recalculate: in image coordinates (Y from top):
        y_from_top = y_pixel
        # ax_top in image coords (from top) = fig_height - ax_top (from bottom)
        ax_top_from_top = self.fig_height_px - self.ax_top
        ax_bottom_from_top = self.fig_height_px - self.ax_bottom

        # Now map y_from_top to price
        # y_from_top = ax_top_from_top -> ylim[1] (max price)
        # y_from_top = ax_bottom_from_top -> ylim[0] (min price)

        # Linear interpolation
        y_frac = (y_from_top - ax_top_from_top) / (ax_bottom_from_top - ax_top_from_top)
        price = self.ylim[1] - y_frac * (self.ylim[1] - self.ylim[0])

        return price

    def pixel_to_bar_index(self, x_pixel: float) -> int:
        """
        Convert pixel X coordinate to bar index.

        Args:
            x_pixel: X coordinate in pixels (from left of image)

        Returns:
            Bar index (0-based)
        """
        # X increases left to right in both image and data coordinates
        x_frac = (x_pixel - self.ax_left) / self.ax_width
        bar_float = self.xlim[0] + x_frac * (self.xlim[1] - self.xlim[0])
        bar_idx = int(round(bar_float))
        return max(0, min(self.num_bars - 1, bar_idx))

    def bbox_to_tradable(self, bbox: List[float], confidence: float = 0.0) -> Dict[str, Any]:
        """
        Convert YOLO bounding box to tradable signal.

        For W_Bottom pattern:
        - range_high = top of bounding box = breakout trigger
        - range_low = bottom of bounding box = stop loss level

        Args:
            bbox: [x1, y1, x2, y2] pixel coordinates from YOLO
            confidence: Detection confidence (0-1)

        Returns:
            Dictionary with tradable signal data
        """
        x1, y1, x2, y2 = bbox

        # Convert Y coordinates to prices
        # y1 is top of bbox (smaller Y in image = higher price)
        # y2 is bottom of bbox (larger Y in image = lower price)
        price_at_y1 = self.pixel_to_price(y1)
        price_at_y2 = self.pixel_to_price(y2)

        # Ensure range_high > range_low
        range_high = max(price_at_y1, price_at_y2)
        range_low = min(price_at_y1, price_at_y2)

        # Convert X coordinates to bar indices and timestamps
        bar_start = self.pixel_to_bar_index(x1)
        bar_end = self.pixel_to_bar_index(x2)

        # Get timestamps
        time_start = self.timestamps[bar_start]
        time_end = self.timestamps[bar_end]

        # Calculate range size
        range_size = range_high - range_low
        range_pct = (range_size / range_low) * 100 if range_low > 0 else 0

        return {
            'range_high': round(range_high, 2),
            'range_low': round(range_low, 2),
            'range_size': round(range_size, 2),
            'range_pct': round(range_pct, 2),
            'time_start': str(time_start),
            'time_end': str(time_end),
            'bar_start': bar_start,
            'bar_end': bar_end,
            'num_bars': bar_end - bar_start + 1,
            'confidence': round(confidence, 4)
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize mapper metadata for JSON storage.

        Returns:
            Dictionary with all metadata needed to reconstruct mapper
        """
        return {
            'chart_path': self.chart_path,
            'dpi': self.dpi,
            'fig_width_px': self.fig_width_px,
            'fig_height_px': self.fig_height_px,
            'ax_left': self.ax_left,
            'ax_right': self.ax_right,
            'ax_bottom': self.ax_bottom,
            'ax_top': self.ax_top,
            'ax_width': self.ax_width,
            'ax_height': self.ax_height,
            'ylim': list(self.ylim),
            'xlim': list(self.xlim),
            'min_price': self.min_price,
            'max_price': self.max_price,
            'num_bars': self.num_bars,
            'timestamps': [str(t) for t in self.timestamps]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartCoordinateMapper':
        """
        Reconstruct mapper from serialized dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            ChartCoordinateMapper instance
        """
        # Create empty instance
        mapper = cls.__new__(cls)

        # Restore all attributes
        mapper.chart_path = data['chart_path']
        mapper.dpi = data['dpi']
        mapper.fig_width_px = data['fig_width_px']
        mapper.fig_height_px = data['fig_height_px']
        mapper.ax_left = data['ax_left']
        mapper.ax_right = data['ax_right']
        mapper.ax_bottom = data['ax_bottom']
        mapper.ax_top = data['ax_top']
        mapper.ax_width = data['ax_width']
        mapper.ax_height = data['ax_height']
        mapper.ylim = tuple(data['ylim'])
        mapper.xlim = tuple(data['xlim'])
        mapper.min_price = data['min_price']
        mapper.max_price = data['max_price']
        mapper.num_bars = data['num_bars']
        mapper.timestamps = [pd.Timestamp(t) for t in data['timestamps']]

        return mapper


def convert_detections_to_tradable(
    detections: List[Dict[str, Any]],
    mapper_dict: Dict[str, Any],
    pattern_filter: Optional[List[str]] = None,
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Convert YOLO detections to tradable signals.

    Args:
        detections: List of YOLO detection dictionaries
        mapper_dict: Serialized ChartCoordinateMapper from to_dict()
        pattern_filter: List of pattern names to convert (None = all)
        min_confidence: Minimum confidence threshold

    Returns:
        List of detections with 'tradable_signal' added for qualifying patterns
    """
    if not detections or not mapper_dict:
        return detections

    # Reconstruct mapper
    mapper = ChartCoordinateMapper.from_dict(mapper_dict)

    # Default filter to W_Bottom only
    if pattern_filter is None:
        pattern_filter = ['W_Bottom']

    for det in detections:
        # Check if pattern qualifies
        if det['class_name'] not in pattern_filter:
            continue

        if det['confidence'] < min_confidence:
            continue

        # Convert bbox to tradable signal
        tradable = mapper.bbox_to_tradable(
            bbox=det['bbox'],
            confidence=det['confidence']
        )
        det['tradable_signal'] = tradable

        logger.info(
            f"Tradable signal: {det['class_name']} "
            f"range_high=${tradable['range_high']:.2f}, "
            f"range_low=${tradable['range_low']:.2f}, "
            f"conf={tradable['confidence']:.1%}"
        )

    return detections
