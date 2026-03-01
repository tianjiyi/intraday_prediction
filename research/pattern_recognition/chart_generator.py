"""
Chart generation for pattern visualization.

Generates:
1. Annotated candlestick charts with pattern overlays (resistance/support lines)
2. YOLO training images (640x640, no axes/borders)

Uses matplotlib and mplfinance for professional-quality financial charts.
"""

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Eastern timezone for consistent chart display
EASTERN = pytz.timezone('US/Eastern')

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    logging.warning("mplfinance not installed. Install with: pip install mplfinance")

from .models import AscendingTrianglePattern, BoundingBox, Pivot, PivotType

logger = logging.getLogger(__name__)


class PatternChartGenerator:
    """
    Generate candlestick charts with pattern annotations.

    Supports two modes:
    1. Annotated charts: Full charts with axes, title, and pattern overlays
    2. YOLO training images: Minimal charts (no axes) for ML training
    """

    def __init__(
        self,
        style: str = "charles",  # mplfinance style
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 100
    ):
        """
        Initialize chart generator.

        Args:
            style: mplfinance chart style ('charles', 'yahoo', 'binance', etc.)
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for output images
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi

    def generate_annotated_chart(
        self,
        df: pd.DataFrame,
        pattern: AscendingTrianglePattern,
        save_path: str,
        show_volume: bool = True,
        padding_bars: int = 20,
        pre_context: Optional[int] = None,
        post_context: Optional[int] = None,
        show_breakout: bool = True,
        show_full_range: bool = False
    ) -> str:
        """
        Generate chart with pattern annotations (resistance + support lines).

        Creates a candlestick chart with:
        - Horizontal resistance line (red dashed)
        - Ascending support trendline (green dashed)
        - Peak markers (red triangles)
        - Valley markers (green triangles)
        - Breakout marker (green star for success, red X for failure)

        Args:
            df: OHLCV DataFrame with DatetimeIndex or timestamp column
            pattern: Detected AscendingTrianglePattern
            save_path: Path to save PNG file
            show_volume: Include volume subplot
            padding_bars: Extra bars to show before/after pattern (fallback)
            pre_context: Override pre-pattern context bars
            post_context: Override post-pattern context bars
            show_breakout: Show breakout marker if detected
            show_full_range: Plot all bars in DataFrame instead of just pattern area

        Returns:
            Path to saved file
        """
        if not MPLFINANCE_AVAILABLE:
            return self._generate_basic_chart(df, pattern, save_path)

        # Prepare data slice
        if show_full_range:
            # Plot entire DataFrame
            start_idx = 0
            end_idx = len(df)
            df_plot = df.copy()
        else:
            # Use pattern context settings if not explicitly provided
            pre_padding = pre_context if pre_context is not None else pattern.pre_context_bars
            post_padding = post_context if post_context is not None else pattern.post_context_bars

            # Fall back to padding_bars if context not set
            if pre_padding == 0:
                pre_padding = padding_bars
            if post_padding == 0:
                post_padding = padding_bars // 2

            start_idx = max(0, pattern.start_index - pre_padding)
            end_idx = min(len(df), pattern.end_index + post_padding + 1)
            df_plot = df.iloc[start_idx:end_idx].copy()

        # Ensure proper index for mplfinance
        if not isinstance(df_plot.index, pd.DatetimeIndex):
            if 'timestamp' in df_plot.columns:
                df_plot = df_plot.set_index('timestamp')
                df_plot.index = pd.to_datetime(df_plot.index)
            else:
                df_plot.index = pd.date_range('2024-01-01', periods=len(df_plot), freq='5min')

        # Convert to Eastern Time (naive) for consistent display regardless of user's local TZ
        # This ensures charts always show Eastern Time labels
        if df_plot.index.tz is not None:
            df_plot.index = df_plot.index.tz_convert(EASTERN).tz_localize(None)

        # Standardize column names for mplfinance
        df_plot = df_plot.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

        n_bars = len(df_plot)

        # Create horizontal resistance line
        resistance_line = [pattern.resistance_level] * n_bars

        # Create ascending support line
        support_line = []
        for i, rel_idx in enumerate(range(start_idx, end_idx)):
            y = pattern.support_slope * rel_idx + pattern.support_intercept
            support_line.append(y)

        # Build addplot list
        apd = [
            mpf.make_addplot(
                resistance_line,
                color='red',
                linestyle='--',
                secondary_y=False
            ),
            mpf.make_addplot(
                support_line,
                color='green',
                linestyle='--',
                secondary_y=False
            ),
        ]

        # Create pivot markers
        peak_markers = [np.nan] * n_bars
        valley_markers = [np.nan] * n_bars

        for peak in pattern.peaks:
            rel_idx = peak.index - start_idx
            if 0 <= rel_idx < n_bars:
                peak_markers[rel_idx] = peak.price * 1.002  # Slightly above

        for valley in pattern.valleys:
            rel_idx = valley.index - start_idx
            if 0 <= rel_idx < n_bars:
                valley_markers[rel_idx] = valley.price * 0.998  # Slightly below

        apd.extend([
            mpf.make_addplot(
                peak_markers,
                type='scatter',
                markersize=120,
                marker='v',
                color='red'
            ),
            mpf.make_addplot(
                valley_markers,
                type='scatter',
                markersize=120,
                marker='^',
                color='green'
            ),
        ])

        # Add breakout marker if detected
        if show_breakout and pattern.breakout_index is not None:
            breakout_rel_idx = pattern.breakout_index - start_idx
            if 0 <= breakout_rel_idx < n_bars:
                breakout_markers = [np.nan] * n_bars

                if pattern.breakout_status == "success":
                    # Green star for successful breakout
                    breakout_markers[breakout_rel_idx] = pattern.breakout_price * 1.003
                    apd.append(mpf.make_addplot(
                        breakout_markers,
                        type='scatter',
                        markersize=200,
                        marker='*',
                        color='lime'
                    ))
                elif pattern.breakout_status == "failure":
                    # Red X for failed breakout
                    breakout_markers[breakout_rel_idx] = pattern.breakout_price * 0.997
                    apd.append(mpf.make_addplot(
                        breakout_markers,
                        type='scatter',
                        markersize=150,
                        marker='x',
                        color='darkred'
                    ))

        # Build chart title with breakout info
        breakout_info = ""
        if pattern.breakout_status != "pending":
            breakout_info = f" | Breakout: {pattern.breakout_status.upper()}"
            if pattern.bars_to_breakout:
                breakout_info += f" ({pattern.bars_to_breakout} bars)"

        # Create figure
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style=self.style,
            addplot=apd,
            volume=show_volume,
            figsize=self.figsize,
            returnfig=True,
            title=f"\n{pattern.ticker} - Ascending Triangle ({pattern.timeframe}) [ET]\n"
                  f"Resistance: ${pattern.resistance_level:.2f} | "
                  f"Confidence: {pattern.confidence:.1%}{breakout_info}"
        )

        # Add legend
        resistance_patch = mpatches.Patch(color='red', label=f'Resistance: ${pattern.resistance_level:.2f}')
        support_patch = mpatches.Patch(color='green', label=f'Support (slope: {pattern.support_slope:.6f})')
        axes[0].legend(handles=[resistance_patch, support_patch], loc='upper left')

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Saved annotated chart to {save_path}")
        return save_path

    def _generate_basic_chart(
        self,
        df: pd.DataFrame,
        pattern: AscendingTrianglePattern,
        save_path: str
    ) -> str:
        """
        Fallback chart generation without mplfinance.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get data slice
        start_idx = max(0, pattern.start_index - 20)
        end_idx = min(len(df), pattern.end_index + 10)
        df_plot = df.iloc[start_idx:end_idx]

        # Get column names
        close_col = 'close' if 'close' in df_plot.columns else 'Close'

        # Plot close prices
        ax.plot(range(len(df_plot)), df_plot[close_col].values, 'b-', linewidth=1)

        # Plot resistance line
        ax.axhline(y=pattern.resistance_level, color='red', linestyle='--', linewidth=2,
                   label=f'Resistance: ${pattern.resistance_level:.2f}')

        # Plot support trendline
        x_support = range(len(df_plot))
        y_support = [pattern.support_slope * (i + start_idx) + pattern.support_intercept for i in x_support]
        ax.plot(x_support, y_support, 'g--', linewidth=2, label='Support')

        # Mark pivots
        for peak in pattern.peaks:
            rel_idx = peak.index - start_idx
            if 0 <= rel_idx < len(df_plot):
                ax.scatter([rel_idx], [peak.price], color='red', marker='v', s=100, zorder=5)

        for valley in pattern.valleys:
            rel_idx = valley.index - start_idx
            if 0 <= rel_idx < len(df_plot):
                ax.scatter([rel_idx], [valley.price], color='green', marker='^', s=100, zorder=5)

        ax.set_title(f"{pattern.ticker} - Ascending Triangle ({pattern.timeframe})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved basic chart to {save_path}")
        return save_path

    def generate_yolo_training_image(
        self,
        df: pd.DataFrame,
        pattern: AscendingTrianglePattern,
        save_path: str,
        image_size: int = 640
    ) -> Tuple[str, BoundingBox]:
        """
        Generate YOLO training image (no axes, borders) with bounding box.

        Creates a clean candlestick chart suitable for object detection training:
        - Fixed size (640x640 by default)
        - No axes, titles, or borders
        - White background

        Args:
            df: OHLCV DataFrame
            pattern: Detected pattern
            save_path: Path to save PNG
            image_size: Output image size (default 640x640)

        Returns:
            (image_path, bounding_box) tuple
        """
        # Calculate view window with padding
        padding = 10
        start_idx = max(0, pattern.start_index - padding)
        end_idx = min(len(df), pattern.end_index + padding // 2)
        df_plot = df.iloc[start_idx:end_idx].copy()

        # Get column names
        open_col = 'open' if 'open' in df_plot.columns else 'Open'
        high_col = 'high' if 'high' in df_plot.columns else 'High'
        low_col = 'low' if 'low' in df_plot.columns else 'Low'
        close_col = 'close' if 'close' in df_plot.columns else 'Close'

        # Create figure with exact size
        fig_size = image_size / 100  # Convert to inches at 100 dpi
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)

        # Remove all decorations
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Draw candlesticks manually
        width = 0.6
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            o, h, l, c = row[open_col], row[high_col], row[low_col], row[close_col]
            color = '#26a69a' if c >= o else '#ef5350'  # Green/Red

            # Body
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height < 0.001:  # Doji
                body_height = 0.001

            ax.add_patch(plt.Rectangle(
                (i - width/2, body_bottom), width, body_height,
                facecolor=color, edgecolor=color
            ))

            # Wicks
            ax.plot([i, i], [l, body_bottom], color=color, linewidth=1)
            ax.plot([i, i], [body_bottom + body_height, h], color=color, linewidth=1)

        # Set axis limits
        n_bars = len(df_plot)
        ax.set_xlim(-1, n_bars)

        y_min = df_plot[low_col].min()
        y_max = df_plot[high_col].max()
        y_padding = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, facecolor='white', edgecolor='none',
                    pad_inches=0, bbox_inches='tight')
        plt.close(fig)

        # Calculate bounding box (normalized 0-1)
        x_total = n_bars
        y_range = (y_max + y_padding) - (y_min - y_padding)

        # Pattern bounds in data coordinates
        pattern_x_start = pattern.start_index - start_idx
        pattern_x_end = pattern.end_index - start_idx

        pattern_y_min = min(v.price for v in pattern.valleys)
        pattern_y_max = pattern.resistance_level

        # Normalize to 0-1
        x_center = ((pattern_x_start + pattern_x_end) / 2) / x_total
        y_center = 1 - ((pattern_y_min + pattern_y_max) / 2 - (y_min - y_padding)) / y_range
        width_norm = (pattern_x_end - pattern_x_start) / x_total
        height_norm = (pattern_y_max - pattern_y_min) / y_range

        # Clamp to valid range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width_norm = max(0.01, min(1.0, width_norm))
        height_norm = max(0.01, min(1.0, height_norm))

        bbox = BoundingBox(
            x_center=x_center,
            y_center=y_center,
            width=width_norm,
            height=height_norm,
            class_id=0  # ascending_triangle
        )

        pattern.bounding_box = bbox

        logger.info(f"Saved YOLO image to {save_path}")
        return save_path, bbox

    def generate_zigzag_chart(
        self,
        df: pd.DataFrame,
        pivots: List[Pivot],
        save_path: str,
        ticker: str = "UNKNOWN"
    ) -> str:
        """
        Generate chart showing ZigZag lines connecting pivots.

        Useful for debugging ZigZag algorithm output.

        Args:
            df: OHLCV DataFrame
            pivots: List of Pivot objects
            save_path: Output path
            ticker: Ticker symbol for title

        Returns:
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        close_col = 'close' if 'close' in df.columns else 'Close'

        # Plot price line
        ax.plot(range(len(df)), df[close_col].values, 'lightgray', linewidth=1, alpha=0.7)

        # Plot ZigZag lines
        if len(pivots) >= 2:
            zigzag_x = [p.index for p in pivots]
            zigzag_y = [p.price for p in pivots]
            ax.plot(zigzag_x, zigzag_y, 'b-', linewidth=2, marker='o', markersize=8)

            # Color peaks and valleys differently
            for p in pivots:
                color = 'red' if p.pivot_type == PivotType.PEAK else 'green'
                marker = 'v' if p.pivot_type == PivotType.PEAK else '^'
                ax.scatter([p.index], [p.price], color=color, marker=marker, s=150, zorder=5)

        ax.set_title(f"{ticker} - ZigZag Analysis ({len(pivots)} pivots)")
        ax.set_xlabel("Bar Index")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved ZigZag chart to {save_path}")
        return save_path
