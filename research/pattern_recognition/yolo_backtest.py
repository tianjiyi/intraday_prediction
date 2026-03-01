"""
YOLO Intraday Backtest Scanner.

Simulates real-time YOLO pattern detection on historical intraday data.
Runs detection minute-by-minute through a trading day, generating tradable signals.

Usage:
    python -m pattern_recognition.yolo_backtest --ticker QQQ --date 2025-12-23

Output:
    - JSON file with detected signals and tradable levels
    - Chart with range_high/range_low marked for each signal
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_recognition.data_fetcher import fetch_bars, load_config, BARS_PER_DAY
from pattern_recognition.yolo_inference import YOLOPatternDetector
from pattern_recognition.coordinate_mapper import ChartCoordinateMapper

logger = logging.getLogger(__name__)


def calculate_daily_sma(df: pd.DataFrame, sma_period: int = 5) -> Dict[str, float]:
    """
    Calculate daily closes and previous N-day SMA for each trading day.

    The SMA for each day is calculated using the PREVIOUS N days' closes,
    so it's known at market open (no lookahead bias).

    Args:
        df: Intraday DataFrame with 'timestamp' and 'close' columns
        sma_period: Number of days for SMA calculation (default: 5)

    Returns:
        Dict mapping date string to previous N-day SMA value
        e.g., {'2025-10-10': 607.51, '2025-10-11': 605.23, ...}
    """
    # Extract daily closes from intraday data
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date

    # Get last close of each day (EOD close)
    daily_closes = df_copy.groupby('date')['close'].last().reset_index()
    daily_closes.columns = ['date', 'close']
    daily_closes = daily_closes.sort_values('date')

    # Calculate SMA using PREVIOUS N days (shift by 1 to avoid lookahead)
    daily_closes['sma'] = daily_closes['close'].shift(1).rolling(sma_period).mean()

    # Convert to dict: date_str -> prev_sma
    result = {}
    for _, row in daily_closes.iterrows():
        date_str = str(row['date'])
        if pd.notna(row['sma']):
            result[date_str] = round(row['sma'], 2)

    return result


class YOLOIntradayBacktester:
    """
    Simulates real-time YOLO detection on a single trading day.

    At each minute from market open, generates chart with all bars
    seen so far (from 9:30 to current minute) and runs detection.
    """

    def __init__(
        self,
        min_bars_for_detection: int = 30,  # Wait 30 mins after open
        step_bars: int = 1,                 # Run detection every N bars
        min_confidence: float = 0.40,       # 40% threshold for tradable signals
        yolo_confidence: float = 0.25,      # YOLO base confidence threshold
        tradable_patterns: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        chart_window_bars: Optional[int] = None  # None = growing window, int = fixed window
    ):
        """
        Initialize the intraday backtester.

        Args:
            min_bars_for_detection: Minimum bars before starting detection (from open)
            step_bars: Run detection every N bars (1 = every minute)
            min_confidence: Minimum confidence for tradable signals
            yolo_confidence: Base YOLO detection confidence
            tradable_patterns: List of pattern names to track (default: ['W_Bottom'])
            model_path: Path to YOLO model (None = download from HuggingFace)
            chart_window_bars: Chart window size for detection.
                               None = growing window (9:30 to current - legacy behavior)
                               int = fixed window of last N bars (recommended: 150)
                               Fixed window improves early detection by keeping patterns
                               visually larger in the chart.
        """
        self.min_bars = min_bars_for_detection
        self.step_bars = step_bars
        self.min_confidence = min_confidence
        # Normalize pattern names to lowercase for case-insensitive matching
        self.tradable_patterns = [p.lower() for p in (tradable_patterns or ['W_Bottom'])]
        self.chart_window_bars = chart_window_bars

        # Initialize YOLO detector
        logger.info("Initializing YOLO detector...")
        self.detector = YOLOPatternDetector(
            model_path=model_path,
            confidence=yolo_confidence
        )

        # Chart style
        self.mc = mpf.make_marketcolors(
            up='#26a69a',
            down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume={'up': '#26a69a', 'down': '#ef5350'}
        )
        self.style = mpf.make_mpf_style(
            marketcolors=self.mc,
            gridcolor='#e0e0e0',
            gridstyle='-',
            gridaxis='both',
            facecolor='white'
        )

    def run_intraday_backtest(
        self,
        day_df: pd.DataFrame,
        ticker: str,
        date_str: str,
        output_dir: Path,
        verbose: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Simulate minute-by-minute detection through one trading day.

        Args:
            day_df: Current day's bars (9:30-16:00) with 'timestamp', OHLCV columns
            ticker: Symbol
            date_str: Trading date (YYYY-MM-DD)
            output_dir: Output directory for results
            verbose: Enable verbose logging

        Returns:
            Tuple of (signals, early_detections) where:
            - signals: List of tradable signals (above threshold)
            - early_detections: List of ALL detections for pattern tracking
        """
        signals = []
        detected_patterns = []  # Track to avoid duplicates
        early_detections = []  # Track ALL detections for analysis

        total_bars = len(day_df)
        detection_steps = (total_bars - self.min_bars) // self.step_bars

        logger.info(f"Starting intraday backtest for {ticker} on {date_str}")
        logger.info(f"Total bars: {total_bars}, Min bars: {self.min_bars}, Step: {self.step_bars}")
        logger.info(f"Detection steps: {detection_steps}")

        # Log window mode
        if self.chart_window_bars is None:
            logger.info("Chart window: GROWING (9:30 to current - legacy mode)")
        else:
            logger.info(f"Chart window: FIXED {self.chart_window_bars} bars (recommended for earlier detection)")

        # Create temp directory for charts
        temp_dir = Path(tempfile.mkdtemp())

        try:
            step_count = 0
            for end_idx in range(self.min_bars, total_bars, self.step_bars):
                step_count += 1

                # Create window based on configuration
                if self.chart_window_bars is None:
                    # Growing window: 9:30 to current minute (legacy behavior)
                    window_df = day_df.iloc[:end_idx + 1].copy()
                else:
                    # Fixed window: last N bars (better for early detection)
                    start_idx = max(0, end_idx - self.chart_window_bars + 1)
                    window_df = day_df.iloc[start_idx:end_idx + 1].copy()

                current_time = window_df['timestamp'].iloc[-1]

                if verbose and step_count % 30 == 0:
                    logger.debug(f"Step {step_count}/{detection_steps}: {current_time}")

                # Generate chart for this point in time
                chart_path, mapper = self._generate_intraday_chart(
                    window_df, ticker, date_str, temp_dir, end_idx
                )

                if chart_path is None:
                    continue

                # Run YOLO detection
                detections = self.detector.detect(chart_path)

                # Process tradable patterns - log ALL detections for analysis
                for det in detections:
                    if det['class_name'].lower() in self.tradable_patterns:
                        # Create signal for analysis (even if below threshold)
                        signal = self._create_signal(
                            det, mapper, window_df, current_time, end_idx
                        )

                        # Log early detection for pattern tracking
                        early_det = {
                            'detection_time': str(current_time),
                            'detection_bar': end_idx,
                            'pattern': det['class_name'],
                            'confidence': det['confidence'],
                            'range_high': signal['range_high'],
                            'range_low': signal['range_low'],
                            'pattern_start': signal['pattern_start'],
                            'pattern_end': signal['pattern_end']
                        }
                        early_detections.append(early_det)

                        # Log if verbose or first time seeing this pattern area
                        if verbose or det['confidence'] >= 0.30:
                            logger.info(
                                f"  [EARLY] @ {current_time}: {det['class_name']} "
                                f"conf={det['confidence']:.1%} "
                                f"range=${signal['range_low']:.2f}-${signal['range_high']:.2f}"
                            )

                        # Only add to tradable signals if above threshold
                        if det['confidence'] >= self.min_confidence:
                            # Deduplicate (same pattern at similar price/time)
                            if not self._is_duplicate(signal, detected_patterns):
                                signals.append(signal)
                                detected_patterns.append(signal)

                                logger.info(
                                    f"  >>> SIGNAL @ {current_time}: {det['class_name']} "
                                    f"[{det['confidence']:.1%}] "
                                    f"range=${signal['range_low']:.2f}-${signal['range_high']:.2f}"
                                )

                # Cleanup temp chart
                try:
                    os.remove(chart_path)
                except:
                    pass

        finally:
            # Cleanup temp directory
            try:
                for f in temp_dir.glob('*'):
                    f.unlink()
                temp_dir.rmdir()
            except:
                pass

        logger.info(f"Backtest complete: {len(signals)} signals, {len(early_detections)} total detections")
        return signals, early_detections

    def _generate_intraday_chart(
        self,
        window_df: pd.DataFrame,
        ticker: str,
        date_str: str,
        temp_dir: Path,
        bar_idx: int
    ) -> Tuple[Optional[str], Optional[ChartCoordinateMapper]]:
        """
        Generate candlestick chart for current window.

        Returns:
            (chart_path, mapper) tuple, or (None, None) on error
        """
        try:
            # Prepare data for mplfinance
            chart_df = window_df.copy()
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            chart_df.set_index('timestamp', inplace=True)

            # Rename columns for mplfinance
            chart_df = chart_df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            chart_path = temp_dir / f"chart_{bar_idx:04d}.png"
            chart_dpi = 100

            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=self.style,
                volume=True,
                title=f'{ticker} - {date_str}',
                figsize=(12, 8),
                returnfig=True,
                tight_layout=True
            )

            # Create coordinate mapper BEFORE saving
            mapper = ChartCoordinateMapper(
                df=chart_df,
                fig=fig,
                axes=axes,
                chart_path=str(chart_path),
                dpi=chart_dpi
            )

            # Save chart
            fig.savefig(str(chart_path), dpi=chart_dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(chart_path), mapper

        except Exception as e:
            logger.error(f"Error generating chart at bar {bar_idx}: {e}")
            return None, None

    def _create_signal(
        self,
        detection: Dict[str, Any],
        mapper: ChartCoordinateMapper,
        window_df: pd.DataFrame,
        detection_time: Any,
        detection_bar: int
    ) -> Dict[str, Any]:
        """
        Create tradable signal from YOLO detection.
        """
        # Convert bbox to price levels
        tradable = mapper.bbox_to_tradable(
            bbox=detection['bbox'],
            confidence=detection['confidence']
        )

        return {
            'detection_time': str(detection_time),
            'detection_bar': detection_bar,
            'pattern': detection['class_name'],
            'confidence': detection['confidence'],
            'range_high': tradable['range_high'],
            'range_low': tradable['range_low'],
            'range_size': tradable['range_size'],
            'range_pct': tradable['range_pct'],
            'pattern_start': tradable['time_start'],
            'pattern_end': tradable['time_end'],
            'pattern_bars': tradable['num_bars'],
            'window_bars': len(window_df),
            'bbox': detection['bbox']
        }

    def _is_duplicate(
        self,
        signal: Dict[str, Any],
        existing: List[Dict[str, Any]],
        price_tolerance: float = 0.01  # 1% price difference
    ) -> bool:
        """
        Check if signal is duplicate of existing one.

        Duplicate if:
        - Same pattern type
        - Similar range_high/range_low (within price_tolerance)

        Note: We compare pattern price levels, not detection time.
        The same visual pattern may be detected at different times
        as the chart grows, but it's still the same pattern.
        """
        for existing_sig in existing:
            if signal['pattern'] != existing_sig['pattern']:
                continue

            # Check price similarity - same pattern will have nearly identical levels
            high_diff = abs(signal['range_high'] - existing_sig['range_high']) / existing_sig['range_high']
            low_diff = abs(signal['range_low'] - existing_sig['range_low']) / existing_sig['range_low']

            if high_diff < price_tolerance and low_diff < price_tolerance:
                # Same pattern - keep the first detection (earlier in day)
                return True

        return False


def generate_signal_chart(
    day_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    ticker: str,
    date_str: str,
    output_path: str,
    daily_5_sma: Optional[float] = None
):
    """
    Generate full-day chart with detected pattern ranges marked as boxes.

    Shows:
    - Box for each pattern (time range x price range)
    - Green top line (range_high = breakout trigger)
    - Red bottom line (range_low = stop loss)
    - Arrow marking breakout point if triggered
    - Daily 5 SMA horizontal line (if provided)

    Args:
        day_df: Full day's bars
        signals: List of detected signals
        ticker: Symbol
        date_str: Trading date
        output_path: Path to save chart
        daily_5_sma: Previous 5-day SMA value (known at market open)
    """
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    # Prepare data for mplfinance
    chart_df = day_df.copy()
    chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
    chart_df.set_index('timestamp', inplace=True)

    chart_df = chart_df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Chart style
    mc = mpf.make_marketcolors(
        up='#26a69a',
        down='#ef5350',
        edge='inherit',
        wick='inherit',
        volume={'up': '#26a69a', 'down': '#ef5350'}
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridcolor='#e0e0e0',
        gridstyle='-',
        gridaxis='both',
        facecolor='white'
    )

    # Title with signal count and SMA info
    sma_info = f' | Daily 5 SMA: ${daily_5_sma:.2f}' if daily_5_sma else ''
    title = f'{ticker} - {date_str} | {len(signals)} W_Bottom Signal(s){sma_info}'

    # Generate chart
    fig, axes = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        volume=True,
        title=title,
        figsize=(16, 10),
        returnfig=True,
        tight_layout=True
    )

    ax = axes[0]

    # Expand y-axis to include stop loss, profit target, pattern levels, and SMA
    if signals:
        current_ylim = ax.get_ylim()
        all_stops = [sig.get('stop_loss', sig['range_low']) for sig in signals]
        all_targets = [sig.get('profit_target', sig['range_high']) for sig in signals]
        all_pattern_lows = [sig.get('actual_pattern_low', sig['range_low']) for sig in signals]
        min_level = min(min(all_stops), min(all_pattern_lows), current_ylim[0])
        max_level = max(max(all_targets), current_ylim[1])
        # Include SMA in y-axis range if provided
        if daily_5_sma:
            min_level = min(min_level, daily_5_sma)
            max_level = max(max_level, daily_5_sma)
        # Add padding
        padding = (max_level - min_level) * 0.05
        ax.set_ylim(min_level - padding, max_level + padding)

    # Draw Daily 5 SMA horizontal line (orange, dashed)
    if daily_5_sma:
        ax.axhline(
            y=daily_5_sma,
            color='orange',
            linestyle='--',
            linewidth=2.5,
            alpha=0.9,
            zorder=1,
            label=f'Daily 5 SMA: ${daily_5_sma:.2f}'
        )

    # Get x-axis mapping (bar index to plot position)
    # mplfinance uses integer indices for x-axis
    timestamps = chart_df.index.tolist()

    def time_to_x(time_str):
        """Convert timestamp string to x-axis position."""
        target_time = pd.Timestamp(time_str)
        for i, ts in enumerate(timestamps):
            if ts >= target_time:
                return i
        return len(timestamps) - 1

    # Draw boxes and breakout markers for each signal
    for i, sig in enumerate(signals):
        range_high = sig['range_high']
        range_low = sig['range_low']
        # Use actual pattern high/low from bar data (more accurate than bbox)
        actual_pattern_high = sig.get('actual_pattern_high', range_high)
        actual_pattern_low = sig.get('actual_pattern_low', range_low)
        stop_loss = sig.get('stop_loss', actual_pattern_low)
        profit_target = sig.get('profit_target', range_high)

        # Get pattern time range
        pattern_start_x = time_to_x(sig['pattern_start'])
        pattern_end_x = time_to_x(sig['pattern_end'])

        # Draw box (rectangle) for the pattern range using ORIGINAL model bbox
        box_width = pattern_end_x - pattern_start_x
        box_height = range_high - range_low  # Original bbox detection

        # Semi-transparent blue box (shows what model detected)
        rect = Rectangle(
            (pattern_start_x, range_low),  # Original bbox
            box_width,
            box_height,
            linewidth=2,
            edgecolor='blue',
            facecolor='lightblue',
            alpha=0.3,
            zorder=2
        )
        ax.add_patch(rect)

        # Draw pattern high line (green, limit entry level)
        ax.hlines(
            actual_pattern_high,
            pattern_start_x,
            len(timestamps) - 1,
            colors='green',
            linestyles='--',
            linewidth=2,
            alpha=0.8,
            zorder=3,
            label='Entry (Pattern High)' if i == 0 else None
        )

        # Draw pattern low line (orange)
        ax.hlines(
            actual_pattern_low,
            pattern_start_x,
            len(timestamps) - 1,
            colors='orange',
            linestyles='--',
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
            label='Pattern Low' if i == 0 else None
        )

        # Draw stop loss line (red, pattern low - 1*ATR)
        ax.hlines(
            stop_loss,
            pattern_start_x,
            len(timestamps) - 1,
            colors='red',
            linestyles='-',
            linewidth=2,
            alpha=0.8,
            zorder=3,
            label='Stop Loss' if i == 0 else None
        )

        # Draw profit target line (cyan)
        ax.hlines(
            profit_target,
            pattern_start_x,
            len(timestamps) - 1,
            colors='cyan',
            linestyles='-',
            linewidth=2,
            alpha=0.8,
            zorder=3,
            label='Profit Target (2R)' if i == 0 else None
        )

        # Add label near the box with trade info
        risk = sig.get('risk', 0)
        label_text = (
            f"#{i+1} W_Bottom\n"
            f"Conf: {sig['confidence']:.1%}\n"
            f"Entry: ${actual_pattern_high:.2f}\n"
            f"PatternLow: ${actual_pattern_low:.2f}\n"
            f"Stop: ${stop_loss:.2f}\n"
            f"Target: ${profit_target:.2f}\n"
            f"Risk: ${risk:.2f} (2:1)"
        )
        ax.annotate(
            label_text,
            xy=(pattern_start_x, profit_target + risk * 0.1),
            fontsize=7,
            color='blue',
            fontweight='bold',
            ha='left',
            va='bottom',
            zorder=5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

        # Mark ENTRY point when trade triggered
        if sig.get('result') == 'triggered' and 'entry_bar' in sig:
            entry_bar = sig['entry_bar']
            entry_price = sig['entry_price']

            # Green up arrow for entry
            ax.annotate(
                '',
                xy=(entry_bar, entry_price),
                xytext=(entry_bar, entry_price - box_height * 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                zorder=5
            )
            ax.annotate(
                f"ENTRY\n${entry_price:.2f}",
                xy=(entry_bar + 2, entry_price + box_height * 0.1),
                fontsize=7, color='green', fontweight='bold',
                ha='left', va='bottom', zorder=5
            )

        # Mark trade exits
        if sig.get('result') == 'triggered' and 'exit_details' in sig:
            for exit in sig['exit_details']:
                exit_bar = exit['bar']
                exit_price = exit['price']
                exit_type = exit['type']
                pnl = exit['pnl']

                if exit_type == 'first_target_1R':
                    # Blue arrow for first target (1R) - 50% exit
                    ax.annotate(
                        '',
                        xy=(exit_bar, exit_price),
                        xytext=(exit_bar, exit_price - box_height * 0.5),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                        zorder=5
                    )
                    ax.annotate(
                        f"1R EXIT 50%\n+${pnl:.2f}",
                        xy=(exit_bar + 2, exit_price + box_height * 0.1),
                        fontsize=7, color='blue', fontweight='bold',
                        ha='left', va='bottom', zorder=5
                    )
                elif exit_type == 'profit_target':
                    # Cyan arrow for profit target (2.0R) - remaining 50%
                    ax.annotate(
                        '',
                        xy=(exit_bar, exit_price),
                        xytext=(exit_bar, exit_price - box_height * 0.5),
                        arrowprops=dict(arrowstyle='->', color='cyan', lw=2),
                        zorder=5
                    )
                    pct_closed = exit.get('position_closed', 0.5)
                    ax.annotate(
                        f"2R TARGET {pct_closed:.0%}\n+${pnl:.2f}",
                        xy=(exit_bar + 2, exit_price + box_height * 0.1),
                        fontsize=7, color='cyan', fontweight='bold',
                        ha='left', va='bottom', zorder=5
                    )
                elif exit_type == 'breakeven':
                    # Yellow arrow for breakeven exit
                    ax.annotate(
                        '',
                        xy=(exit_bar, exit_price),
                        xytext=(exit_bar, exit_price + box_height * 0.5),
                        arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                        zorder=5
                    )
                    ax.annotate(
                        f"BREAKEVEN\n${pnl:+.2f}",
                        xy=(exit_bar + 2, exit_price - box_height * 0.1),
                        fontsize=7, color='gold', fontweight='bold',
                        ha='left', va='top', zorder=5
                    )
                elif exit_type in ['stop_loss', 'trailing_stop']:
                    # Red down arrow for stop
                    ax.annotate(
                        '',
                        xy=(exit_bar, exit_price),
                        xytext=(exit_bar, exit_price + box_height * 0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        zorder=5
                    )
                    label = 'STOP' if exit_type == 'stop_loss' else 'TRAIL STOP'
                    ax.annotate(
                        f"{label}\n${pnl:+.2f}",
                        xy=(exit_bar + 2, exit_price - box_height * 0.1),
                        fontsize=7, color='red', fontweight='bold',
                        ha='left', va='top', zorder=5
                    )
                elif 'eod_close' in exit_type:
                    # Yellow marker for EOD close
                    pct_closed = exit.get('position_closed', 0.5)
                    ax.plot(exit_bar, exit_price, 'yo', markersize=8, zorder=5)
                    ax.annotate(
                        f"EOD {pct_closed:.0%}\n${pnl:+.2f}",
                        xy=(exit_bar + 2, exit_price),
                        fontsize=7, color='orange', fontweight='bold',
                        ha='left', va='center', zorder=5
                    )

            # Show total P&L
            total_pnl = sig.get('realized_pnl', 0)
            outcome = sig.get('trade_outcome', '')
            pnl_color = 'green' if total_pnl > 0 else 'red' if total_pnl < 0 else 'gray'
            ax.annotate(
                f"Total: ${total_pnl:+.2f} ({sig.get('realized_pnl_pct', 0):+.2f}%)",
                xy=(len(timestamps) - 5, profit_target + risk * 0.3),
                fontsize=9, color=pnl_color, fontweight='bold',
                ha='right', va='bottom', zorder=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
            )

        # Mark stop hit if stopped before entry
        elif sig.get('result') == 'stopped_before_entry' and 'stop_bar' in sig:
            stop_bar = sig['stop_bar']
            ax.annotate(
                '',
                xy=(stop_bar, stop_loss),
                xytext=(stop_bar, stop_loss + box_height * 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                zorder=5
            )
            ax.annotate(
                'STOPPED\nBEFORE ENTRY',
                xy=(stop_bar + 2, stop_loss - box_height * 0.2),
                fontsize=8, color='red', fontweight='bold',
                ha='left', va='top', zorder=5
            )

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='green', linestyle='--', label='Entry (Range High)'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='Stop Loss'),
        Line2D([0], [0], color='cyan', linestyle='-', linewidth=2, label='Profit Target (2:1)'),
        Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue',
                  alpha=0.3, label='Pattern Zone')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved signal chart: {output_path}")


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range for stop loss buffer.

    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default 14)

    Returns:
        ATR value
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr_list = []
    for i in range(1, len(df)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.0

    return np.mean(tr_list[-period:])


def is_pivot_low(df: pd.DataFrame, bar_idx: int, lookback: int = 3) -> bool:
    """
    Detect zigzag pivot low.

    A pivot low occurs when the bar's low is lower than N bars before AND N bars after.

    Args:
        df: DataFrame with 'low' column
        bar_idx: Index position in DataFrame to check
        lookback: Number of bars to check before and after (default 3)

    Returns:
        True if bar_idx is a pivot low, False otherwise
    """
    # Need enough bars before and after
    if bar_idx < lookback or bar_idx >= len(df) - lookback:
        return False

    current_low = df.iloc[bar_idx]['low']

    # Check bars before - current must be LOWER than all previous N bars
    for i in range(1, lookback + 1):
        if df.iloc[bar_idx - i]['low'] <= current_low:
            return False

    # Check bars after - current must be LOWER than all following N bars
    for i in range(1, lookback + 1):
        if df.iloc[bar_idx + i]['low'] <= current_low:
            return False

    return True


def evaluate_signals(
    signals: List[Dict[str, Any]],
    day_df: pd.DataFrame,
    atr_multiplier: float = 1.0,
    reward_risk_ratio: float = 2.0,
    partial_exit_pct: float = 0.80,
    trailing_stop_atr: float = 2.5
) -> List[Dict[str, Any]]:
    """
    Evaluate each signal's outcome with hybrid entry and trade management.

    Hybrid Entry Logic:
    1. Place limit order at pattern_high (support level)
    2. If price pulls back to pattern_high -> limit fills
    3. After 30 mins from pattern_end, if no fill -> look for zigzag pivot low
    4. On pivot low detection -> enter market at confirmation bar's close

    Trade Management:
    1. Stop Loss: Actual pattern low - (ATR * atr_multiplier)
    2. 1st Target: +1R (close 50%)
    3. 2nd Target: +2R (close remaining 50%)
    4. Trailing stop after 1R reached

    Args:
        signals: List of detected signals
        day_df: Full day's bars
        atr_multiplier: ATR multiplier for stop buffer (default 1.0)
        reward_risk_ratio: Reward/Risk ratio (default 2.0)
        partial_exit_pct: Percentage to exit at target (default 0.80 = 80%)
        trailing_stop_atr: ATR multiplier for trailing stop (default 2.5)

    Returns:
        Updated signals with trade management results
    """
    for sig in signals:
        detection_bar = sig['detection_bar']

        # Get bars up to detection for ATR calculation
        pre_detection_df = day_df.iloc[:detection_bar + 1]
        atr = calculate_atr(pre_detection_df)
        sig['atr'] = round(atr, 4)

        # Use pattern high/low directly from YOLO model detection (bbox)
        # No adjustment - trust the model's price-based training
        actual_pattern_low = sig['range_low']
        actual_pattern_high = sig['range_high']
        sig['actual_pattern_low'] = round(actual_pattern_low, 2)
        sig['actual_pattern_high'] = round(actual_pattern_high, 2)

        # Store pattern timestamps for reference
        pattern_start = pd.Timestamp(sig['pattern_start'])
        pattern_end = pd.Timestamp(sig['pattern_end'])

        # Convert day_df timestamps for pattern_end bar index calculation
        day_df_ts = day_df.copy()
        day_df_ts['ts'] = pd.to_datetime(day_df_ts['timestamp'])

        # Hybrid Entry Logic:
        # 1. Place limit order at pattern_high
        # 2. If price pulls back to pattern_high -> limit fills
        # 3. After 30 mins from pattern_end, if no fill -> look for pivot low
        # 4. On pivot low detection -> enter market at pivot close + cancel limit

        detection_bar_data = day_df.iloc[detection_bar]
        limit_entry_price = actual_pattern_high  # Limit order at pattern high

        # Stop loss = pattern low - 1*ATR
        stop_loss = actual_pattern_low - (atr * atr_multiplier)
        sig['stop_loss'] = round(stop_loss, 2)
        sig['limit_entry_price'] = round(limit_entry_price, 2)

        # Get remaining bars after detection for entry simulation
        remaining_df = day_df.iloc[detection_bar + 1:].copy()

        if len(remaining_df) == 0:
            sig['result'] = 'no_data'
            continue

        # Find pattern_end bar index
        pattern_end_mask = day_df_ts['ts'] <= pattern_end
        pattern_end_bar_idx = pattern_end_mask.sum() - 1 if pattern_end_mask.any() else detection_bar

        # Simulate bar-by-bar to determine entry
        entry_price = None
        entry_bar = None
        entry_type = None
        limit_filled = False
        pivot_entry = False
        pivot_bar_idx = None
        minutes_from_pattern_end = 0
        wait_time_for_pivot = 30  # Minutes to wait before looking for pivot
        pivot_lookback = 3  # Zigzag pivot lookback

        # Iterate through post-detection bars
        for i, (idx, row) in enumerate(remaining_df.iterrows()):
            bar_num = day_df.index.get_loc(idx)

            # Calculate minutes from pattern end (assuming 1-min bars)
            bars_since_pattern_end = bar_num - pattern_end_bar_idx
            minutes_from_pattern_end = bars_since_pattern_end  # 1 bar = 1 min

            # Check 1: Limit order fill (price touches pattern_high from above)
            # Limit fills when bar's low <= limit_entry_price
            # Fill price = bar's close (realistic execution, not limit price)
            if row['low'] <= limit_entry_price:
                entry_price = row['close']  # Use bar close as fill price
                entry_bar = bar_num
                entry_type = 'limit'
                limit_filled = True
                sig['entry_timestamp'] = str(row['timestamp'])
                sig['limit_order_price'] = round(limit_entry_price, 2)
                break

            # Check 2: After wait_time, look for pivot low
            if minutes_from_pattern_end >= wait_time_for_pivot:
                # Need at least pivot_lookback bars after this one to confirm pivot
                # We're checking bars retrospectively: bar at (bar_num - pivot_lookback) is the potential pivot
                potential_pivot_bar_num = bar_num - pivot_lookback

                # Ensure potential pivot is after pattern_end and we have enough bars
                if potential_pivot_bar_num > pattern_end_bar_idx + wait_time_for_pivot - pivot_lookback:
                    # Check if that bar is a pivot low
                    if potential_pivot_bar_num >= pivot_lookback and potential_pivot_bar_num < len(day_df) - pivot_lookback:
                        if is_pivot_low(day_df, potential_pivot_bar_num, pivot_lookback):
                            # Pivot confirmed! Enter market at CONFIRMATION bar's close (current bar)
                            entry_price = row['close']
                            entry_bar = bar_num
                            entry_type = 'pivot_market'
                            pivot_entry = True
                            pivot_bar_idx = potential_pivot_bar_num
                            sig['entry_timestamp'] = str(row['timestamp'])
                            sig['pivot_bar'] = potential_pivot_bar_num
                            sig['pivot_price'] = round(day_df.iloc[potential_pivot_bar_num]['low'], 2)
                            break

        # No entry triggered
        if entry_price is None:
            sig['result'] = 'no_entry'
            sig['reason'] = 'no_limit_fill_or_pivot'
            continue

        # Check if entry price is valid (not below stop)
        if entry_price <= stop_loss:
            sig['result'] = 'invalid_entry'
            sig['reason'] = 'entry_below_stop'
            continue

        # Check if entry price is below pattern low (pattern has broken down)
        if entry_price < actual_pattern_low:
            sig['result'] = 'invalid_entry'
            sig['reason'] = 'entry_below_pattern_low'
            sig['entry_price'] = round(entry_price, 2)
            sig['entry_type'] = entry_type
            continue

        # Calculate R based on actual entry price
        risk = entry_price - stop_loss
        sig['risk'] = round(risk, 2)

        # Record entry details
        sig['result'] = 'triggered'
        sig['entry_bar'] = entry_bar
        sig['bars_to_entry'] = entry_bar - detection_bar
        sig['entry_price'] = round(entry_price, 2)
        sig['avg_entry_price'] = round(entry_price, 2)
        sig['entry_type'] = entry_type
        sig['limit_filled'] = limit_filled
        sig['pivot_entry'] = pivot_entry
        sig['initial_position'] = 1.0

        # Profit targets based on R
        profit_target = entry_price + (reward_risk_ratio * risk)
        sig['profit_target'] = round(profit_target, 2)
        sig['reward_risk_ratio'] = reward_risk_ratio

        # Get post-entry bars
        post_entry = day_df.iloc[entry_bar + 1:]

        if len(post_entry) == 0:
            sig['result'] = 'eod_entry'
            sig['total_return_pct'] = 0.0
            continue

        # Track trade outcome
        sig['mfe'] = round(post_entry['high'].max() - entry_price, 2)
        sig['mfe_pct'] = round((sig['mfe'] / entry_price) * 100, 3)
        sig['mae'] = round(entry_price - post_entry['low'].min(), 2)
        sig['mae_pct'] = round((sig['mae'] / entry_price) * 100, 3)

        # Simulate bar-by-bar trade management
        position = 1.0  # Always start with full position (market order)
        realized_pnl = 0.0
        trailing_stop = stop_loss
        highest_high = entry_price
        exit_details = []
        first_target_triggered = False
        first_target_threshold = risk * 1.0  # At +1R: close 50% and move to breakeven
        one_r_reached = False
        one_r_threshold = first_target_threshold  # Same as 1st target now
        # Trailing stop starts after 1st target (1R) is reached

        for idx, row in post_entry.iterrows():
            bar_num = day_df.index.get_loc(idx)

            # Check first target (+1.0R): close 50% position
            if not first_target_triggered and row['high'] >= entry_price + first_target_threshold:
                # Close 50% at +1.0R
                exit_pct = 0.5
                exit_price = entry_price + first_target_threshold  # Exit at 1.0R level
                pnl = (exit_price - entry_price) * exit_pct
                realized_pnl += pnl
                exit_details.append({
                    'type': 'first_target_1R',
                    'bar': bar_num,
                    'price': round(exit_price, 2),
                    'position_closed': round(exit_pct, 2),
                    'pnl': round(pnl, 2)
                })
                position -= exit_pct
                # No breakeven - trailing stop will kick in after 1R
                first_target_triggered = True

            # Check stop loss first (only trigger on CLOSE below stop, not wick)
            if row['close'] <= trailing_stop and position > 0:
                # Stopped out - determine exit type
                exit_price = row['close']  # Exit at actual close, not stop level
                pnl = (exit_price - entry_price) * position
                realized_pnl += pnl

                # Determine exit type
                if trailing_stop == stop_loss:
                    exit_type = 'stop_loss'
                else:
                    exit_type = 'trailing_stop'

                exit_details.append({
                    'type': exit_type,
                    'bar': bar_num,
                    'price': round(exit_price, 2),
                    'position_closed': round(position, 2),
                    'pnl': round(pnl, 2)
                })
                position = 0
                break

            # Check if 1R profit level reached
            if not one_r_reached and row['high'] >= entry_price + one_r_threshold:
                one_r_reached = True

            # Update highest high for trailing stop
            if row['high'] > highest_high:
                highest_high = row['high']
                # Only update trailing stop if:
                # 1. Position is < 1.0 (50% already exited at 1R)
                # 2. 1R has been reached (prevents early exit on pullbacks)
                if position < 1.0 and one_r_reached:
                    new_trailing = highest_high - (atr * trailing_stop_atr)
                    trailing_stop = max(trailing_stop, new_trailing)

            # Check profit target (2.0R) - close remaining position
            if row['high'] >= profit_target and position > 0 and first_target_triggered:
                # Close remaining position at profit target
                exit_price = profit_target
                pnl = (exit_price - entry_price) * position
                realized_pnl += pnl
                exit_details.append({
                    'type': 'profit_target',
                    'bar': bar_num,
                    'price': round(exit_price, 2),
                    'position_closed': round(position, 2),
                    'pnl': round(pnl, 2)
                })
                position = 0
                break  # Trade complete

        # End of day - close remaining position at 15:57 EST (3 min before close)
        # Find the bar closest to 15:57
        if position > 0:
            eod_bar = None
            eod_price = None

            for idx, row in post_entry.iterrows():
                ts = pd.Timestamp(row['timestamp']) if 'timestamp' in row else idx
                # Check if this bar is at or after 15:57
                if hasattr(ts, 'hour') and hasattr(ts, 'minute'):
                    if ts.hour == 15 and ts.minute >= 57:
                        eod_bar = day_df.index.get_loc(idx)
                        eod_price = row['close']
                        break
                    elif ts.hour >= 16:
                        eod_bar = day_df.index.get_loc(idx)
                        eod_price = row['close']
                        break

            # Fallback to last bar if 15:57 not found
            if eod_bar is None:
                eod_bar = day_df.index.get_loc(post_entry.index[-1])
                eod_price = post_entry.iloc[-1]['close']

            pnl = (eod_price - entry_price) * position
            realized_pnl += pnl
            exit_details.append({
                'type': 'eod_close_1557',
                'bar': eod_bar,
                'price': round(eod_price, 2),
                'position_closed': round(position, 2),
                'pnl': round(pnl, 2)
            })
            position = 0

        # Store trade results
        sig['realized_pnl'] = round(realized_pnl, 2)
        sig['realized_pnl_pct'] = round((realized_pnl / entry_price) * 100, 3)
        sig['exit_details'] = exit_details

        # Determine final outcome
        if realized_pnl > 0:
            sig['trade_outcome'] = 'profit'
        elif realized_pnl < 0:
            sig['trade_outcome'] = 'loss'
        else:
            sig['trade_outcome'] = 'breakeven'

    return signals


def split_by_trading_day(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split DataFrame into separate DataFrames for each trading day."""
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    days = {}
    for date_val, group in df.groupby('date'):
        days[str(date_val)] = group.drop(columns=['date']).reset_index(drop=True)

    return days


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLO Intraday Backtest - Minute-by-minute pattern detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Backtest single day
    python -m pattern_recognition.yolo_backtest --ticker QQQ --date 2025-12-23

    # Backtest date range
    python -m pattern_recognition.yolo_backtest --ticker QQQ --start-date 2025-12-20 --end-date 2025-12-27

    # Run every 5 minutes (faster)
    python -m pattern_recognition.yolo_backtest --ticker QQQ --date 2025-12-23 --step 5
        """
    )

    parser.add_argument(
        '--ticker', '-t',
        default='QQQ',
        help='Ticker symbol (default: QQQ)'
    )
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Single date to backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for range backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for range backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.30,
        help='Minimum confidence for tradable signals (default: 0.30)'
    )
    parser.add_argument(
        '--min-bars',
        type=int,
        default=30,
        help='Minimum bars before starting detection (default: 30 = 10:00 AM)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='Run detection every N bars (default: 1 = every minute)'
    )
    parser.add_argument(
        '--chart-window',
        type=int,
        default=None,
        help='Chart window size in bars. None/0 = growing window from 9:30 (legacy). '
             'Recommended: 150 for earlier detection. Fixed window keeps patterns '
             'visually larger, improving detection confidence and reducing lag.'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./runs/detect/w_bottom_train/weights/best.pt',
        help='Path to custom YOLO model weights (default: locally trained w_bottom model)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./yolo_backtest_output',
        help='Output directory (default: ./yolo_backtest_output)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml file'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Starting capital for portfolio simulation (default: $100,000)'
    )
    parser.add_argument(
        '--sma-filter',
        action='store_true',
        default=False,
        help='Enable daily 5 SMA filter: skip signals when price < previous 5-day SMA'
    )
    parser.add_argument(
        '--save-charts',
        action='store_true',
        default=False,
        help='Save individual signal charts for each day'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Print header
    print("\n" + "=" * 70)
    print("  YOLO Intraday Backtest Scanner")
    print("=" * 70)
    print(f"  Ticker:     {args.ticker}")
    print(f"  Model:      {args.model_path}")
    print(f"  Confidence: {args.confidence:.0%}")
    print(f"  Min bars:   {args.min_bars} (start detection at 10:{args.min_bars - 30:02d} AM)")
    print(f"  Step:       Every {args.step} bar(s)")
    print(f"  SMA Filter: {'ENABLED (skip signals when price < prev 5d SMA)' if args.sma_filter else 'Disabled'}")
    print("=" * 70)
    print()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print("ERROR: Could not load configuration.")
        sys.exit(1)

    # Determine date range
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        dates = []
        current = start
        while current <= end:
            dates.append(str(current))
            current += timedelta(days=1)
    else:
        # Default: yesterday
        yesterday = date.today() - timedelta(days=1)
        dates = [str(yesterday)]

    print(f"Dates to backtest: {dates}")
    print()

    # Fetch data for all dates plus buffer
    first_date = min(dates)
    last_date = max(dates)

    # Calculate days to fetch from TODAY back to the earliest requested date
    # (data fetch goes back from current date, not from requested date)
    start_dt = datetime.strptime(first_date, "%Y-%m-%d")
    today = datetime.now()
    days_from_today = (today - start_dt).days + 5  # Add buffer for weekends

    lookback_bars = days_from_today * BARS_PER_DAY.get('1min', 390)

    print(f"Fetching {lookback_bars} bars (~{days_from_today} days from today to {first_date})...")

    try:
        data = fetch_bars(
            tickers=[args.ticker],
            lookback_bars=lookback_bars,
            timeframe='1min',
            config=config,
            rth_only=True
        )
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.ticker not in data:
        print(f"ERROR: No data returned for {args.ticker}")
        sys.exit(1)

    df = data[args.ticker]
    print(f"Fetched {len(df)} bars")

    # Split by trading day
    daily_data = split_by_trading_day(df)
    print(f"Available trading days: {list(daily_data.keys())}")

    # Calculate daily 5 SMA for all days (uses previous 5 days - no lookahead)
    daily_5_sma = calculate_daily_sma(df, sma_period=5)
    if args.sma_filter:
        print(f"Daily 5 SMA calculated for {len(daily_5_sma)} days")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize backtester
    # Convert chart_window=0 to None for legacy growing window behavior
    chart_window = args.chart_window if args.chart_window and args.chart_window > 0 else None

    backtester = YOLOIntradayBacktester(
        min_bars_for_detection=args.min_bars,
        step_bars=args.step,
        min_confidence=args.confidence,
        chart_window_bars=chart_window,
        model_path=args.model_path
    )

    # Portfolio simulation
    starting_capital = args.capital
    current_capital = starting_capital
    portfolio_history = []

    print(f"\n  Portfolio Simulation:")
    print(f"  Starting Capital:  ${starting_capital:,.2f}")
    print("=" * 70)

    # Run backtest for each date
    all_signals = {}
    summary_stats = {
        'total_signals': 0,
        'triggered': 0,
        'stopped_before_entry': 0,
        'expired': 0,
        'no_entry': 0,
        'sma_filtered': 0,  # Signals skipped due to price < prev 5 SMA
        'days_processed': 0,
        'total_pnl_dollars': 0.0,
        'winning_trades': 0,
        'losing_trades': 0,
        'profit_target_hits': 0,
        'stop_loss_hits': 0,
        'breakeven_hits': 0,
        'trailing_stop_hits': 0,
        'eod_closes': 0,
        # Entry type breakdown
        'entry_limit': 0,
        'entry_pivot': 0
    }

    all_early_detections = {}

    for date_str in dates:
        if date_str not in daily_data:
            print(f"Skipping {date_str}: No data available")
            continue

        day_df = daily_data[date_str]

        # Get daily 5 SMA for this date (may be None if not enough history)
        sma_value = daily_5_sma.get(date_str, None)
        sma_info = f", 5d SMA: ${sma_value:.2f}" if sma_value else ""
        print(f"\n--- Processing {date_str} ({len(day_df)} bars{sma_info}) ---")

        # Run backtest - now returns (signals, early_detections)
        signals, early_detections = backtester.run_intraday_backtest(
            day_df=day_df,
            ticker=args.ticker,
            date_str=date_str,
            output_dir=output_dir,
            verbose=args.verbose
        )

        all_early_detections[date_str] = early_detections

        if signals:
            # Apply SMA filter if enabled: skip signals when detection price < prev 5 SMA
            if args.sma_filter and sma_value:
                filtered_signals = []
                for sig in signals:
                    detection_bar = sig['detection_bar']
                    detection_price = day_df.iloc[detection_bar]['close']

                    if detection_price < sma_value:
                        # Signal filtered - price below daily 5 SMA
                        sig['result'] = 'sma_filtered'
                        sig['sma_filter_reason'] = f'Detection price ${detection_price:.2f} < 5d SMA ${sma_value:.2f}'
                        summary_stats['sma_filtered'] += 1
                        print(f"  [FILTERED] Signal at bar {detection_bar}: price ${detection_price:.2f} < 5d SMA ${sma_value:.2f}")
                    else:
                        filtered_signals.append(sig)

                # Keep filtered signals in list for reporting but mark them
                all_filtered = [s for s in signals if s.get('result') == 'sma_filtered']
                signals = filtered_signals  # Only evaluate unfiltered signals

            # Evaluate outcomes for unfiltered signals
            if signals:
                signals = evaluate_signals(signals, day_df)

            # Combine filtered and evaluated signals for charting
            if args.sma_filter and sma_value:
                signals = signals + all_filtered

            # Generate chart with ranges marked (always include SMA line)
            if args.save_charts or signals:
                chart_path = output_dir / f"{args.ticker}_{date_str.replace('-', '')}_signals.png"
                generate_signal_chart(
                    day_df=day_df,
                    signals=signals,
                    ticker=args.ticker,
                    date_str=date_str,
                    output_path=str(chart_path),
                    daily_5_sma=sma_value
                )

            # Portfolio simulation - calculate position size and actual P&L
            daily_pnl_dollars = 0.0

            for sig in signals:
                result = sig.get('result', 'unknown')
                if result in summary_stats:
                    summary_stats[result] += 1

                # Track P&L for triggered trades
                if result == 'triggered':
                    # Track entry type
                    entry_type = sig.get('entry_type', 'unknown')
                    if entry_type == 'limit':
                        summary_stats['entry_limit'] += 1
                    elif entry_type == 'pivot_market':
                        summary_stats['entry_pivot'] += 1

                    entry_price = sig.get('entry_price', 0)
                    pnl_per_share = sig.get('realized_pnl', 0)

                    # Calculate shares: full position for hybrid entry
                    initial_position = sig.get('initial_position', 1.0)
                    if entry_price > 0:
                        shares = int((current_capital * initial_position) / entry_price)
                        sig['shares'] = shares
                        sig['position_value'] = round(shares * entry_price, 2)

                        # Calculate dollar P&L
                        pnl_dollars = pnl_per_share * shares
                        sig['pnl_dollars'] = round(pnl_dollars, 2)
                        sig['pnl_pct'] = round((pnl_dollars / (shares * entry_price)) * 100, 3) if shares > 0 else 0

                        daily_pnl_dollars += pnl_dollars
                        summary_stats['total_pnl_dollars'] += pnl_dollars

                        if pnl_dollars > 0:
                            summary_stats['winning_trades'] += 1
                        elif pnl_dollars < 0:
                            summary_stats['losing_trades'] += 1

                    # Track exit types
                    for exit in sig.get('exit_details', []):
                        if exit['type'] == 'profit_target':
                            summary_stats['profit_target_hits'] += 1
                        elif exit['type'] == 'stop_loss':
                            summary_stats['stop_loss_hits'] += 1
                        elif exit['type'] == 'breakeven':
                            summary_stats['breakeven_hits'] += 1
                        elif exit['type'] == 'trailing_stop':
                            summary_stats['trailing_stop_hits'] += 1
                        elif exit['type'] == 'eod_close_1557':
                            summary_stats['eod_closes'] += 1

            # Update portfolio
            current_capital += daily_pnl_dollars
            portfolio_history.append({
                'date': date_str,
                'signals': len(signals),
                'daily_pnl': round(daily_pnl_dollars, 2),
                'capital': round(current_capital, 2),
                'return_pct': round((current_capital / starting_capital - 1) * 100, 2)
            })

            if daily_pnl_dollars != 0:
                pnl_color = '+' if daily_pnl_dollars > 0 else ''
                print(f"    Portfolio: ${pnl_color}{daily_pnl_dollars:,.2f} | Capital: ${current_capital:,.2f}")

            summary_stats['total_signals'] += len(signals)

        all_signals[date_str] = signals
        summary_stats['days_processed'] += 1

        # Save daily results
        daily_json = output_dir / f"{args.ticker}_{date_str.replace('-', '')}_signals.json"
        with open(daily_json, 'w') as f:
            json.dump({'date': date_str, 'signals': signals}, f, indent=2, default=str)
        print(f"  Saved: {daily_json}")

        # Save early detections for analysis
        early_json = output_dir / f"{args.ticker}_{date_str.replace('-', '')}_early_detections.json"
        with open(early_json, 'w') as f:
            json.dump({
                'date': date_str,
                'total_detections': len(early_detections),
                'detections': early_detections
            }, f, indent=2, default=str)
        print(f"  Saved early detections: {early_json} ({len(early_detections)} detections)")

    # Save summary
    summary_path = output_dir / f"{args.ticker}_backtest_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'ticker': args.ticker,
            'dates': dates,
            'parameters': {
                'min_confidence': args.confidence,
                'min_bars': args.min_bars,
                'step_bars': args.step
            },
            'stats': summary_stats,
            'signals_by_date': {d: len(s) for d, s in all_signals.items()}
        }, f, indent=2)

    # Save portfolio history
    portfolio_path = output_dir / f"{args.ticker}_portfolio_history.json"
    with open(portfolio_path, 'w') as f:
        json.dump({
            'starting_capital': starting_capital,
            'final_capital': current_capital,
            'total_return_pct': round((current_capital / starting_capital - 1) * 100, 2),
            'history': portfolio_history
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  Backtest Summary")
    print("=" * 70)
    print(f"  Days processed:        {summary_stats['days_processed']}")
    print(f"  Total signals:         {summary_stats['total_signals']}")
    if summary_stats['sma_filtered'] > 0:
        print(f"  SMA filtered:          {summary_stats['sma_filtered']} (skipped: price < 5d SMA)")
    print(f"  Triggered (entry):     {summary_stats['triggered']}")
    print(f"  No entry (limit/pivot):{summary_stats['no_entry']}")
    print(f"  Stopped before entry:  {summary_stats['stopped_before_entry']}")
    print(f"  Expired (no action):   {summary_stats['expired']}")

    if summary_stats['total_signals'] > 0:
        # Calculate trigger rate excluding SMA filtered signals
        effective_signals = summary_stats['total_signals'] - summary_stats['sma_filtered']
        if effective_signals > 0:
            trigger_rate = summary_stats['triggered'] / effective_signals
            print(f"  Trigger rate:          {trigger_rate:.1%}")

    print()
    print("  " + "=" * 50)
    print("  PORTFOLIO PERFORMANCE")
    print("  " + "=" * 50)
    print(f"  Starting Capital:      ${starting_capital:,.2f}")
    print(f"  Final Capital:         ${current_capital:,.2f}")
    total_return = (current_capital / starting_capital - 1) * 100
    return_color = '+' if total_return >= 0 else ''
    print(f"  Total Return:          {return_color}{total_return:.2f}%")
    print(f"  Total P&L:             ${return_color}{summary_stats['total_pnl_dollars']:,.2f}")

    print()
    print("  Trade Statistics:")
    print("  " + "-" * 40)
    print(f"  Winning trades:        {summary_stats['winning_trades']}")
    print(f"  Losing trades:         {summary_stats['losing_trades']}")

    if summary_stats['triggered'] > 0:
        win_rate = summary_stats['winning_trades'] / summary_stats['triggered']
        avg_pnl = summary_stats['total_pnl_dollars'] / summary_stats['triggered']
        print(f"  Win rate:              {win_rate:.1%}")
        print(f"  Avg P&L per trade:     ${avg_pnl:,.2f}")

    print()
    print("  Entry Breakdown (Hybrid Entry):")
    print(f"  Limit fills:           {summary_stats['entry_limit']}")
    print(f"  Pivot market entries:  {summary_stats['entry_pivot']}")

    print()
    print("  Exit Breakdown:")
    print(f"  Profit targets hit:    {summary_stats['profit_target_hits']}")
    print(f"  Stop losses hit:       {summary_stats['stop_loss_hits']}")
    print(f"  Breakeven exits:       {summary_stats['breakeven_hits']}")
    print(f"  Trailing stops hit:    {summary_stats['trailing_stop_hits']}")
    print(f"  EOD closes (15:57):    {summary_stats['eod_closes']}")

    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Portfolio history: {portfolio_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
