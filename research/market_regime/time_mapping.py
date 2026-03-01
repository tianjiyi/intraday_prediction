"""
Time Mapping Module for Volume Bar Display

This module provides functions to map volume-bar data and predictions
to time-aligned grids for display in TradingView Lightweight Charts.

TradingView requires time-indexed, sequential data. Volume bars have
non-uniform time spacing (each bar completes when volume threshold is met).
This module bridges that gap using forward-fill interpolation.
"""

import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def map_volume_bars_to_time(
    volume_bar_df: pd.DataFrame,
    target_timeframe: str = '1T'
) -> pd.DataFrame:
    """
    Map volume-bar data to a regular time grid.

    Volume bars have non-uniform time spacing (each bar completes when
    volume threshold is met). This function creates a regular time grid
    and forward-fills volume bar values to each time point.

    Strategy:
    - Create regular time grid from first to last volume bar
    - Forward-fill: each time bar gets the most recent volume bar's values
    - This ensures the chart shows the "current" state at each time point

    Args:
        volume_bar_df: DataFrame with volume-bar data, indexed by completion time
                       Expected columns: ofi, vpin, close, hmm_state (optional)
        target_timeframe: Pandas resample string ('1T'=1min, '5T'=5min, etc.)

    Returns:
        DataFrame with time-aligned bars, forward-filled from volume bars
    """
    if volume_bar_df.empty:
        return volume_bar_df

    # Ensure datetime index
    df = volume_bar_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have 'time' or 'timestamp' column or DatetimeIndex")

    # Create time grid from first to last volume bar
    start = df.index.min().floor(target_timeframe)
    end = df.index.max().ceil(target_timeframe)
    time_grid = pd.date_range(start, end, freq=target_timeframe)

    if len(time_grid) == 0:
        logger.warning("Empty time grid created")
        return pd.DataFrame()

    # Reindex with forward fill
    # Each time bar gets the regime from the most recent completed volume bar
    combined_index = df.index.union(time_grid)
    result = df.reindex(combined_index).sort_index().ffill()

    # Keep only the time grid points
    result = result.loc[time_grid]

    # Ensure no NaN values at the start (before first volume bar)
    result = result.ffill().bfill()

    logger.info(
        f"Mapped {len(df)} volume bars to {len(result)} time bars "
        f"(timeframe: {target_timeframe})"
    )

    return result


def align_regime_history_to_time(
    volume_bar_history: List[Dict[str, Any]],
    timeframe_minutes: int = 1
) -> List[Dict[str, Any]]:
    """
    Convert volume-bar regime history to time-aligned format for frontend.

    This is called by regime_service.get_regime_history() when bar_type='volume'.
    The frontend expects time-aligned data for TradingView charts.

    Args:
        volume_bar_history: List of dicts with volume-bar regime predictions
                           Each dict has: timestamp, state, state_id, ofi, vpin, close
        timeframe_minutes: Target timeframe in minutes (1, 5, 15, 30, etc.)

    Returns:
        List of dicts with time-aligned regime predictions, same structure
        but with uniform timestamp spacing
    """
    if not volume_bar_history:
        return []

    # Convert to DataFrame
    df = pd.DataFrame(volume_bar_history)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    else:
        logger.warning("No timestamp column in volume bar history")
        return volume_bar_history

    # Map to time grid
    target_timeframe = f'{timeframe_minutes}T'
    mapped = map_volume_bars_to_time(df, target_timeframe)

    if mapped.empty:
        logger.warning("Time mapping produced empty result")
        return []

    # Convert back to list of dicts
    result = []
    for ts, row in mapped.iterrows():
        entry = {
            'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
            'bar_type': 'volume_mapped'  # Indicate this was volume bar data
        }

        # Copy all other columns
        for col in mapped.columns:
            val = row[col]
            if pd.isna(val):
                val = 0 if col in ['ofi', 'vpin'] else None
            elif isinstance(val, (np.integer, np.floating)):
                val = float(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            entry[col] = val

        result.append(entry)

    logger.info(f"Aligned {len(volume_bar_history)} volume bars to {len(result)} time bars")

    return result


def interpolate_regime_for_display(
    volume_bar_states: pd.Series,
    target_index: pd.DatetimeIndex
) -> pd.Series:
    """
    Interpolate regime states to match a target time index.

    Used when you have HMM states from volume bars but need to display
    them alongside time-bar candlesticks.

    Args:
        volume_bar_states: Series with HMM state IDs, indexed by volume bar time
        target_index: DatetimeIndex from time-bar candlesticks

    Returns:
        Series of HMM states aligned to target_index, forward-filled
    """
    if volume_bar_states.empty or len(target_index) == 0:
        return pd.Series(index=target_index, data=0)

    # Combine indices
    combined = volume_bar_states.index.union(target_index)
    aligned = volume_bar_states.reindex(combined).sort_index().ffill().bfill()

    # Return only target index values
    result = aligned.loc[target_index]

    return result


def calculate_time_bar_density(
    volume_bars_df: pd.DataFrame,
    timeframe: str = '1T'
) -> pd.DataFrame:
    """
    Calculate how many volume bars fall within each time bar.

    Useful for understanding market activity patterns:
    - Many volume bars in one time bar = high activity
    - Few volume bars = low activity

    Args:
        volume_bars_df: DataFrame with volume bars indexed by completion time
        timeframe: Time bar frequency

    Returns:
        DataFrame with time bars and volume bar counts
    """
    if volume_bars_df.empty:
        return pd.DataFrame(columns=['count', 'avg_ofi', 'max_vpin'])

    df = volume_bars_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index('time')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'time' column")

    # Resample to count volume bars per time bar
    agg_dict = {'close': 'count'}  # Count of volume bars

    if 'ofi' in df.columns:
        agg_dict['ofi'] = 'sum'
    if 'vpin' in df.columns:
        agg_dict['vpin'] = 'max'

    density = df.resample(timeframe).agg(agg_dict)
    density = density.rename(columns={'close': 'volume_bar_count'})

    return density


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create synthetic volume bar data with non-uniform timestamps
    np.random.seed(42)

    # Simulate volume bars completing at random intervals
    base_time = pd.Timestamp('2024-12-24 09:30:00')
    intervals = np.random.exponential(scale=30, size=100)  # ~30 sec avg
    cumulative_times = np.cumsum(intervals)
    timestamps = [base_time + pd.Timedelta(seconds=s) for s in cumulative_times]

    volume_bar_df = pd.DataFrame({
        'time': timestamps,
        'ofi': np.random.randn(100) * 1000,
        'vpin': np.random.uniform(0.3, 0.7, 100),
        'close': 500 + np.cumsum(np.random.randn(100) * 0.1),
        'hmm_state': np.random.randint(0, 4, 100)
    })

    print("Volume bars (non-uniform spacing):")
    print(volume_bar_df.head(10))
    print(f"\nTime span: {timestamps[0]} to {timestamps[-1]}")

    # Map to 1-minute time grid
    mapped = map_volume_bars_to_time(volume_bar_df.set_index('time'), '1T')
    print(f"\nMapped to 1-minute grid:")
    print(f"  Original: {len(volume_bar_df)} bars")
    print(f"  Mapped: {len(mapped)} bars")
    print(mapped.head(10))

    # Test the history alignment function
    history = [
        {'timestamp': str(t), 'state': f'State_{i%4}', 'state_id': i % 4,
         'ofi': float(volume_bar_df['ofi'].iloc[i]),
         'vpin': float(volume_bar_df['vpin'].iloc[i]),
         'close': float(volume_bar_df['close'].iloc[i])}
        for i, t in enumerate(timestamps[:20])
    ]

    aligned = align_regime_history_to_time(history, timeframe_minutes=5)
    print(f"\nAligned history (5-minute):")
    print(f"  Original: {len(history)} entries")
    print(f"  Aligned: {len(aligned)} entries")
    for entry in aligned[:5]:
        print(f"    {entry['timestamp']}: {entry.get('state', 'N/A')}")
