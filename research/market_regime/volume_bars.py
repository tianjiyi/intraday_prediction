"""
Volume Bar Resampling Module

Resamples tick-level data into Volume Bars where each bar represents
a fixed volume threshold (e.g., 10,000 shares traded).

Volume bars normalize trading activity across time, making OFI and VPIN
calculations more meaningful compared to time-based bars.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def resample_to_volume_bars(
    tick_df: pd.DataFrame,
    volume_threshold: int = 10000
) -> pd.DataFrame:
    """
    Resample tick data into Volume Bars.

    Each bar is formed when cumulative volume reaches the threshold.
    This ensures each bar represents the same amount of trading activity,
    making cross-time comparisons more meaningful.

    Args:
        tick_df: Tick DataFrame with columns:
            - timestamp: DateTime of tick
            - price: Trade price
            - volume: Trade size (shares)
            - bid_price: Best bid price
            - bid_size: Best bid size
            - ask_price: Best ask price
            - ask_size: Best ask size
        volume_threshold: Shares per bar (default 10,000)

    Returns:
        DataFrame with columns:
            - time: Bar close timestamp
            - open: First price in bar
            - high: Highest price in bar
            - low: Lowest price in bar
            - close: Last price in bar
            - volume: Total volume (should be ~threshold)
            - bid_price: Last bid price in bar
            - bid_size: Last bid size in bar
            - ask_price: Last ask price in bar
            - ask_size: Last ask size in bar
    """
    if tick_df.empty:
        logger.warning("Empty tick DataFrame provided")
        return pd.DataFrame(columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'bid_price', 'bid_size', 'ask_price', 'ask_size'
        ])

    # Ensure proper column names
    required_cols = ['timestamp', 'price', 'volume']
    for col in required_cols:
        if col not in tick_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sort by timestamp
    df = tick_df.sort_values('timestamp').reset_index(drop=True)

    # Initialize accumulators
    bars = []
    current_bar = {
        'open': None,
        'high': float('-inf'),
        'low': float('inf'),
        'close': None,
        'volume': 0,
        'start_time': None,
        'end_time': None,
        'bid_price': None,
        'bid_size': None,
        'ask_price': None,
        'ask_size': None
    }

    for idx, row in df.iterrows():
        price = row['price']
        vol = row['volume']
        timestamp = row['timestamp']

        # Initialize bar start
        if current_bar['open'] is None:
            current_bar['open'] = price
            current_bar['start_time'] = timestamp

        # Update OHLC
        current_bar['high'] = max(current_bar['high'], price)
        current_bar['low'] = min(current_bar['low'], price)
        current_bar['close'] = price
        current_bar['end_time'] = timestamp

        # Update quote data (use last available)
        if 'bid_price' in df.columns and pd.notna(row.get('bid_price')):
            current_bar['bid_price'] = row['bid_price']
        if 'bid_size' in df.columns and pd.notna(row.get('bid_size')):
            current_bar['bid_size'] = row['bid_size']
        if 'ask_price' in df.columns and pd.notna(row.get('ask_price')):
            current_bar['ask_price'] = row['ask_price']
        if 'ask_size' in df.columns and pd.notna(row.get('ask_size')):
            current_bar['ask_size'] = row['ask_size']

        # Add volume
        current_bar['volume'] += vol

        # Check if bar is complete
        if current_bar['volume'] >= volume_threshold:
            bars.append({
                'time': current_bar['end_time'],
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close'],
                'volume': current_bar['volume'],
                'bid_price': current_bar['bid_price'],
                'bid_size': current_bar['bid_size'],
                'ask_price': current_bar['ask_price'],
                'ask_size': current_bar['ask_size']
            })

            # Reset for next bar
            current_bar = {
                'open': None,
                'high': float('-inf'),
                'low': float('inf'),
                'close': None,
                'volume': 0,
                'start_time': None,
                'end_time': None,
                'bid_price': current_bar['bid_price'],  # Carry forward
                'bid_size': current_bar['bid_size'],
                'ask_price': current_bar['ask_price'],
                'ask_size': current_bar['ask_size']
            }

    # Handle incomplete final bar (optional: include if > 50% threshold)
    if current_bar['volume'] > 0 and current_bar['volume'] >= volume_threshold * 0.5:
        bars.append({
            'time': current_bar['end_time'],
            'open': current_bar['open'],
            'high': current_bar['high'],
            'low': current_bar['low'],
            'close': current_bar['close'],
            'volume': current_bar['volume'],
            'bid_price': current_bar['bid_price'],
            'bid_size': current_bar['bid_size'],
            'ask_price': current_bar['ask_price'],
            'ask_size': current_bar['ask_size']
        })

    if not bars:
        logger.warning(
            f"No volume bars created. Total volume: {df['volume'].sum()}, "
            f"Threshold: {volume_threshold}"
        )
        return pd.DataFrame(columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'bid_price', 'bid_size', 'ask_price', 'ask_size'
        ])

    result_df = pd.DataFrame(bars)

    logger.info(
        f"Created {len(result_df)} volume bars from {len(df)} ticks "
        f"(threshold: {volume_threshold} shares)"
    )

    return result_df


def resample_to_volume_bars_vectorized(
    tick_df: pd.DataFrame,
    volume_threshold: int = 10000
) -> pd.DataFrame:
    """
    Vectorized version of volume bar resampling for better performance.

    Uses numpy operations instead of row-by-row iteration.
    Approximately 10-50x faster for large tick datasets.

    Args:
        tick_df: Same as resample_to_volume_bars
        volume_threshold: Shares per bar (default 10,000)

    Returns:
        Same as resample_to_volume_bars
    """
    if tick_df.empty:
        logger.warning("Empty tick DataFrame provided")
        return pd.DataFrame(columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'bid_price', 'bid_size', 'ask_price', 'ask_size'
        ])

    # Sort by timestamp
    df = tick_df.sort_values('timestamp').reset_index(drop=True)

    # Extract arrays for speed
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values

    # Optional quote data
    has_quotes = all(col in df.columns for col in
                     ['bid_price', 'bid_size', 'ask_price', 'ask_size'])
    if has_quotes:
        bid_prices = df['bid_price'].values
        bid_sizes = df['bid_size'].values
        ask_prices = df['ask_price'].values
        ask_sizes = df['ask_size'].values

    # Cumulative volume to determine bar boundaries
    cumvol = np.cumsum(volumes)

    # Bar assignments: floor(cumvol / threshold)
    bar_ids = (cumvol / volume_threshold).astype(int)

    # Create grouping DataFrame
    group_df = pd.DataFrame({
        'bar_id': bar_ids,
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })

    if has_quotes:
        group_df['bid_price'] = bid_prices
        group_df['bid_size'] = bid_sizes
        group_df['ask_price'] = ask_prices
        group_df['ask_size'] = ask_sizes

    # Aggregate using groupby
    agg_dict = {
        'timestamp': 'last',      # Bar close time
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    }

    if has_quotes:
        agg_dict['bid_price'] = 'last'
        agg_dict['bid_size'] = 'last'
        agg_dict['ask_price'] = 'last'
        agg_dict['ask_size'] = 'last'

    grouped = group_df.groupby('bar_id').agg(agg_dict)

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                       for col in grouped.columns]

    # Rename to standard OHLCV format
    result_df = pd.DataFrame({
        'time': grouped['timestamp_last'].values,
        'open': grouped['price_first'].values,
        'high': grouped['price_max'].values,
        'low': grouped['price_min'].values,
        'close': grouped['price_last'].values,
        'volume': grouped['volume_sum'].values
    })

    if has_quotes:
        result_df['bid_price'] = grouped['bid_price_last'].values
        result_df['bid_size'] = grouped['bid_size_last'].values
        result_df['ask_price'] = grouped['ask_price_last'].values
        result_df['ask_size'] = grouped['ask_size_last'].values
    else:
        result_df['bid_price'] = np.nan
        result_df['bid_size'] = np.nan
        result_df['ask_price'] = np.nan
        result_df['ask_size'] = np.nan

    logger.info(
        f"Created {len(result_df)} volume bars from {len(df)} ticks "
        f"(threshold: {volume_threshold} shares) [vectorized]"
    )

    return result_df


if __name__ == "__main__":
    # Example usage with synthetic data
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create synthetic tick data
    np.random.seed(42)
    n_ticks = 10000

    timestamps = pd.date_range('2024-12-20 09:30:00', periods=n_ticks, freq='100ms')
    base_price = 500.0
    prices = base_price + np.cumsum(np.random.randn(n_ticks) * 0.01)
    volumes = np.random.randint(10, 500, n_ticks)
    bid_prices = prices - 0.01
    ask_prices = prices + 0.01
    bid_sizes = np.random.randint(100, 1000, n_ticks)
    ask_sizes = np.random.randint(100, 1000, n_ticks)

    tick_df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'bid_price': bid_prices,
        'bid_size': bid_sizes,
        'ask_price': ask_prices,
        'ask_size': ask_sizes
    })

    print(f"Input: {len(tick_df)} ticks")
    print(tick_df.head())

    # Test iterative version
    bars_iter = resample_to_volume_bars(tick_df, volume_threshold=10000)
    print(f"\nIterative: {len(bars_iter)} bars")
    print(bars_iter.head())

    # Test vectorized version
    bars_vec = resample_to_volume_bars_vectorized(tick_df, volume_threshold=10000)
    print(f"\nVectorized: {len(bars_vec)} bars")
    print(bars_vec.head())

    # Compare timing
    import time

    start = time.time()
    for _ in range(10):
        resample_to_volume_bars(tick_df, volume_threshold=10000)
    iter_time = (time.time() - start) / 10

    start = time.time()
    for _ in range(10):
        resample_to_volume_bars_vectorized(tick_df, volume_threshold=10000)
    vec_time = (time.time() - start) / 10

    print(f"\nTiming (10 runs avg):")
    print(f"  Iterative:  {iter_time*1000:.2f}ms")
    print(f"  Vectorized: {vec_time*1000:.2f}ms")
    print(f"  Speedup:    {iter_time/vec_time:.1f}x")
