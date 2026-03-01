"""
Volume-Synchronized Probability of Informed Trading (VPIN) Module

Implements VPIN from:
Easley, D., López de Prado, M., & O'Hara, M. (2012).
"Flow Toxicity and Liquidity in a High-frequency World"

VPIN measures the probability that trades are informed (toxic flow),
which often precedes large price movements.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


def classify_volume_tick_rule(
    prices: np.ndarray,
    volumes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify volume as Buy or Sell using the Tick Rule.

    Tick Rule:
        - If price > previous price: BUY
        - If price < previous price: SELL
        - If price == previous price: Same as previous classification

    Args:
        prices: Array of trade prices
        volumes: Array of trade volumes

    Returns:
        Tuple of (buy_volumes, sell_volumes) arrays
    """
    n = len(prices)
    if n == 0:
        return np.array([]), np.array([])

    if n == 1:
        # No previous price, assume neutral (split 50/50)
        return np.array([volumes[0] / 2]), np.array([volumes[0] / 2])

    # Calculate price changes
    price_diff = np.diff(prices)

    # Initialize classification (1 = buy, -1 = sell, 0 = unchanged)
    classification = np.zeros(n)
    classification[1:] = np.sign(price_diff)

    # Forward fill zeros (unchanged prices keep previous classification)
    last_class = 0
    for i in range(n):
        if classification[i] == 0:
            classification[i] = last_class
        else:
            last_class = classification[i]

    # First observation: assume neutral
    if classification[0] == 0:
        classification[0] = 1  # Default to buy if no info

    # Separate buy and sell volumes
    buy_volumes = np.where(classification >= 0, volumes, 0)
    sell_volumes = np.where(classification < 0, volumes, 0)

    return buy_volumes, sell_volumes


def classify_volume_bvc(
    prices: np.ndarray,
    volumes: np.ndarray,
    sigma: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify volume using Bulk Volume Classification (BVC).

    BVC uses the normalized price change to probabilistically assign
    volume to buy/sell buckets using a CDF approach.

    Formula:
        Z = (price_change - mean) / sigma
        P(buy) = CDF(Z) using standard normal
        buy_volume = volume * P(buy)
        sell_volume = volume * (1 - P(buy))

    Args:
        prices: Array of trade prices
        volumes: Array of trade volumes
        sigma: Standard deviation of price changes (auto-calculated if None)

    Returns:
        Tuple of (buy_volumes, sell_volumes) arrays
    """
    from scipy.stats import norm

    n = len(prices)
    if n == 0:
        return np.array([]), np.array([])

    if n == 1:
        return np.array([volumes[0] / 2]), np.array([volumes[0] / 2])

    # Calculate log returns
    log_returns = np.diff(np.log(prices))

    # Estimate sigma if not provided
    if sigma is None:
        sigma = np.std(log_returns)
        if sigma == 0:
            sigma = 1e-8  # Avoid division by zero

    # Normalize returns
    z_scores = log_returns / sigma

    # Calculate buy probability using normal CDF
    p_buy = norm.cdf(z_scores)

    # Prepend 0.5 for first observation (no previous price)
    p_buy_full = np.concatenate([[0.5], p_buy])

    # Allocate volumes
    buy_volumes = volumes * p_buy_full
    sell_volumes = volumes * (1 - p_buy_full)

    return buy_volumes, sell_volumes


def calculate_vpin(
    df: pd.DataFrame,
    n_buckets: int = 50,
    volume_bucket_size: Optional[int] = None,
    classification_method: Literal['tick', 'bvc'] = 'tick'
) -> np.ndarray:
    """
    Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN measures the absolute order imbalance normalized by total volume,
    averaged over n_buckets volume buckets.

    Algorithm:
        1. Classify each trade's volume as buy or sell
        2. Group trades into volume buckets (equal volume per bucket)
        3. For each bucket: OI = |Buy_Vol - Sell_Vol| / Total_Vol
        4. VPIN = Rolling average of OI over n_buckets

    VPIN ranges from 0 to 1:
        - 0: Perfectly balanced flow (equal buy/sell)
        - 1: Completely one-sided flow (all buy or all sell)

    High VPIN (>0.5) often precedes large price movements.

    Args:
        df: DataFrame with columns:
            - price: Trade price (or 'close' if volume bars)
            - volume: Trade volume
        n_buckets: Number of volume buckets for rolling average (default 50)
        volume_bucket_size: Volume per bucket (auto-calculated if None)
        classification_method: 'tick' or 'bvc' (default 'tick')

    Returns:
        np.ndarray of VPIN values (one per volume bucket)
    """
    # Determine price column
    price_col = 'price' if 'price' in df.columns else 'close'
    if price_col not in df.columns:
        raise ValueError("DataFrame must have 'price' or 'close' column")

    if 'volume' not in df.columns:
        raise ValueError("DataFrame must have 'volume' column")

    prices = df[price_col].values.astype(float)
    volumes = df['volume'].values.astype(float)

    n = len(prices)
    if n == 0:
        return np.array([])

    # Classify volume
    if classification_method == 'bvc':
        buy_volumes, sell_volumes = classify_volume_bvc(prices, volumes)
    else:
        buy_volumes, sell_volumes = classify_volume_tick_rule(prices, volumes)

    # Calculate cumulative volumes
    cum_buy = np.cumsum(buy_volumes)
    cum_sell = np.cumsum(sell_volumes)
    cum_total = np.cumsum(volumes)

    # Determine bucket size if not provided
    total_volume = volumes.sum()
    if volume_bucket_size is None:
        # Default: divide total volume into ~n_buckets buckets
        volume_bucket_size = max(total_volume / (n_buckets * 2), 1)

    # Assign observations to buckets
    bucket_ids = (cum_total / volume_bucket_size).astype(int)

    # Create bucket aggregation DataFrame
    bucket_df = pd.DataFrame({
        'bucket_id': bucket_ids,
        'buy_volume': buy_volumes,
        'sell_volume': sell_volumes,
        'volume': volumes
    })

    # Aggregate by bucket
    bucket_agg = bucket_df.groupby('bucket_id').agg({
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'volume': 'sum'
    })

    # Calculate order imbalance for each bucket
    # OI = |Buy - Sell| / Total
    total_vol = bucket_agg['volume'].values
    buy_vol = bucket_agg['buy_volume'].values
    sell_vol = bucket_agg['sell_volume'].values

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        order_imbalance = np.abs(buy_vol - sell_vol) / total_vol
        order_imbalance = np.nan_to_num(order_imbalance, nan=0.0)

    # Rolling VPIN over n_buckets
    if len(order_imbalance) < n_buckets:
        # Not enough buckets, use expanding window
        vpin = pd.Series(order_imbalance).expanding(min_periods=1).mean().values
    else:
        vpin = pd.Series(order_imbalance).rolling(
            window=n_buckets,
            min_periods=1
        ).mean().values

    logger.debug(
        f"Calculated VPIN: {len(vpin)} buckets, "
        f"Mean: {np.mean(vpin):.4f}, Max: {np.max(vpin):.4f}"
    )

    return vpin


def calculate_vpin_for_volume_bars(
    volume_bars_df: pd.DataFrame,
    n_buckets: int = 50
) -> np.ndarray:
    """
    Calculate VPIN for pre-aggregated volume bars (OHLC data).

    For OHLC bars, we estimate buy/sell volume using the bar's internal
    price movement relative to its range (high-low).

    Buy/Sell Split Formula (based on bar position):
        buy_ratio = (close - low) / (high - low)
        sell_ratio = 1 - buy_ratio

    This approximates how much of the bar's volume was buying vs selling
    pressure based on where the close is within the bar's range.

    Args:
        volume_bars_df: DataFrame of volume bars with columns:
            - open, high, low, close: OHLC prices
            - volume: Bar volume
        n_buckets: Rolling window for VPIN (default 50)

    Returns:
        np.ndarray of VPIN values (one per bar)
    """
    if len(volume_bars_df) == 0:
        return np.array([])

    # Extract OHLC and volume
    opens = volume_bars_df['open'].values.astype(float)
    highs = volume_bars_df['high'].values.astype(float)
    lows = volume_bars_df['low'].values.astype(float)
    closes = volume_bars_df['close'].values.astype(float)
    volumes = volume_bars_df['volume'].values.astype(float)

    # Calculate bar range
    bar_range = highs - lows

    # Avoid division by zero for doji/flat bars
    bar_range = np.where(bar_range < 1e-8, 1e-8, bar_range)

    # Estimate buy ratio based on close position within bar range
    # If close is at high -> buy_ratio = 1.0 (all buying)
    # If close is at low -> buy_ratio = 0.0 (all selling)
    # If close is in middle -> buy_ratio = 0.5 (balanced)
    buy_ratio = (closes - lows) / bar_range

    # Clamp to [0, 1] range
    buy_ratio = np.clip(buy_ratio, 0.0, 1.0)

    # Split volume into buy and sell
    buy_vols = volumes * buy_ratio
    sell_vols = volumes * (1 - buy_ratio)

    # Calculate order imbalance per bar: |Buy - Sell| / Total
    with np.errstate(divide='ignore', invalid='ignore'):
        order_imbalance = np.abs(buy_vols - sell_vols) / volumes
        order_imbalance = np.nan_to_num(order_imbalance, nan=0.0)

    # Clamp to valid range [0, 1]
    order_imbalance = np.clip(order_imbalance, 0.0, 1.0)

    # Rolling VPIN
    vpin = pd.Series(order_imbalance).rolling(
        window=n_buckets,
        min_periods=1
    ).mean().values

    logger.debug(
        f"VPIN for volume bars: {len(vpin)} values, "
        f"Mean: {np.mean(vpin):.4f}, Min: {np.min(vpin):.4f}, Max: {np.max(vpin):.4f}"
    )

    return vpin


if __name__ == "__main__":
    # Example usage with synthetic data
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create synthetic trade data
    np.random.seed(42)
    n_trades = 1000

    # Simulate prices with trend + noise
    base_price = 100.0
    trend = np.linspace(0, 2, n_trades)  # Upward trend
    noise = np.cumsum(np.random.randn(n_trades) * 0.1)
    prices = base_price + trend + noise

    # Simulate volumes (higher during trend)
    volumes = np.random.randint(100, 500, n_trades)

    df = pd.DataFrame({
        'price': prices,
        'volume': volumes
    })

    print("Sample data:")
    print(df.head(10))

    # Calculate VPIN with tick rule
    vpin_tick = calculate_vpin(df, n_buckets=20, classification_method='tick')
    print(f"\nVPIN (Tick Rule): {len(vpin_tick)} buckets")
    print(f"First 10: {vpin_tick[:10]}")

    # Calculate VPIN with BVC
    vpin_bvc = calculate_vpin(df, n_buckets=20, classification_method='bvc')
    print(f"\nVPIN (BVC): {len(vpin_bvc)} buckets")
    print(f"First 10: {vpin_bvc[:10]}")

    # Statistics
    print(f"\nVPIN Statistics (Tick Rule):")
    print(f"  Mean: {np.mean(vpin_tick):.4f}")
    print(f"  Std:  {np.std(vpin_tick):.4f}")
    print(f"  Min:  {np.min(vpin_tick):.4f}")
    print(f"  Max:  {np.max(vpin_tick):.4f}")

    print(f"\nVPIN Statistics (BVC):")
    print(f"  Mean: {np.mean(vpin_bvc):.4f}")
    print(f"  Std:  {np.std(vpin_bvc):.4f}")
    print(f"  Min:  {np.min(vpin_bvc):.4f}")
    print(f"  Max:  {np.max(vpin_bvc):.4f}")

    # Test volume bar VPIN
    print("\n--- Volume Bar VPIN Test ---")

    # Create synthetic volume bars
    volume_bars = pd.DataFrame({
        'open': prices[::10],
        'high': prices[::10] + 0.5,
        'low': prices[::10] - 0.5,
        'close': prices[5::10][:len(prices[::10])],
        'volume': np.full(len(prices[::10]), 10000)
    })

    vpin_bars = calculate_vpin_for_volume_bars(volume_bars, n_buckets=10)
    print(f"Volume Bar VPIN: {len(vpin_bars)} bars")
    print(f"First 10: {vpin_bars[:10]}")
