"""
Order Flow Imbalance (OFI) Calculation Module

Implements the OFI metric from:
Cont, R., Kukanov, A., & Stoikov, S. (2014).
"The Price Impact of Order Book Events"

OFI measures the net buying/selling pressure from changes in the
limit order book (bid and ask queues).
"""

import numpy as np
import pandas as pd
from typing import Union
import logging

logger = logging.getLogger(__name__)


def calculate_ofi(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Order Flow Imbalance using Cont (2014) algorithm.

    VECTORIZED IMPLEMENTATION - No for loops.

    The OFI captures the net order flow by measuring changes in the
    bid and ask queues. Positive OFI indicates buying pressure,
    negative indicates selling pressure.

    Algorithm (Cont 2014):
        For each time step t:

        Bid Flow:
            if bid_price[t] > bid_price[t-1]:
                bid_flow = bid_size[t]  (new buyers at higher price)
            elif bid_price[t] < bid_price[t-1]:
                bid_flow = -bid_size[t-1]  (buyers withdrew)
            else:
                bid_flow = bid_size[t] - bid_size[t-1]  (queue change)

        Ask Flow:
            if ask_price[t] > ask_price[t-1]:
                ask_flow = -ask_size[t-1]  (sellers withdrew)
            elif ask_price[t] < ask_price[t-1]:
                ask_flow = ask_size[t]  (new sellers at lower price)
            else:
                ask_flow = ask_size[t] - ask_size[t-1]  (queue change)

        OFI = Bid_Flow - Ask_Flow

    Args:
        df: DataFrame with columns:
            - bid_price: Best bid price
            - bid_size: Best bid size (shares)
            - ask_price: Best ask price
            - ask_size: Best ask size (shares)

    Returns:
        np.ndarray of OFI values (first value is 0)

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    n = len(df)
    if n == 0:
        return np.array([])

    if n == 1:
        return np.array([0.0])

    # Extract numpy arrays for speed
    bid_price = df['bid_price'].values.astype(float)
    bid_size = df['bid_size'].values.astype(float)
    ask_price = df['ask_price'].values.astype(float)
    ask_size = df['ask_size'].values.astype(float)

    # Calculate price changes (shifted by 1)
    bid_price_change = np.diff(bid_price)  # bid_price[1:] - bid_price[:-1]
    ask_price_change = np.diff(ask_price)  # ask_price[1:] - ask_price[:-1]

    # Calculate size changes
    bid_size_change = bid_size[1:] - bid_size[:-1]
    ask_size_change = ask_size[1:] - ask_size[:-1]

    # Bid Flow calculation (vectorized)
    # Case 1: bid_price increased -> bid_flow = new bid_size
    # Case 2: bid_price decreased -> bid_flow = -previous bid_size
    # Case 3: bid_price unchanged -> bid_flow = size change
    bid_flow = np.where(
        bid_price_change > 0,
        bid_size[1:],              # New buyers at higher price
        np.where(
            bid_price_change < 0,
            -bid_size[:-1],        # Buyers withdrew
            bid_size_change        # Queue change at same price
        )
    )

    # Ask Flow calculation (vectorized)
    # Case 1: ask_price increased -> ask_flow = -previous ask_size (sellers withdrew)
    # Case 2: ask_price decreased -> ask_flow = new ask_size (new sellers)
    # Case 3: ask_price unchanged -> ask_flow = size change
    ask_flow = np.where(
        ask_price_change > 0,
        -ask_size[:-1],            # Sellers withdrew
        np.where(
            ask_price_change < 0,
            ask_size[1:],          # New sellers at lower price
            ask_size_change        # Queue change at same price
        )
    )

    # OFI = Bid Flow - Ask Flow
    ofi = bid_flow - ask_flow

    # Prepend 0 for first observation (no previous data)
    ofi_full = np.concatenate([[0.0], ofi])

    logger.debug(f"Calculated OFI for {n} observations. "
                 f"Mean: {np.mean(ofi_full):.2f}, Std: {np.std(ofi_full):.2f}")

    return ofi_full


def calculate_ofi_cumulative(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate cumulative OFI (running sum of OFI).

    Useful for visualizing net order flow over time.

    Args:
        df: Same as calculate_ofi

    Returns:
        np.ndarray of cumulative OFI values
    """
    ofi = calculate_ofi(df)
    return np.cumsum(ofi)


def calculate_ofi_normalized(
    df: pd.DataFrame,
    window: int = 50
) -> np.ndarray:
    """
    Calculate rolling Z-score normalized OFI.

    Normalizes OFI to have mean 0 and std 1 over a rolling window,
    making it comparable across different stocks and time periods.

    Uses only past data to avoid look-ahead bias.

    Args:
        df: Same as calculate_ofi
        window: Rolling window size for normalization (default 50)

    Returns:
        np.ndarray of normalized OFI values
    """
    ofi = calculate_ofi(df)

    if len(ofi) < window:
        # Not enough data for rolling window, use expanding
        ofi_series = pd.Series(ofi)
        rolling_mean = ofi_series.expanding(min_periods=1).mean()
        rolling_std = ofi_series.expanding(min_periods=1).std()
    else:
        ofi_series = pd.Series(ofi)
        rolling_mean = ofi_series.rolling(window=window, min_periods=1).mean()
        rolling_std = ofi_series.rolling(window=window, min_periods=1).std()

    # Normalize (add small epsilon to avoid division by zero)
    normalized = (ofi_series - rolling_mean) / (rolling_std + 1e-8)

    return normalized.values


def calculate_ofi_delta(
    df: pd.DataFrame,
    periods: int = 5
) -> np.ndarray:
    """
    Calculate OFI momentum (change in OFI over n periods).

    Measures the acceleration/deceleration of order flow.

    Args:
        df: Same as calculate_ofi
        periods: Look-back periods for delta (default 5)

    Returns:
        np.ndarray of OFI delta values
    """
    ofi = calculate_ofi(df)
    ofi_series = pd.Series(ofi)

    delta = ofi_series.diff(periods).fillna(0)

    return delta.values


if __name__ == "__main__":
    # Example usage with synthetic data
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create synthetic order book data
    np.random.seed(42)
    n = 100

    # Simulate price movements
    base_bid = 100.0
    base_ask = 100.05

    bid_prices = base_bid + np.cumsum(np.random.choice([-0.01, 0, 0.01], n, p=[0.3, 0.4, 0.3]))
    ask_prices = bid_prices + 0.05 + np.random.uniform(0, 0.02, n)

    # Simulate sizes
    bid_sizes = np.random.randint(100, 1000, n)
    ask_sizes = np.random.randint(100, 1000, n)

    df = pd.DataFrame({
        'bid_price': bid_prices,
        'bid_size': bid_sizes,
        'ask_price': ask_prices,
        'ask_size': ask_sizes
    })

    print("Sample data:")
    print(df.head(10))

    # Calculate OFI
    ofi = calculate_ofi(df)
    print(f"\nOFI values (first 10):")
    print(ofi[:10])

    # Calculate cumulative OFI
    cum_ofi = calculate_ofi_cumulative(df)
    print(f"\nCumulative OFI (first 10):")
    print(cum_ofi[:10])

    # Calculate normalized OFI
    norm_ofi = calculate_ofi_normalized(df, window=20)
    print(f"\nNormalized OFI (first 10):")
    print(norm_ofi[:10])

    # Summary statistics
    print(f"\nOFI Statistics:")
    print(f"  Mean:   {np.mean(ofi):.2f}")
    print(f"  Std:    {np.std(ofi):.2f}")
    print(f"  Min:    {np.min(ofi):.2f}")
    print(f"  Max:    {np.max(ofi):.2f}")
    print(f"  Final Cumulative: {cum_ofi[-1]:.2f}")

    # Test with edge cases
    print("\n--- Edge Case Tests ---")

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=['bid_price', 'bid_size', 'ask_price', 'ask_size'])
    print(f"Empty DF: {calculate_ofi(empty_df)}")

    # Single row
    single_df = df.head(1)
    print(f"Single row: {calculate_ofi(single_df)}")
