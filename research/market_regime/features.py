"""
Feature Engineering Module for HMM Regime Detection

Provides feature calculation functions for the HMM model:
- Rolling Z-Score normalization (prevents look-ahead bias)
- Log returns
- Realized volatility

CRITICAL: All calculations use only past data to prevent future function leakage.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def rolling_z_score(
    series: Union[pd.Series, np.ndarray],
    window: int = 50
) -> pd.Series:
    """
    Standardize using ONLY past data (rolling window).

    CRITICAL: This function uses only past data to prevent look-ahead bias.
    Do NOT use global mean/std as it leaks future information.

    Formula:
        z[t] = (x[t] - rolling_mean[t-window:t]) / rolling_std[t-window:t]

    Args:
        series: Input data series
        window: Rolling window size (default 50)

    Returns:
        pd.Series of z-score normalized values
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    if len(series) == 0:
        return pd.Series(dtype=float)

    # Rolling mean and std using only past data
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()

    # Normalize with epsilon to avoid division by zero
    z_scores = (series - rolling_mean) / (rolling_std + 1e-8)

    return z_scores


def calculate_log_return(
    prices: Union[pd.Series, np.ndarray],
    periods: int = 1
) -> np.ndarray:
    """
    Calculate log returns from price series.

    Formula:
        log_return[t] = ln(price[t] / price[t-periods])

    Log returns have better statistical properties than simple returns:
    - Time-additive (can sum across time)
    - More normally distributed
    - Symmetric for gains and losses

    Args:
        prices: Price series (close prices)
        periods: Look-back periods for return calculation (default 1)

    Returns:
        np.ndarray of log returns (first `periods` values are 0)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    if len(prices) <= periods:
        return np.zeros(len(prices))

    prices = prices.astype(float)

    # Calculate log returns
    # log_return[t] = log(price[t] / price[t-periods])
    log_returns = np.log(prices[periods:] / prices[:-periods])

    # Prepend zeros for first `periods` observations
    log_returns_full = np.concatenate([np.zeros(periods), log_returns])

    return log_returns_full


def calculate_volatility(
    prices: Union[pd.Series, np.ndarray],
    window: int = 20,
    annualize: bool = False,
    trading_periods_per_year: int = 252 * 390  # For 1-min bars
) -> np.ndarray:
    """
    Calculate rolling realized volatility.

    Volatility is the standard deviation of log returns over a rolling window.

    Args:
        prices: Price series (close prices)
        window: Rolling window for volatility calculation (default 20)
        annualize: If True, annualize the volatility (default False)
        trading_periods_per_year: Periods per year for annualization

    Returns:
        np.ndarray of volatility values
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    if len(prices) < 2:
        return np.zeros(len(prices))

    # Calculate log returns
    log_returns = calculate_log_return(prices, periods=1)

    # Rolling standard deviation
    returns_series = pd.Series(log_returns)
    rolling_vol = returns_series.rolling(window=window, min_periods=1).std()

    vol = rolling_vol.values

    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(trading_periods_per_year)

    return vol


def calculate_realized_volatility(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 20
) -> np.ndarray:
    """
    Calculate Parkinson (1980) realized volatility using high-low range.

    This estimator is more efficient than close-to-close volatility
    as it uses intraday price information.

    Formula:
        sigma = sqrt(1/(4*ln(2)) * (ln(high/low))^2)

    Args:
        high: High prices
        low: Low prices
        close: Close prices (for fallback)
        window: Rolling window for averaging

    Returns:
        np.ndarray of realized volatility values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    n = len(high)
    if n == 0:
        return np.array([])

    # Parkinson volatility estimator
    # sqrt(1/(4*ln(2)) * (ln(H/L))^2)
    log_hl = np.log(high / low)
    parkinson_var = (log_hl ** 2) / (4 * np.log(2))

    # Rolling average
    parkinson_series = pd.Series(parkinson_var)
    rolling_var = parkinson_series.rolling(window=window, min_periods=1).mean()

    # Take square root for volatility
    vol = np.sqrt(rolling_var.values)

    return vol


def calculate_spread(
    bid_price: Union[pd.Series, np.ndarray],
    ask_price: Union[pd.Series, np.ndarray],
    mid_price: Optional[Union[pd.Series, np.ndarray]] = None
) -> np.ndarray:
    """
    Calculate bid-ask spread (relative).

    Formula:
        spread = (ask - bid) / mid
        where mid = (ask + bid) / 2

    Args:
        bid_price: Best bid prices
        ask_price: Best ask prices
        mid_price: Optional pre-calculated mid prices

    Returns:
        np.ndarray of relative spread values
    """
    if isinstance(bid_price, pd.Series):
        bid_price = bid_price.values
    if isinstance(ask_price, pd.Series):
        ask_price = ask_price.values

    if mid_price is None:
        mid_price = (ask_price + bid_price) / 2
    elif isinstance(mid_price, pd.Series):
        mid_price = mid_price.values

    # Relative spread
    with np.errstate(divide='ignore', invalid='ignore'):
        spread = (ask_price - bid_price) / mid_price
        spread = np.nan_to_num(spread, nan=0.0, posinf=0.0, neginf=0.0)

    return spread


def prepare_hmm_features(
    df: pd.DataFrame,
    ofi: np.ndarray,
    vpin: np.ndarray,
    z_score_window: int = 50,
    volatility_window: int = 20
) -> pd.DataFrame:
    """
    Prepare complete feature set for HMM training/inference.

    Features:
        1. OFI (z-score normalized)
        2. VPIN (already normalized 0-1)
        3. Log return (z-score normalized)
        4. Volatility (z-score normalized)

    All features use rolling normalization to prevent look-ahead bias.

    Args:
        df: DataFrame with at least 'close' column
        ofi: Pre-calculated OFI values
        vpin: Pre-calculated VPIN values
        z_score_window: Window for z-score normalization (default 50)
        volatility_window: Window for volatility calculation (default 20)

    Returns:
        DataFrame with columns: [ofi, vpin, log_return, volatility]
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    n = len(df)

    # Ensure arrays are the right length
    if len(ofi) != n:
        logger.warning(f"OFI length mismatch: {len(ofi)} vs {n}")
        # Pad or truncate
        if len(ofi) < n:
            ofi = np.concatenate([np.zeros(n - len(ofi)), ofi])
        else:
            ofi = ofi[-n:]

    if len(vpin) != n:
        logger.warning(f"VPIN length mismatch: {len(vpin)} vs {n}")
        if len(vpin) < n:
            vpin = np.concatenate([np.zeros(n - len(vpin)), vpin])
        else:
            vpin = vpin[-n:]

    # Calculate log returns
    log_returns = calculate_log_return(df['close'].values)

    # Calculate volatility
    volatility = calculate_volatility(df['close'].values, window=volatility_window)

    # Z-score normalize OFI, log_return, volatility
    ofi_z = rolling_z_score(pd.Series(ofi), window=z_score_window).values
    log_return_z = rolling_z_score(pd.Series(log_returns), window=z_score_window).values
    volatility_z = rolling_z_score(pd.Series(volatility), window=z_score_window).values

    # VPIN is already normalized (0-1 range), but can optionally z-score it too
    # For HMM, keeping it in 0-1 range is fine
    vpin_normalized = vpin

    # Create feature DataFrame
    features_df = pd.DataFrame({
        'ofi': ofi_z,
        'vpin': vpin_normalized,
        'log_return': log_return_z,
        'volatility': volatility_z
    })

    # Handle NaN/Inf values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)

    logger.info(f"Prepared {len(features_df)} feature vectors with 4 features")

    return features_df


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create synthetic price data
    np.random.seed(42)
    n = 200

    base_price = 100.0
    returns = np.random.randn(n) * 0.02  # 2% daily vol
    prices = base_price * np.exp(np.cumsum(returns))

    print("Sample prices:")
    print(prices[:10])

    # Test rolling z-score
    z_scores = rolling_z_score(pd.Series(prices), window=20)
    print(f"\nRolling Z-Scores (first 10):")
    print(z_scores.values[:10])

    # Test log returns
    log_returns = calculate_log_return(prices)
    print(f"\nLog Returns (first 10):")
    print(log_returns[:10])

    # Test volatility
    vol = calculate_volatility(prices, window=20)
    print(f"\nVolatility (first 10):")
    print(vol[:10])

    # Test with OHLC data
    high = prices * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = prices * (1 - np.abs(np.random.randn(n)) * 0.01)

    realized_vol = calculate_realized_volatility(high, low, prices, window=20)
    print(f"\nRealized Volatility (first 10):")
    print(realized_vol[:10])

    # Test spread calculation
    bid_prices = prices - 0.05
    ask_prices = prices + 0.05
    spread = calculate_spread(bid_prices, ask_prices)
    print(f"\nBid-Ask Spread (first 10):")
    print(spread[:10])

    # Test full feature preparation
    print("\n--- Full Feature Preparation ---")

    df = pd.DataFrame({
        'close': prices,
        'high': high,
        'low': low
    })

    # Mock OFI and VPIN (normally from other modules)
    mock_ofi = np.random.randn(n) * 100
    mock_vpin = np.random.uniform(0, 1, n)

    features = prepare_hmm_features(
        df, mock_ofi, mock_vpin,
        z_score_window=20,
        volatility_window=10
    )

    print("Feature DataFrame:")
    print(features.head(10))
    print("\nFeature Statistics:")
    print(features.describe())
