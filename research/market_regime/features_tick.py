"""
Tick-Level Feature Calculations for HMM Market Regime Detection

This module implements accurate OFI and VPIN calculations using
tick-level trade and quote data from TimescaleDB.

Key differences from OHLC-based calculations:
- OFI: Computed from actual quote changes, not price bar positions
- VPIN: Uses BVC (Bulk Volume Classification) with actual bid/ask spreads

References:
- OFI: Cont, Kukanov, Stoikov (2014) "The Price Impact of Order Book Events"
- VPIN: Easley, Lopez de Prado, O'Hara (2012) "Flow Toxicity and Liquidity"
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ORDER FLOW IMBALANCE (OFI)
# =============================================================================

def calculate_ofi_from_quotes(
    quotes_df: pd.DataFrame,
    include_depth: bool = True
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI) from quote changes.

    OFI measures the net order flow at the best bid/ask levels.
    Based on Cont, Kukanov, Stoikov (2014).

    Formula:
        OFI_t = (bid_size_change * I(bid_up)) - (ask_size_change * I(ask_down))

    Where:
        - I(bid_up) = 1 if bid_price increased or stayed same, 0 if decreased
        - I(ask_down) = 1 if ask_price decreased or stayed same, 0 if increased
        - bid_size_change and ask_size_change are signed changes

    Args:
        quotes_df: DataFrame with columns: time, bid_price, bid_size, ask_price, ask_size
        include_depth: If True, include full depth contribution on price changes

    Returns:
        Series with OFI value for each quote update
    """
    if quotes_df.empty:
        return pd.Series(dtype=float)

    df = quotes_df.copy()

    # Calculate changes
    df['bid_price_change'] = df['bid_price'].diff()
    df['ask_price_change'] = df['ask_price'].diff()
    df['bid_size_change'] = df['bid_size'].diff()
    df['ask_size_change'] = df['ask_size'].diff()

    # IMPORTANT: Calculate previous sizes BEFORE filtering
    # This fixes the bug where .shift(1) after filtering gives wrong values
    df['prev_bid_size'] = df['bid_size'].shift(1)
    df['prev_ask_size'] = df['ask_size'].shift(1)

    # Initialize OFI
    ofi = pd.Series(0.0, index=df.index)

    # Bid side contribution
    # When bid goes up: new liquidity added (positive)
    # When bid stays same: size change matters
    # When bid goes down: liquidity removed (negative)

    bid_up = df['bid_price_change'] > 0
    bid_same = df['bid_price_change'] == 0
    bid_down = df['bid_price_change'] < 0

    if include_depth:
        # Full depth on price improvement
        ofi[bid_up] = df.loc[bid_up, 'bid_size']
        # FIXED: Use pre-computed prev_bid_size instead of .shift(1) after filter
        ofi[bid_down] = -df.loc[bid_down, 'prev_bid_size']
    else:
        # Only size changes
        ofi[bid_up] = df.loc[bid_up, 'bid_size_change']
        ofi[bid_down] = -df.loc[bid_down, 'bid_size_change']

    # When price unchanged, use size change
    ofi[bid_same] = df.loc[bid_same, 'bid_size_change']

    # Ask side contribution (negative for selling pressure)
    ask_up = df['ask_price_change'] > 0
    ask_same = df['ask_price_change'] == 0
    ask_down = df['ask_price_change'] < 0

    if include_depth:
        # OFI = bid_flow - ask_flow
        # ask_down (aggressive sellers): ask_flow = +ask_size → SUBTRACT from OFI
        ofi[ask_down] -= df.loc[ask_down, 'ask_size']
        # ask_up (sellers retreat): ask_flow = -prev_ask_size → ADD to OFI
        ofi[ask_up] += df.loc[ask_up, 'prev_ask_size']
    else:
        # Non-depth mode: use size changes only
        # ask_down: more supply added → bearish → subtract
        ofi[ask_down] -= df.loc[ask_down, 'ask_size_change']
        # ask_up: supply removed → bullish → add
        ofi[ask_up] += df.loc[ask_up, 'ask_size_change']

    # When ask price unchanged, use negative of size change
    ofi[ask_same] -= df.loc[ask_same, 'ask_size_change']

    # First row has no valid diff
    ofi.iloc[0] = 0

    return ofi


def aggregate_ofi_to_bars(
    quotes_df: pd.DataFrame,
    ofi: pd.Series,
    freq: str = '1min'
) -> pd.DataFrame:
    """
    Aggregate tick-level OFI to time bars.

    Args:
        quotes_df: DataFrame with 'time' column
        ofi: Series of tick-level OFI values
        freq: Pandas frequency string for bar size

    Returns:
        DataFrame with time index and aggregated OFI
    """
    df = pd.DataFrame({
        'time': quotes_df['time'],
        'ofi': ofi
    })
    df = df.set_index('time')

    # Aggregate: sum OFI within each bar
    bars = df.resample(freq).agg({
        'ofi': 'sum'
    }).dropna()

    return bars


def aggregate_ofi_to_volume_bars(
    quotes_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    ofi: pd.Series,
    bucket_volume: int = 10000
) -> pd.DataFrame:
    """
    Aggregate tick-level OFI to volume bars.

    Volume bars are more stable than time bars because they
    normalize for trading activity variations.

    Args:
        quotes_df: DataFrame with quote data and OFI
        trades_df: DataFrame with trade data (time, size)
        ofi: Series of tick-level OFI values
        bucket_volume: Volume per bar

    Returns:
        DataFrame with volume bar index and aggregated OFI
    """
    # Merge OFI with quotes
    df = pd.DataFrame({
        'time': quotes_df['time'],
        'ofi': ofi
    })

    # Assign trades to volume buckets
    trades = trades_df.copy()
    trades['cumvol'] = trades['size'].cumsum()
    trades['bucket'] = (trades['cumvol'] // bucket_volume).astype(int)

    # Merge quotes with nearest trade bucket
    df = df.sort_values('time')
    trades = trades.sort_values('time')

    # Use merge_asof to assign each quote to a bucket
    merged = pd.merge_asof(
        df,
        trades[['time', 'bucket']],
        on='time',
        direction='backward'
    )

    # Aggregate OFI by bucket
    bars = merged.groupby('bucket').agg({
        'time': 'last',
        'ofi': 'sum'
    }).reset_index()

    bars = bars.set_index('time')
    return bars[['ofi']]


# =============================================================================
# VOLUME-SYNCHRONIZED PROBABILITY OF INFORMED TRADING (VPIN)
# =============================================================================

def classify_trades_bvc(
    trades_df: pd.DataFrame,
    method: str = 'tick_rule'
) -> pd.Series:
    """
    Classify trades as buy or sell initiated.

    Methods:
        - 'tick_rule': Buy if price > previous price, sell if <
        - 'quote_rule': Buy if price >= ask, sell if <= bid
        - 'hybrid': Use quote rule when available, fall back to tick

    Args:
        trades_df: DataFrame with columns: time, price, size, bid_price, ask_price
        method: Classification method

    Returns:
        Series with values: 1 (buy), -1 (sell), 0 (unknown)
    """
    df = trades_df.copy()

    if method == 'tick_rule':
        # Simple tick rule: compare to previous trade
        price_change = df['price'].diff()
        classification = pd.Series(0, index=df.index)
        classification[price_change > 0] = 1   # Uptick = buy
        classification[price_change < 0] = -1  # Downtick = sell
        # Zero tick: use previous classification
        classification = classification.replace(0, np.nan).ffill().fillna(0)

    elif method == 'quote_rule':
        # Quote rule: compare to prevailing bid/ask
        if 'bid_price' not in df.columns or 'ask_price' not in df.columns:
            raise ValueError("quote_rule requires bid_price and ask_price columns")

        classification = pd.Series(0, index=df.index)
        classification[df['price'] >= df['ask_price']] = 1   # At or above ask = buy
        classification[df['price'] <= df['bid_price']] = -1  # At or below bid = sell

        # Midpoint trades: probabilistic assignment
        mid = (df['bid_price'] + df['ask_price']) / 2
        spread = df['ask_price'] - df['bid_price']
        at_mid = (classification == 0) & (spread > 0)

        if at_mid.any():
            # Probability based on position in spread
            prob_buy = (df.loc[at_mid, 'price'] - df.loc[at_mid, 'bid_price']) / spread[at_mid]
            random_vals = np.random.random(at_mid.sum())
            classification.loc[at_mid] = np.where(random_vals < prob_buy, 1, -1)

    elif method == 'hybrid':
        # Use quote rule if quotes available, else tick rule
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            classification = classify_trades_bvc(df, method='quote_rule')
            # Fill unknowns with tick rule
            unknown = classification == 0
            if unknown.any():
                tick_class = classify_trades_bvc(df, method='tick_rule')
                classification[unknown] = tick_class[unknown]
        else:
            classification = classify_trades_bvc(df, method='tick_rule')

    else:
        raise ValueError(f"Unknown classification method: {method}")

    return classification.astype(int)


def calculate_vpin_from_trades(
    trades_df: pd.DataFrame,
    bucket_volume: int = 10000,
    n_buckets: int = 50,
    classification_method: str = 'bvc'
) -> pd.DataFrame:
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

    VPIN is based on Easley, Lopez de Prado, O'Hara (2012).
    Uses Bulk Volume Classification (BVC) for continuous trade classification.

    Formula:
        VPIN = sum(|V_buy - V_sell|) / sum(V_total) over n buckets

    Args:
        trades_df: DataFrame with trades (must have bid_price/ask_price for BVC)
        bucket_volume: Volume per bucket (V)
        n_buckets: Number of buckets for rolling VPIN
        classification_method: 'bvc' (continuous), 'tick_rule', 'quote_rule', or 'hybrid'

    Returns:
        DataFrame with columns: time, vpin, bucket, buy_volume, sell_volume
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['time', 'vpin', 'bucket', 'buy_volume', 'sell_volume'])

    df = trades_df.copy()

    # Use continuous BVC (Bulk Volume Classification) for smoother results
    if classification_method == 'bvc' and 'bid_price' in df.columns and 'ask_price' in df.columns:
        # Position in spread: 0 = at bid, 1 = at ask
        spread = df['ask_price'] - df['bid_price']
        spread = spread.replace(0, 0.01)  # Avoid division by zero
        position = (df['price'] - df['bid_price']) / spread
        buy_probability = position.clip(0, 1)

        # Fractional volume assignment (continuous, not discrete)
        df['vol_buy'] = df['size'] * buy_probability
        df['vol_sell'] = df['size'] * (1 - buy_probability)
    else:
        # Fall back to discrete classification
        df['direction'] = classify_trades_bvc(df, method=classification_method if classification_method != 'bvc' else 'tick_rule')
        df['vol_buy'] = df['size'] * (df['direction'] == 1).astype(float)
        df['vol_sell'] = df['size'] * (df['direction'] == -1).astype(float)

    # Calculate cumulative volume and assign buckets
    df['cumvol'] = df['size'].cumsum()
    df['bucket'] = (df['cumvol'] // bucket_volume).astype(int)

    # Aggregate by bucket
    bucket_data = df.groupby('bucket').agg({
        'time': 'last',
        'size': 'sum',
        'vol_buy': 'sum',
        'vol_sell': 'sum'
    }).reset_index()

    bucket_data.columns = ['bucket', 'time', 'volume', 'buy_volume', 'sell_volume']

    # Calculate imbalance
    bucket_data['imbalance'] = (bucket_data['buy_volume'] - bucket_data['sell_volume']).abs()

    # Rolling VPIN over n buckets
    bucket_data['rolling_imbalance'] = bucket_data['imbalance'].rolling(n_buckets, min_periods=1).sum()
    bucket_data['rolling_volume'] = bucket_data['volume'].rolling(n_buckets, min_periods=1).sum()
    bucket_data['vpin'] = bucket_data['rolling_imbalance'] / bucket_data['rolling_volume']

    # VPIN is between 0 and 1
    bucket_data['vpin'] = bucket_data['vpin'].clip(0, 1)

    result = bucket_data[['time', 'vpin', 'bucket', 'buy_volume', 'sell_volume']].copy()
    result = result.set_index('time')

    logger.info(f"Calculated VPIN for {len(bucket_data)} volume buckets (bucket_size={bucket_volume}, method={classification_method})")

    return result


def resample_vpin_to_time_bars(
    vpin_df: pd.DataFrame,
    freq: str = '1min'
) -> pd.DataFrame:
    """
    Resample volume-bar VPIN to time bars.

    Takes the last VPIN value within each time bar.

    Args:
        vpin_df: DataFrame with VPIN indexed by time
        freq: Pandas frequency string

    Returns:
        DataFrame with time-indexed VPIN
    """
    return vpin_df[['vpin']].resample(freq).last().dropna()


# =============================================================================
# COMBINED FEATURE CALCULATION
# =============================================================================

def calculate_tick_features(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    freq: str = '1min',
    bucket_volume: int = 10000,
    vpin_buckets: int = 50
) -> pd.DataFrame:
    """
    Calculate all tick-level features and aggregate to time bars.

    Features computed:
        - ofi: Order Flow Imbalance (sum per bar)
        - vpin: Volume-Synchronized Probability of Informed Trading
        - volume: Total volume per bar
        - trades: Number of trades per bar
        - spread: Average bid-ask spread

    Args:
        trades_df: DataFrame with trades (time, price, size, bid_price, ask_price)
        quotes_df: DataFrame with quotes (time, bid_price, bid_size, ask_price, ask_size)
        freq: Bar frequency (e.g., '1min', '5min')
        bucket_volume: Volume per VPIN bucket
        vpin_buckets: Rolling window for VPIN

    Returns:
        DataFrame with features indexed by time
    """
    # Calculate OFI from quotes
    logger.info("Calculating OFI from quote changes...")
    ofi_series = calculate_ofi_from_quotes(quotes_df)
    ofi_bars = aggregate_ofi_to_bars(quotes_df, ofi_series, freq=freq)

    # Calculate VPIN from trades using continuous BVC
    logger.info("Calculating VPIN from trades (BVC method)...")
    vpin_df = calculate_vpin_from_trades(
        trades_df,
        bucket_volume=bucket_volume,
        n_buckets=vpin_buckets,
        classification_method='bvc' if 'bid_price' in trades_df.columns else 'tick_rule'
    )
    vpin_bars = resample_vpin_to_time_bars(vpin_df, freq=freq)

    # Aggregate trades to bars
    trades_resampled = trades_df.set_index('time').resample(freq).agg({
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum'
    })
    trades_resampled.columns = ['open', 'high', 'low', 'close', 'volume']

    # Calculate returns
    trades_resampled['return'] = np.log(trades_resampled['close'] / trades_resampled['close'].shift(1))

    # Aggregate quotes to bars (average spread)
    quotes_bars = quotes_df.set_index('time').resample(freq).agg({
        'bid_price': 'mean',
        'ask_price': 'mean'
    })
    quotes_bars['spread'] = quotes_bars['ask_price'] - quotes_bars['bid_price']

    # Combine all features
    features = pd.DataFrame(index=trades_resampled.index)
    features['ofi'] = ofi_bars['ofi']
    features['vpin'] = vpin_bars['vpin']
    features['volume'] = trades_resampled['volume']
    features['return'] = trades_resampled['return']
    features['spread'] = quotes_bars['spread']
    features['close'] = trades_resampled['close']
    features['high'] = trades_resampled['high']
    features['low'] = trades_resampled['low']

    # Forward fill VPIN (it updates less frequently than time bars)
    features['vpin'] = features['vpin'].ffill()

    # Drop rows with missing data
    features = features.dropna()

    logger.info(f"Generated {len(features)} feature bars at {freq} frequency")

    return features


# =============================================================================
# Z-SCORE NORMALIZATION (for HMM)
# =============================================================================

def zscore_normalize(
    series: pd.Series,
    window: int = 50,
    min_periods: int = 10
) -> pd.Series:
    """
    Calculate rolling z-score normalization.

    Uses only past data to prevent look-ahead bias.

    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum observations required

    Returns:
        Z-score normalized series
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z = (series - rolling_mean) / rolling_std
    return z.fillna(0)


def normalize_features_for_hmm(
    features_df: pd.DataFrame,
    zscore_window: int = 50
) -> pd.DataFrame:
    """
    Normalize features for HMM training.

    - OFI: z-score normalized
    - VPIN: Already 0-1, no normalization needed
    - Return: z-score normalized
    - Volume: z-score normalized (log scale)

    Args:
        features_df: DataFrame with raw features
        zscore_window: Window for z-score calculation

    Returns:
        DataFrame with normalized features
    """
    df = features_df.copy()

    # Z-score normalize OFI
    df['ofi_z'] = zscore_normalize(df['ofi'], window=zscore_window)

    # VPIN is already 0-1, keep as-is
    df['vpin_norm'] = df['vpin']

    # Z-score normalize returns
    df['return_z'] = zscore_normalize(df['return'], window=zscore_window)

    # Z-score normalize log volume
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_z'] = zscore_normalize(df['log_volume'], window=zscore_window)

    # Calculate realized volatility (rolling std of returns)
    df['volatility'] = df['return'].rolling(window=20, min_periods=5).std()
    df['volatility_z'] = zscore_normalize(df['volatility'], window=zscore_window)

    # Select HMM input features
    hmm_features = df[['ofi_z', 'vpin_norm', 'return_z', 'volatility_z']].copy()
    hmm_features.columns = ['ofi', 'vpin', 'log_return', 'volatility']

    return hmm_features.dropna()


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    print("Creating sample data...")

    np.random.seed(42)
    n_quotes = 1000
    n_trades = 500

    times = pd.date_range('2025-12-23 09:30:00', periods=n_quotes, freq='100ms')

    # Sample quotes
    base_price = 620.0
    quotes_df = pd.DataFrame({
        'time': times,
        'bid_price': base_price + np.random.randn(n_quotes).cumsum() * 0.01 - 0.05,
        'bid_size': np.random.randint(100, 1000, n_quotes),
        'ask_price': base_price + np.random.randn(n_quotes).cumsum() * 0.01 + 0.05,
        'ask_size': np.random.randint(100, 1000, n_quotes)
    })
    quotes_df['ask_price'] = quotes_df['bid_price'] + np.random.uniform(0.01, 0.05, n_quotes)

    # Sample trades
    trade_times = times[np.random.choice(n_quotes, n_trades, replace=False)].sort_values()
    trades_df = pd.DataFrame({
        'time': trade_times,
        'price': base_price + np.random.randn(n_trades).cumsum() * 0.01,
        'size': np.random.randint(1, 100, n_trades)
    })

    # Add bid/ask to trades (simulate quote lookup)
    trades_df = pd.merge_asof(
        trades_df.sort_values('time'),
        quotes_df[['time', 'bid_price', 'ask_price']].sort_values('time'),
        on='time',
        direction='backward'
    )

    print(f"Sample data: {len(quotes_df)} quotes, {len(trades_df)} trades")

    # Test OFI calculation
    print("\nCalculating OFI...")
    ofi = calculate_ofi_from_quotes(quotes_df)
    print(f"OFI stats: mean={ofi.mean():.2f}, std={ofi.std():.2f}")

    # Test VPIN calculation
    print("\nCalculating VPIN...")
    vpin_df = calculate_vpin_from_trades(trades_df, bucket_volume=1000, n_buckets=10)
    print(f"VPIN stats: mean={vpin_df['vpin'].mean():.3f}, std={vpin_df['vpin'].std():.3f}")

    # Test combined features
    print("\nCalculating all features...")
    features = calculate_tick_features(
        trades_df, quotes_df,
        freq='1min',
        bucket_volume=1000,
        vpin_buckets=10
    )
    print(f"Features shape: {features.shape}")
    print(features.head())

    # Test normalization
    print("\nNormalizing for HMM...")
    hmm_features = normalize_features_for_hmm(features)
    print(f"HMM features shape: {hmm_features.shape}")
    print(hmm_features.describe())
