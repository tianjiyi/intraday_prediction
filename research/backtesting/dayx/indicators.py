"""
Technical indicators ported from PineScript Day_X strategy.
All functions operate on a pandas DataFrame with OHLCV columns.
"""

import pandas as pd
import numpy as np

from .config import DayXConfig


# ---------------------------------------------------------------------------
# CCI (Commodity Channel Index)
# ---------------------------------------------------------------------------

def _rolling_mad(arr: np.ndarray, period: int) -> np.ndarray:
    """Fast rolling mean absolute deviation using numpy strides."""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # Vectorized rolling window via stride tricks
    shape = (n - period + 1, period)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    means = windows.mean(axis=1)
    # MAD = mean(|x - mean(x)|) for each window
    result[period - 1:] = np.mean(np.abs(windows - means[:, None]), axis=1)
    return result


def cci(df: pd.DataFrame, period: int) -> pd.Series:
    """
    CCI = (typical_price - SMA(typical_price)) / (0.015 * mean_deviation)
    Uses numpy stride tricks for fast rolling MAD (no lambda apply).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = pd.Series(_rolling_mad(tp.values.astype(np.float64), period), index=df.index)
    return (tp - sma) / (0.015 * mad)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(df: pd.DataFrame, period: int, std: float) -> pd.DataFrame:
    """Returns DataFrame with bb_mid, bb_upper, bb_lower columns."""
    mid = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": mid + std * sigma,
        "bb_lower": mid - std * sigma,
    }, index=df.index)


# ---------------------------------------------------------------------------
# VWAP (reset daily)
# ---------------------------------------------------------------------------

def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-reset VWAP using cumulative (price * volume) / cumulative volume."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]

    result = pd.Series(np.nan, index=df.index, dtype=float)

    for date in df["session_date"].unique():
        mask = df["session_date"] == date
        cum_tpv = tpv[mask].cumsum()
        cum_vol = df["volume"][mask].cumsum()
        result[mask] = cum_tpv / cum_vol.replace(0, np.nan)

    return result


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ATR."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# CCI Divergence
# ---------------------------------------------------------------------------

def cci_divergence_bullish(df: pd.DataFrame, cci_series: pd.Series,
                           lookback: int = 20) -> pd.Series:
    """
    Bullish divergence: price makes a lower low but CCI makes a higher low.
    Simplified: compare current bar vs lowest in lookback window.
    """
    price_low = df["low"]
    price_min = price_low.rolling(lookback).min()
    cci_min = cci_series.rolling(lookback).min()

    # Current bar is at or near the price low, but CCI is above its prior low
    at_price_low = price_low <= price_min * 1.002  # within 0.2%
    cci_higher = cci_series > cci_min + 5  # CCI meaningfully above its lookback low

    return at_price_low & cci_higher


def cci_divergence_bearish(df: pd.DataFrame, cci_series: pd.Series,
                           lookback: int = 20) -> pd.Series:
    """
    Bearish divergence: price makes a higher high but CCI makes a lower high.
    """
    price_high = df["high"]
    price_max = price_high.rolling(lookback).max()
    cci_max = cci_series.rolling(lookback).max()

    at_price_high = price_high >= price_max * 0.998
    cci_lower = cci_series < cci_max - 5

    return at_price_high & cci_lower


# ---------------------------------------------------------------------------
# Exhaustion signals
# ---------------------------------------------------------------------------

def exhaustion_up(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Count consecutive higher closes. Signal when count >= lookback."""
    up = (df["close"] > df["close"].shift(1)).astype(int)
    # Count consecutive ups
    streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    streak = streak * up  # zero out down bars
    return streak >= lookback


def exhaustion_down(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Count consecutive lower closes. Signal when count >= lookback."""
    down = (df["close"] < df["close"].shift(1)).astype(int)
    streak = down.groupby((down != down.shift()).cumsum()).cumcount() + 1
    streak = streak * down
    return streak >= lookback


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------

def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def hammer(df: pd.DataFrame) -> pd.Series:
    """
    Hammer: small body at top, long lower shadow (≥2x body), tiny upper shadow.
    """
    body = _body(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    candle_range = df["high"] - df["low"]

    return (
        (lower >= 2 * body) &
        (upper <= body * 0.5) &
        (candle_range > 0) &
        (body > 0)
    )


def inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Inverted hammer: small body at bottom, long upper shadow (≥2x body).
    """
    body = _body(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    candle_range = df["high"] - df["low"]

    return (
        (upper >= 2 * body) &
        (lower <= body * 0.5) &
        (candle_range > 0) &
        (body > 0)
    )


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current green candle's body fully engulfs prior red candle's body."""
    curr_bullish = df["close"] > df["open"]
    prev_bearish = df["close"].shift(1) < df["open"].shift(1)

    curr_body_low = df[["open", "close"]].min(axis=1)
    curr_body_high = df[["open", "close"]].max(axis=1)
    prev_body_low = df[["open", "close"]].shift(1).min(axis=1)
    prev_body_high = df[["open", "close"]].shift(1).max(axis=1)

    return (
        curr_bullish & prev_bearish &
        (curr_body_low <= prev_body_low) &
        (curr_body_high >= prev_body_high)
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current red candle's body fully engulfs prior green candle's body."""
    curr_bearish = df["close"] < df["open"]
    prev_bullish = df["close"].shift(1) > df["open"].shift(1)

    curr_body_low = df[["open", "close"]].min(axis=1)
    curr_body_high = df[["open", "close"]].max(axis=1)
    prev_body_low = df[["open", "close"]].shift(1).min(axis=1)
    prev_body_high = df[["open", "close"]].shift(1).max(axis=1)

    return (
        curr_bearish & prev_bullish &
        (curr_body_low <= prev_body_low) &
        (curr_body_high >= prev_body_high)
    )


# ---------------------------------------------------------------------------
# Master: compute all indicators and attach to DataFrame
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame, cfg: DayXConfig) -> pd.DataFrame:
    """Attach all indicator columns to the DataFrame (in-place + return)."""
    df["cci14"] = cci(df, cfg.cci_fast)
    df["cci20"] = cci(df, cfg.cci_slow)

    bb = bollinger_bands(df, cfg.bb_period, cfg.bb_std)
    df["bb_mid"] = bb["bb_mid"]
    df["bb_upper"] = bb["bb_upper"]
    df["bb_lower"] = bb["bb_lower"]

    df["vwap"] = vwap(df)
    df["atr"] = atr(df, cfg.atr_period)

    # CCI crossovers
    df["cci14_prev"] = df["cci14"].shift(1)
    df["cci14_cross_above_100"] = (df["cci14"] > 100) & (df["cci14_prev"] <= 100)
    df["cci14_cross_below_m100"] = (df["cci14"] < -100) & (df["cci14_prev"] >= -100)

    # Divergences
    df["bull_divergence"] = cci_divergence_bullish(df, df["cci14"])
    df["bear_divergence"] = cci_divergence_bearish(df, df["cci14"])

    # Exhaustion
    df["exhaust_up"] = exhaustion_up(df, cfg.exhaustion_lookback)
    df["exhaust_down"] = exhaustion_down(df, cfg.exhaustion_lookback)

    # Candlestick patterns
    df["hammer"] = hammer(df)
    df["inv_hammer"] = inverted_hammer(df)
    df["bull_engulfing"] = bullish_engulfing(df)
    df["bear_engulfing"] = bearish_engulfing(df)

    # Volatility regime: ATR percentile over rolling window (fast numpy)
    atr_vals = df["atr"].values
    lookback = cfg.vol_atr_lookback
    atr_pctile = np.full(len(atr_vals), np.nan)
    for i in range(lookback - 1, len(atr_vals)):
        window = atr_vals[i - lookback + 1:i + 1]
        atr_pctile[i] = np.sum(window <= atr_vals[i]) / lookback * 100
    df["atr_pctile"] = atr_pctile

    # Trend confirmation: SMA slope
    sma = df["close"].rolling(cfg.trend_sma_period).mean()
    df["sma_slope"] = sma - sma.shift(cfg.trend_slope_bars)

    # --- v2 Mean Reversion indicators ---

    # BB distance (% from band)
    df["bb_lower_dist"] = (df["close"] - df["bb_lower"]) / df["close"] * 100
    df["bb_upper_dist"] = (df["bb_upper"] - df["close"]) / df["close"] * 100

    # CCI turning (momentum shift)
    df["cci14_turning_up"] = (df["cci14"] > df["cci14"].shift(1)) & (df["cci14"].shift(1) <= df["cci14"].shift(2))
    df["cci14_turning_down"] = (df["cci14"] < df["cci14"].shift(1)) & (df["cci14"].shift(1) >= df["cci14"].shift(2))

    # VWAP slope (is VWAP rising or falling over N bars?)
    df["vwap_slope"] = df["vwap"] - df["vwap"].shift(cfg.vwap_slope_bars)

    # % of session bars where close > VWAP (trend day detector) — vectorized
    above_vwap = (df["close"] > df["vwap"]).astype(float)
    df["vwap_pct_above"] = above_vwap.groupby(df["session_date"]).expanding().mean().droplevel(0)

    # VWAP distance (% from VWAP)
    df["vwap_dist_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    return df
