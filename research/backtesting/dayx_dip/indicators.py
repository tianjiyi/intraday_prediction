"""
Technical indicators for DayX Dip strategy.
Includes all dayx indicators plus RSI (Wilder's smoothing).
"""

import pandas as pd
import numpy as np

from .config import DayXDipConfig


# ---------------------------------------------------------------------------
# RSI (Wilder's smoothing via EWM)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    RSI using Wilder's exponential smoothing (alpha = 1/period).
    Standard implementation: gain/loss EWM then RS = avg_gain / avg_loss.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# CCI (Commodity Channel Index)
# ---------------------------------------------------------------------------

def _rolling_mad(arr: np.ndarray, period: int) -> np.ndarray:
    """Fast rolling mean absolute deviation using numpy strides."""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    shape = (n - period + 1, period)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    means = windows.mean(axis=1)
    result[period - 1:] = np.mean(np.abs(windows - means[:, None]), axis=1)
    return result


def cci(df: pd.DataFrame, period: int) -> pd.Series:
    """CCI = (typical_price - SMA) / (0.015 * mean_deviation)."""
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
    """Session-reset VWAP."""
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
# ATR (Wilder's)
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
# Exhaustion (consecutive down closes)
# ---------------------------------------------------------------------------

def exhaustion_down_count(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive lower closes (raw count, not boolean)."""
    down = (df["close"] < df["close"].shift(1)).astype(int)
    streak = down.groupby((down != down.shift()).cumsum()).cumcount() + 1
    return streak * down


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
    """Small body at top, long lower shadow (≥2x body), tiny upper shadow."""
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


# ---------------------------------------------------------------------------
# Master: compute all indicators
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame, cfg: DayXDipConfig) -> pd.DataFrame:
    """Attach all indicator columns to the DataFrame (in-place + return)."""
    df["rsi14"] = rsi(df["close"], cfg.rsi_period)
    df["cci14"] = cci(df, cfg.cci_period)

    bb = bollinger_bands(df, cfg.bb_period, cfg.bb_std)
    df["bb_mid"] = bb["bb_mid"]
    df["bb_upper"] = bb["bb_upper"]
    df["bb_lower"] = bb["bb_lower"]

    df["vwap"] = vwap(df)
    df["atr"] = atr(df, cfg.atr_period)

    df["exhaustion_down"] = exhaustion_down_count(df)
    df["hammer"] = hammer(df)
    df["bullish_engulfing"] = bullish_engulfing(df)

    # BB upper cross: close crosses above bb_upper (profit target)
    df["bb_upper_cross"] = df["close"] >= df["bb_upper"]

    return df
