"""
5 Day_X entry strategies ported from PineScript.
Each function returns a boolean Series tagged with the strategy name.
"""

import pandas as pd

from .config import DayXConfig


def long_trend(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Strong trend continuation long.
    CCI14 > 100 AND price > VWAP AND price > BB_mid AND price > opening_range_high.
    Optional: CCI20 > 0 for alignment, exhaustion_up for momentum confirmation.
    """
    cond = (
        (df["cci14"] > 100) &
        (df["close"] > df["vwap"]) &
        (df["close"] > df["bb_mid"]) &
        (df["close"] > df["or_high"])
    )
    if cfg.require_cci20_alignment:
        cond = cond & (df["cci20"] > 0)
    return cond


def short_trend(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Strong trend continuation short.
    CCI14 < -100 AND price < VWAP AND price < BB_mid AND price < opening_range_low.
    """
    cond = (
        (df["cci14"] < -100) &
        (df["close"] < df["vwap"]) &
        (df["close"] < df["bb_mid"]) &
        (df["close"] < df["or_low"])
    )
    if cfg.require_cci20_alignment:
        cond = cond & (df["cci20"] < 0)
    return cond


def long_chaseUp(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Momentum breakout long.
    CCI14 crosses above 100 AND price > VWAP AND (bullish engulfing OR hammer).
    """
    cond = (
        df["cci14_cross_above_100"] &
        (df["close"] > df["vwap"]) &
        (df["bull_engulfing"] | df["hammer"])
    )
    if cfg.require_exhaustion:
        # Exhaustion up confirms sustained buying pressure
        cond = cond & df["exhaust_up"]
    return cond


def long_dip(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Mean reversion long from oversold.
    CCI14 < -100 (oversold) AND price near BB_lower AND bullish divergence AND hammer.
    """
    near_bb_lower = df["close"] <= df["bb_lower"] * 1.005  # within 0.5% of lower band

    cond = (
        (df["cci14"] < -100) &
        near_bb_lower &
        df["bull_divergence"] &
        (df["hammer"] | df["bull_engulfing"])
    )
    return cond


def short_chaseDown(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Momentum breakdown short.
    CCI14 crosses below -100 AND price < VWAP AND (bearish engulfing OR inverted hammer).
    """
    cond = (
        df["cci14_cross_below_m100"] &
        (df["close"] < df["vwap"]) &
        (df["bear_engulfing"] | df["inv_hammer"])
    )
    if cfg.require_exhaustion:
        cond = cond & df["exhaust_down"]
    return cond


# ---------------------------------------------------------------------------
# v2 Mean Reversion strategies
# ---------------------------------------------------------------------------

def buy_dip(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Mean reversion long: price recently hit BB lower, RSI now recovering.
    Extreme:   low dipped below BB_lower within last N bars
    Timing:    RSI crosses back above 30 (was oversold, now recovering)
    Entry:     bullish candle (hammer, engulfing, or rejection wick)
    """
    bb_touched = (df["low"] <= df["bb_lower"]).rolling(
        cfg.bb_lookback_bars, min_periods=1
    ).max().astype(bool)
    rsi_recovering = df["rsi_cross_above_30"]
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    rejection_wick = (lower_wick > body) & (body > 0)
    reversal = df["hammer"] | df["bull_engulfing"] | rejection_wick

    return bb_touched & rsi_recovering & reversal


def sell_rip(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Mean reversion short: price recently hit BB upper, RSI now rejecting.
    Extreme:   high pushed above BB_upper within last N bars
    Timing:    RSI crosses back below 70 (was overbought, now fading)
    Entry:     bearish candle (inv hammer, engulfing, or rejection top wick)
    """
    bb_touched = (df["high"] >= df["bb_upper"]).rolling(
        cfg.bb_lookback_bars, min_periods=1
    ).max().astype(bool)
    rsi_rejecting = df["rsi_cross_below_70"]
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    rejection_top = (upper_wick > body) & (body > 0)
    reversal = df["inv_hammer"] | df["bear_engulfing"] | rejection_top

    return bb_touched & rsi_rejecting & reversal


def trend_dip(df: pd.DataFrame, cfg: DayXConfig) -> pd.Series:
    """
    Buy dip in strong uptrend: pullback to VWAP in a trending day.
    Trend:   VWAP slope positive + >60% of session bars above VWAP
    Zone:    close pulls back to within 0.2% of VWAP
    Confirm: CCI14 between -50 and 50 (neutral, healthy pullback)
    Trigger: hammer OR bullish engulfing OR green candle
    """
    trend_up = (df["vwap_slope"] > 0) & (df["vwap_pct_above"] >= cfg.trend_dip_above_pct)
    near_vwap = df["vwap_dist_pct"].abs() <= cfg.trend_dip_vwap_pct
    cci_neutral = (df["cci14"] >= cfg.cci_neutral_lo) & (df["cci14"] <= cfg.cci_neutral_hi)
    trigger = df["hammer"] | df["bull_engulfing"] | (df["close"] > df["open"])

    return trend_up & near_vwap & cci_neutral & trigger


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_FUNCS = {
    # v1 strategies
    "long_trend": (long_trend, "long"),
    "short_trend": (short_trend, "short"),
    "long_chaseUp": (long_chaseUp, "long"),
    "long_dip": (long_dip, "long"),
    "short_chaseDown": (short_chaseDown, "short"),
    # v2 strategies
    "buy_dip": (buy_dip, "long"),
    "sell_rip": (sell_rip, "short"),
    "trend_dip": (trend_dip, "long"),
}


def generate_signals(df: pd.DataFrame, cfg: DayXConfig) -> pd.DataFrame:
    """
    Evaluate all enabled strategies and add signal columns.

    Adds columns:
        signal_<name>: bool — True when strategy fires
        signal_any: bool — True when any strategy fires
        signal_name: str — name of first firing strategy (priority order)
        signal_direction: str — 'long' or 'short'
    """
    df["signal_any"] = False
    df["signal_name"] = ""
    df["signal_direction"] = ""

    # No entries after cutoff time
    cutoff_h, cutoff_m = map(int, cfg.no_new_entries_after.split(":"))
    time_ok = (df.index.hour < cutoff_h) | (
        (df.index.hour == cutoff_h) & (df.index.minute <= cutoff_m)
    )

    # Improvement 2: Time window filter
    if cfg.time_filter:
        early_h, early_m = map(int, cfg.entry_earliest.split(":"))
        late_h, late_m = map(int, cfg.entry_latest.split(":"))
        after_earliest = (df.index.hour > early_h) | (
            (df.index.hour == early_h) & (df.index.minute >= early_m)
        )
        before_latest = (df.index.hour < late_h) | (
            (df.index.hour == late_h) & (df.index.minute <= late_m)
        )
        time_ok = time_ok & after_earliest & before_latest

    # Improvement 3: Volatility regime filter
    vol_ok = pd.Series(True, index=df.index)
    if cfg.vol_filter:
        vol_ok = df["atr_pctile"] <= cfg.vol_atr_max_pctile

    for name in cfg.strategies:
        if name not in STRATEGY_FUNCS:
            continue
        func, direction = STRATEGY_FUNCS[name]
        sig = func(df, cfg) & time_ok & vol_ok

        # Improvement 4: Trend confirmation for trend entries
        if cfg.trend_filter and name in ("long_trend",):
            sig = sig & (df["sma_slope"] > 0)
        if cfg.trend_filter and name in ("short_trend",):
            sig = sig & (df["sma_slope"] < 0)

        df[f"signal_{name}"] = sig

        # First-match priority: don't overwrite if already signaled
        new_signal = sig & ~df["signal_any"]
        df.loc[new_signal, "signal_name"] = name
        df.loc[new_signal, "signal_direction"] = direction
        df["signal_any"] = df["signal_any"] | sig

    return df
