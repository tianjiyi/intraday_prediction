"""
Bear exhaustion dip signal for DayX Dip strategy.
Single long-only entry: below BB lower + RSI oversold + CCI oversold +
consecutive down closes + bullish pivot candle + time filter.
"""

import pandas as pd

from .config import DayXDipConfig


def bear_exhaustion_dip(df: pd.DataFrame, cfg: DayXDipConfig) -> pd.DataFrame:
    """
    Compute bear_exhaustion_dip signal column.

    Entry conditions (ALL required):
      1. close <= bb_lower * (1 + be_bb_zone_pct/100)  — at or near BB lower
      2. rsi14 < rsi_oversold
      3. cci14 < cci_oversold
      4. exhaustion_down >= exhaustion_lookback — N consecutive lower closes
      5. hammer OR bullish_engulfing
      6. Time: entry_earliest <= bar_time <= no_new_entries_after
    """
    early_h, early_m = map(int, cfg.entry_earliest.split(":"))
    cutoff_h, cutoff_m = map(int, cfg.no_new_entries_after.split(":"))

    after_earliest = (df.index.hour > early_h) | (
        (df.index.hour == early_h) & (df.index.minute >= early_m)
    )
    before_cutoff = (df.index.hour < cutoff_h) | (
        (df.index.hour == cutoff_h) & (df.index.minute <= cutoff_m)
    )
    time_ok = after_earliest & before_cutoff

    at_bb_lower = df["close"] <= df["bb_lower"] * (1 + cfg.be_bb_zone_pct / 100)
    rsi_ok = df["rsi14"] < cfg.rsi_oversold
    cci_ok = df["cci14"] < cfg.cci_oversold
    exhausted = df["exhaustion_down"] >= cfg.exhaustion_lookback
    pivot = df["hammer"] | df["bullish_engulfing"]

    df["signal_bear_exhaustion_dip"] = (
        at_bb_lower & rsi_ok & cci_ok & exhausted & pivot & time_ok
    )

    return df
