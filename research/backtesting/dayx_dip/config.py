"""
DayX Dip strategy configuration.
Long-only: buy bear exhaustion at BB lower band, exit at BB upper or stop loss.
"""

from dataclasses import dataclass


@dataclass
class DayXDipConfig:
    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std: float = 2.0

    # --- RSI ---
    rsi_period: int = 14
    rsi_oversold: float = 35.0

    # --- CCI ---
    cci_period: int = 14
    cci_oversold: float = -100.0

    # --- Bear exhaustion ---
    exhaustion_lookback: int = 3     # N consecutive lower closes required

    # --- Entry zone ---
    be_bb_zone_pct: float = 0.5     # % above bb_lower still counts as "at the band"

    # --- ATR / stop ---
    atr_period: int = 14
    stop_atr_mult: float = 2.0

    # --- Session ---
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    eod_flatten_time: str = "15:55"
    entry_earliest: str = "10:00"
    no_new_entries_after: str = "14:00"

    # --- Backtest ---
    initial_capital: float = 100_000.0
    position_size_pct: float = 1.0
    commission_per_share: float = 0.0
    slippage_per_share: float = 0.01

    # --- Data ---
    symbol: str = "QQQ"
    timeframe: str = "1Min"
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
