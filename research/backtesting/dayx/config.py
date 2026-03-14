"""
Day_X strategy configuration.
v1: Original PineScript port (long_trend, short_trend, long_dip, etc.)
v2: Mean Reversion + Trend Dip (buy_dip, sell_rip, trend_dip)
"""

from dataclasses import dataclass, field


@dataclass
class DayXConfig:
    # --- Indicator parameters ---
    cci_fast: int = 14
    cci_slow: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    exhaustion_lookback: int = 3

    # --- Opening range ---
    opening_range_bars: int = 1

    # --- v1 signal filters (kept for backward compat) ---
    require_exhaustion: bool = True
    require_cci20_alignment: bool = True
    vol_filter: bool = False
    vol_atr_lookback: int = 100
    vol_atr_max_pctile: float = 75.0
    trend_filter: bool = False
    trend_sma_period: int = 20
    trend_slope_bars: int = 5

    # --- v2 Mean Reversion zone parameters ---
    bb_zone_pct: float = 1.0         # Within X% of BB band = "in the zone"
    vwap_slope_bars: int = 6         # 6 bars * 5min = 30 min for VWAP slope
    trend_dip_vwap_pct: float = 0.2  # Within 0.2% of VWAP = "touching VWAP"
    trend_dip_above_pct: float = 0.6 # >60% of session bars above VWAP = trend day
    cci_neutral_lo: int = -50        # CCI neutral zone for trend_dip
    cci_neutral_hi: int = 50

    # --- v2 exit logic ---
    use_vwap_target: bool = False    # True = target VWAP for MR, False = fixed R
    mr_stop_atr_mult: float = 2.0   # Wider stops for mean reversion
    trend_dip_stop_atr_mult: float = 1.0  # Tighter stop for trend dip

    # --- Risk / position management ---
    stop_atr_mult: float = 2.0
    target1_r: float = 1.0
    target2_r: float = 2.0
    partial_exit_pct: float = 0.5
    trail_after_target1: bool = True
    trail_mode: str = "breakeven"
    trail_progressive_step: float = 0.5
    max_positions: int = 1

    # --- Session ---
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    eod_flatten_time: str = "15:55"
    no_new_entries_after: str = "15:30"

    # --- Time filter ---
    time_filter: bool = False
    entry_earliest: str = "10:00"
    entry_latest: str = "14:00"

    # --- Backtest ---
    initial_capital: float = 100_000.0
    position_size_pct: float = 1.0
    commission_per_share: float = 0.0
    slippage_per_share: float = 0.01

    # --- Data ---
    symbol: str = "QQQ"
    timeframe: str = "5Min"
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"

    # --- Enabled strategies ---
    strategies: list = field(default_factory=lambda: [
        "long_trend",
        "short_trend",
        "long_chaseUp",
        "long_dip",
        "short_chaseDown",
    ])
