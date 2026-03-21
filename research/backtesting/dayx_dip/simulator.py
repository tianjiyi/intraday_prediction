"""
Bar-by-bar simulator for DayX Dip strategy.
"""

import logging
import pandas as pd

from .config import DayXDipConfig
from .strategy import PositionManager, Trade

logger = logging.getLogger(__name__)


def run_simulation(df: pd.DataFrame, cfg: DayXDipConfig) -> tuple[list[Trade], list[tuple]]:
    """
    Walk forward through bars, check exits then entries.

    Returns:
        (closed_trades, equity_curve)
    """
    pm = PositionManager(cfg)
    flatten_h, flatten_m = map(int, cfg.eod_flatten_time.split(":"))

    for timestamp, row in df.iterrows():
        is_eod = (
            (timestamp.hour > flatten_h) or
            (timestamp.hour == flatten_h and timestamp.minute >= flatten_m)
        )

        # Check exits first
        pm.check_exits(
            timestamp=timestamp,
            high=row["high"],
            low=row["low"],
            close=row["close"],
            bb_upper=row.get("bb_upper", float("inf")),
            is_eod=is_eod,
        )

        # Enter if signal and no position and not EOD
        if not pm.has_position and not is_eod and row.get("signal_bear_exhaustion_dip", False):
            atr_val = row.get("atr", 0)
            if atr_val > 0:
                pm.try_enter(
                    timestamp=timestamp,
                    price=row["close"],
                    atr_val=atr_val,
                )

        pm.record_equity(timestamp)

    # Force close remaining position at backtest end
    if pm.has_position:
        last_ts = df.index[-1]
        last_row = df.iloc[-1]
        pm._close(last_ts, last_row["close"], "backtest_end")

    logger.info(f"Simulation complete: {len(pm.closed_trades)} trades")
    return pm.closed_trades, pm.equity_curve
