"""
Bar-by-bar trade simulator. Walks through prepared DataFrame and executes trades.
"""

import pandas as pd
import logging

from .config import DayXConfig
from .strategy import PositionManager, Trade

logger = logging.getLogger(__name__)


def run_simulation(df: pd.DataFrame, cfg: DayXConfig) -> tuple[list[Trade], list[tuple]]:
    """
    Walk forward through bars, check exits then entries.

    Args:
        df: DataFrame with indicators and signals computed
        cfg: Strategy configuration

    Returns:
        (closed_trades, equity_curve)
    """
    pm = PositionManager(cfg)

    flatten_h, flatten_m = map(int, cfg.eod_flatten_time.split(":"))

    for timestamp, row in df.iterrows():
        # Is this bar at or past the EOD flatten time?
        is_eod = (
            (timestamp.hour > flatten_h) or
            (timestamp.hour == flatten_h and timestamp.minute >= flatten_m)
        )

        # --- Check exits first ---
        pm.check_exits(
            timestamp=timestamp,
            high=row["high"],
            low=row["low"],
            close=row["close"],
            is_eod=is_eod,
        )

        # --- Check entries (only if no position and not EOD) ---
        if not pm.has_position and not is_eod and row.get("signal_any", False):
            atr_val = row.get("atr", 0)
            if atr_val > 0 and row["signal_name"]:
                pm.try_enter(
                    timestamp=timestamp,
                    price=row["close"],
                    direction=row["signal_direction"],
                    strategy=row["signal_name"],
                    atr_val=atr_val,
                    vwap_price=row.get("vwap", 0.0),
                    bb_upper=row.get("bb_upper", 0.0),
                    bb_lower=row.get("bb_lower", 0.0),
                )

        pm.record_equity(timestamp)

    # Force close any remaining position at last bar
    if pm.has_position:
        last_ts = df.index[-1]
        last_row = df.iloc[-1]
        pm._close_position(last_ts, last_row["close"], "backtest_end")

    logger.info(f"Simulation complete: {len(pm.closed_trades)} trades")
    return pm.closed_trades, pm.equity_curve
