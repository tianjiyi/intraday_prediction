"""Build observation vector from a signal bar's indicator values."""

import numpy as np
import pandas as pd

# Signal type encoding
SIGNAL_TYPES = {"buy_dip": 0.0, "sell_rip": 0.5, "trend_dip": 1.0}

N_FEATURES = 15


def build_signal_observation(row: pd.Series, signal_name: str,
                             volume_sma: float = 1.0) -> np.ndarray:
    """
    Build normalized observation vector from the bar where a signal fired.

    Returns np.ndarray of shape (14,) with all features in ~[-1, 1] or [0, 1].
    """
    obs = np.zeros(N_FEATURES, dtype=np.float32)

    # 0: BB position — where price sits within bands
    bb_upper = row.get("bb_upper", 0)
    bb_lower = row.get("bb_lower", 0)
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        obs[0] = np.clip((row["close"] - bb_lower) / bb_range, -0.5, 1.5)
    else:
        obs[0] = 0.5

    # 1: CCI14 normalized
    obs[1] = np.clip(row.get("cci14", 0) / 200.0, -1.0, 1.0)

    # 2: CCI20 normalized
    obs[2] = np.clip(row.get("cci20", 0) / 200.0, -1.0, 1.0)

    # 3: VWAP distance in ATR units
    atr_val = row.get("atr", 0)
    if atr_val > 0:
        vwap_dist = (row["close"] - row.get("vwap", row["close"])) / atr_val
        obs[3] = np.clip(vwap_dist / 3.0, -1.0, 1.0)

    # 4: ATR as % of price
    if row["close"] > 0:
        obs[4] = np.clip(atr_val / row["close"] * 100 / 5.0, 0, 1.0)

    # 5: Exhaustion down
    obs[5] = float(row.get("exhaust_down", False))

    # 6: Exhaustion up
    obs[6] = float(row.get("exhaust_up", False))

    # 7: Hammer
    obs[7] = float(row.get("hammer", False))

    # 8: Bullish engulfing
    obs[8] = float(row.get("bull_engulfing", False))

    # 9: Time of day (minutes since 9:30 / 390)
    if hasattr(row.name, 'hour'):
        ts = row.name
    else:
        ts = row.get("timestamp", None)
    if ts is not None:
        minutes = (ts.hour - 9) * 60 + ts.minute - 30
        obs[9] = np.clip(minutes / 390.0, 0, 1.0)

    # 10: Relative volume
    if volume_sma > 0:
        obs[10] = np.clip(row.get("volume", 0) / volume_sma / 5.0, 0, 1.0)

    # 11: Signal type
    obs[11] = SIGNAL_TYPES.get(signal_name, 0.5)

    # 12: BB lower distance (% from lower band)
    obs[12] = np.clip(row.get("bb_lower_dist", 0) / 2.0, 0, 1.0)

    # 13: BB upper distance (% from upper band)
    obs[13] = np.clip(row.get("bb_upper_dist", 0) / 2.0, 0, 1.0)

    # 14: RSI normalized to [0, 1]
    obs[14] = np.clip(row.get("rsi", 50) / 100.0, 0, 1.0)

    # Replace any NaN with 0
    np.nan_to_num(obs, copy=False)
    return obs
