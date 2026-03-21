"""Build normalized observation vector from bar data."""

import numpy as np
import pandas as pd


def build_observation(row: pd.Series, has_position: bool,
                      volume_sma: float, session_start_min: int = 570) -> np.ndarray:
    """
    Build 11-feature observation vector, all normalized to ~[-1, 1].

    Features:
        0: bb_position — where price sits within BB bands [0=lower, 1=upper]
        1: rsi_norm — RSI / 100 [0, 1]
        2: cci_norm — CCI / 200 clipped to [-1, 1]
        3: vwap_dist — (close - vwap) / atr
        4: atr_pct — atr / close * 100 (volatility %)
        5: exhaust_down — consecutive down count / 10
        6: hammer — 0 or 1
        7: bull_engulfing — 0 or 1
        8: time_of_day — session progress [0, 1]
        9: has_position — 0 or 1
        10: rel_volume — volume / rolling avg volume
    """
    close = _safe(row, "close", 0)
    bb_upper = _safe(row, "bb_upper", close)
    bb_lower = _safe(row, "bb_lower", close)
    bb_range = bb_upper - bb_lower

    obs = np.zeros(11, dtype=np.float32)

    # 0: BB position
    obs[0] = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    # 1: RSI normalized
    obs[1] = _safe(row, "rsi14", 50) / 100.0

    # 2: CCI normalized
    obs[2] = np.clip(_safe(row, "cci14", 0) / 200.0, -1, 1)

    # 3: VWAP distance in ATR units
    atr_val = _safe(row, "atr", 1)
    vwap_val = _safe(row, "vwap", close)
    obs[3] = np.clip((close - vwap_val) / atr_val, -3, 3) / 3.0 if atr_val > 0 else 0

    # 4: ATR as % of price
    obs[4] = np.clip(atr_val / close * 100, 0, 5) / 5.0 if close > 0 else 0

    # 5: Exhaustion down count
    obs[5] = min(_safe(row, "exhaustion_down", 0), 10) / 10.0

    # 6: Hammer
    obs[6] = float(_safe(row, "hammer", False))

    # 7: Bullish engulfing
    obs[7] = float(_safe(row, "bullish_engulfing", False))

    # 8: Time of day (session progress)
    if hasattr(row, "name"):
        ts = row.name
    else:
        ts = row.get("timestamp", None)
    if ts is not None and hasattr(ts, "hour"):
        minutes_since_open = (ts.hour * 60 + ts.minute) - session_start_min
        obs[8] = np.clip(minutes_since_open / 390.0, 0, 1)
    else:
        obs[8] = 0.5

    # 9: Position flag
    obs[9] = float(has_position)

    # 10: Relative volume
    vol = _safe(row, "volume", 0)
    obs[10] = np.clip(vol / volume_sma, 0, 5) / 5.0 if volume_sma > 0 else 0

    return obs


def _safe(row, key, default):
    """Get value from row, returning default if NaN or missing."""
    try:
        v = row[key] if key in row.index else default
        if pd.isna(v):
            return default
        return float(v)
    except (KeyError, TypeError):
        return default
