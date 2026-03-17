"""
Signal service: wraps the DayX backtesting strategy to compute trading signals
on live/historical bar data.

Uses per-timeframe optimized configs from Optuna trials.
"""

import logging
import sys
import types
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Prevent research/backtesting/__init__.py from importing broken Kronos deps
sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
sys.modules["research.backtesting"].__path__ = [
    str(Path(__file__).resolve().parents[2] / "research" / "backtesting")
]

from research.backtesting.dayx.config import DayXConfig
from research.backtesting.dayx.data_loader import (
    filter_rth,
    add_session_markers,
    compute_opening_range,
)
from research.backtesting.dayx.indicators import compute_all
from research.backtesting.dayx.signals import generate_signals
from research.backtesting.dayx.simulator import run_simulation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared fixed params across all timeframes
# ---------------------------------------------------------------------------
FIXED_PARAMS = {
    "strategies": ["buy_dip", "sell_rip", "trend_dip"],
    "use_vwap_target": False,
    "trail_mode": "progressive",
    "time_filter": True,
    "require_exhaustion": False,
    "require_cci20_alignment": False,
}

# ---------------------------------------------------------------------------
# Per-timeframe optimized configs (Optuna best trials)
# ---------------------------------------------------------------------------

def _minutes_to_time(m: int) -> str:
    """Convert minutes-from-midnight to HH:MM string."""
    return f"{m // 60:02d}:{m % 60:02d}"


def _build_config(timeframe: str, params: dict) -> DayXConfig:
    """Build a DayXConfig from fixed + per-timeframe Optuna params."""
    merged = {**FIXED_PARAMS, **params}
    # Convert entry minutes to HH:MM
    if "entry_earliest_min" in merged:
        merged["entry_earliest"] = _minutes_to_time(merged.pop("entry_earliest_min"))
    if "entry_latest_min" in merged:
        merged["entry_latest"] = _minutes_to_time(merged.pop("entry_latest_min"))
    merged["timeframe"] = timeframe
    return DayXConfig(**merged)


_OPTUNA_1MIN = {
    "atr_period": 20,
    "bb_period": 25,
    "bb_std": 2.607561262448639,
    "bb_zone_pct": 1.519771310330316,
    "cci_fast": 8,
    "cci_neutral_hi": 87,
    "cci_neutral_lo": -98,
    "entry_earliest_min": 585,
    "entry_latest_min": 856,
    "exhaustion_lookback": 6,
    "partial_exit_pct": 0.379748642740468,
    "stop_atr_mult": 2.445185179418562,
    "target1_r": 1.9061932802548016,
    "target2_r": 1.9810203997568108,
    "trail_progressive_step": 0.841185459816253,
    "trend_dip_above_pct": 0.6472578948587814,
    "trend_dip_vwap_pct": 0.37372826786089836,
}

_OPTUNA_3MIN = {
    "atr_period": 18,
    "bb_period": 25,
    "bb_std": 1.7174070072031054,
    "bb_zone_pct": 1.1496938645805672,
    "cci_fast": 18,
    "cci_neutral_hi": 98,
    "cci_neutral_lo": -96,
    "entry_earliest_min": 597,
    "entry_latest_min": 877,
    "exhaustion_lookback": 8,
    "partial_exit_pct": 0.31331708845805645,
    "stop_atr_mult": 2.9129001572148017,
    "target1_r": 0.5560863645057493,
    "target2_r": 2.4469755196181096,
    "trail_progressive_step": 0.8388121485046761,
    "trend_dip_above_pct": 0.42490883781903666,
    "trend_dip_vwap_pct": 0.29172223403907527,
}

_OPTUNA_5MIN = {
    "atr_period": 16,
    "bb_period": 29,
    "bb_std": 2.91271247302306,
    "bb_zone_pct": 0.9358179982543291,
    "cci_fast": 20,
    "cci_neutral_hi": 63,
    "cci_neutral_lo": -89,
    "entry_earliest_min": 614,
    "entry_latest_min": 821,
    "exhaustion_lookback": 2,
    "partial_exit_pct": 0.3694705392406415,
    "stop_atr_mult": 2.219556234335626,
    "target1_r": 1.1936817048904038,
    "target2_r": 3.2628323407688504,
    "trail_progressive_step": 0.8267614003908604,
    "trend_dip_above_pct": 0.4391618291772076,
    "trend_dip_vwap_pct": 0.49983993002375887,
}

TIMEFRAME_CONFIGS: dict[str, DayXConfig] = {
    "1Min": _build_config("1Min", _OPTUNA_1MIN),
    "3Min": _build_config("3Min", _OPTUNA_3MIN),
    "5Min": _build_config("5Min", _OPTUNA_5MIN),
}

# ---------------------------------------------------------------------------
# Signal direction mapping
# ---------------------------------------------------------------------------
_LONG_SIGNALS = {"buy_dip", "long_trend", "long_dip", "long_chaseUp", "trend_dip"}
_SHORT_SIGNALS = {"sell_rip", "short_trend", "short_chaseDown"}


def _get_direction(signal_name: str) -> str:
    """Map signal name to 'long' or 'short'."""
    if signal_name in _LONG_SIGNALS:
        return "long"
    if signal_name in _SHORT_SIGNALS:
        return "short"
    # Fallback: check prefix
    if signal_name.startswith("long") or signal_name.startswith("buy"):
        return "long"
    return "short"


# ---------------------------------------------------------------------------
# DataFrame construction from API bar dicts
# ---------------------------------------------------------------------------

def _bars_to_dataframe(bars: list[dict]) -> pd.DataFrame:
    """
    Convert list of bar dicts from the API into a pandas DataFrame
    with DatetimeIndex in US/Eastern timezone.

    Expected bar format:
        {"timestamp": "2025-01-15T10:30:00-05:00", "open": ..., "high": ...,
         "low": ..., "close": ..., "volume": ...}
    """
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "timestamp"

    # Ensure numeric types
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _resample_bars(df: pd.DataFrame, timeframe_str: str) -> pd.DataFrame:
    """
    Resample 1Min bars to a coarser timeframe (3Min, 5Min).
    Returns unchanged if timeframe is 1Min.
    """
    tf_minutes = {"1Min": 1, "3Min": 3, "5Min": 5}.get(timeframe_str, 1)
    if tf_minutes <= 1:
        return df

    rule = f"{tf_minutes}min"
    resampled = df.resample(rule, offset="30min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    return resampled


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(df: pd.DataFrame, cfg: DayXConfig) -> pd.DataFrame:
    """
    Run the full DayX pipeline on a prepared DataFrame:
    filter_rth -> add_session_markers -> compute_opening_range ->
    compute_all -> generate_signals.
    """
    df = filter_rth(df, cfg)
    if df.empty:
        return df
    df = add_session_markers(df, cfg)
    df = compute_opening_range(df, cfg)
    df = compute_all(df, cfg)
    df = generate_signals(df, cfg)
    return df


def _row_to_signal(row: pd.Series, cfg: DayXConfig) -> dict:
    """
    Convert a DataFrame row with signal_any == True into a signal dict.
    Computes stop/target from ATR.
    """
    signal_name = row["signal_name"]
    direction = _get_direction(signal_name)
    price = float(row["close"])
    atr_val = float(row["atr"]) if pd.notna(row["atr"]) else 0.0

    if direction == "long":
        stop = price - atr_val * cfg.stop_atr_mult
        target1 = price + atr_val * cfg.stop_atr_mult * cfg.target1_r
        target2 = price + atr_val * cfg.stop_atr_mult * cfg.target2_r
    else:
        stop = price + atr_val * cfg.stop_atr_mult
        target1 = price - atr_val * cfg.stop_atr_mult * cfg.target1_r
        target2 = price - atr_val * cfg.stop_atr_mult * cfg.target2_r

    # Convert index timestamp to Unix seconds (UTC)
    ts = row.name
    if hasattr(ts, "timestamp"):
        unix_ts = int(ts.timestamp())
    else:
        unix_ts = int(pd.Timestamp(ts).timestamp())

    return {
        "time": unix_ts,
        "signal": signal_name,
        "direction": direction,
        "price": round(price, 2),
        "stop": round(stop, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
    }


# ---------------------------------------------------------------------------
# Public API: historical signals
# ---------------------------------------------------------------------------

def _ts_to_unix(ts) -> int:
    """Convert pandas Timestamp to Unix seconds (UTC)."""
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp())
    return int(pd.Timestamp(ts).timestamp())


_EXIT_REASON_LABELS = {
    "stop_loss": "stop loss",
    "target1": "T1 partial",
    "target2": "T2 full",
    "eod_flatten": "EOD close",
    "backtest_end": "session end",
}


def compute_signals_historical(
    bars: list[dict],
    symbol: str,
    timeframe_str: str,
) -> list[dict]:
    """
    Run the full DayX simulation on historical bars and return
    both entry and exit markers.

    Args:
        bars: List of bar dicts from the API.
        symbol: Ticker symbol (e.g. "QQQ").
        timeframe_str: Timeframe string ("1Min", "3Min", "5Min").

    Returns:
        List of marker dicts (entries + exits).
    """
    if not bars:
        logger.warning("compute_signals_historical: no bars provided")
        return []

    cfg = TIMEFRAME_CONFIGS.get(timeframe_str)
    if cfg is None:
        logger.error(
            "No optimized config for timeframe %s. Available: %s",
            timeframe_str,
            list(TIMEFRAME_CONFIGS.keys()),
        )
        return []

    try:
        df = _bars_to_dataframe(bars)
        df = _resample_bars(df, timeframe_str)
        min_warmup = max(cfg.bb_period, cfg.cci_fast, cfg.atr_period)
        if len(df) < min_warmup:
            logger.warning(
                "Insufficient data for indicator warmup: have %d bars, need >= %d",
                len(df),
                min_warmup,
            )
            return []

        df = _run_pipeline(df, cfg)
        if df.empty:
            return []

        # Run full simulation to get trades with entries AND exits
        trades, _eq = run_simulation(df, cfg)

        markers: list[dict] = []
        for trade in trades:
            direction = trade.direction  # "long" or "short"

            # Entry marker
            markers.append({
                "time": _ts_to_unix(trade.entry_time),
                "type": "entry",
                "signal": trade.strategy,
                "direction": direction,
                "price": round(trade.entry_price, 2),
                "stop": round(trade.stop_price, 2),
                "target1": round(trade.target1_price, 2),
                "target2": round(trade.target2_price, 2),
            })

            # Exit marker (if closed)
            if trade.exit_time is not None:
                exit_label = _EXIT_REASON_LABELS.get(
                    trade.exit_reason, trade.exit_reason
                )
                markers.append({
                    "time": _ts_to_unix(trade.exit_time),
                    "type": "exit",
                    "signal": exit_label,
                    "direction": direction,
                    "price": round(trade.exit_price, 2),
                    "pnl": round(trade.pnl, 2),
                    "exitReason": trade.exit_reason,
                })

        logger.info(
            "compute_signals_historical: %s %s — %d bars, %d trades (%d markers)",
            symbol,
            timeframe_str,
            len(df),
            len(trades),
            len(markers),
        )
        return markers

    except Exception:
        logger.exception(
            "Error computing historical signals for %s %s",
            symbol,
            timeframe_str,
        )
        return []


# ---------------------------------------------------------------------------
# Public API: real-time bar processing
# ---------------------------------------------------------------------------

# Rolling session DataFrames keyed by (symbol, timeframe)
_session_buffers: dict[tuple[str, str], pd.DataFrame] = {}


def process_bar(
    bar: dict,
    symbol: str,
    timeframe_str: str,
) -> Optional[dict]:
    """
    Process a single incoming bar for real-time signal detection.

    Maintains a rolling session DataFrame per (symbol, timeframe).
    On each bar: append, recompute full pipeline, check last row for signal.
    Resets the session buffer when a new trading day is detected.

    Args:
        bar: Single bar dict from the API.
        symbol: Ticker symbol.
        timeframe_str: Timeframe string.

    Returns:
        Signal dict if a signal fires on this bar, otherwise None.
    """
    cfg = TIMEFRAME_CONFIGS.get(timeframe_str)
    if cfg is None:
        logger.error(
            "No optimized config for timeframe %s. Available: %s",
            timeframe_str,
            list(TIMEFRAME_CONFIGS.keys()),
        )
        return None

    key = (symbol, timeframe_str)

    try:
        # Convert single bar to a 1-row DataFrame
        new_row = _bars_to_dataframe([bar])
        if new_row.empty:
            return None

        new_bar_date = new_row.index[0].date()

        # Get or create session buffer
        existing = _session_buffers.get(key)
        if existing is not None and not existing.empty:
            prev_bar_date = existing.index[-1].date()
            if new_bar_date != prev_bar_date:
                # New trading day — reset session buffer
                logger.info(
                    "New session detected for %s %s: %s -> %s",
                    symbol,
                    timeframe_str,
                    prev_bar_date,
                    new_bar_date,
                )
                _session_buffers[key] = new_row
            else:
                # Same session — append
                _session_buffers[key] = pd.concat([existing, new_row])
        else:
            # First bar for this key
            _session_buffers[key] = new_row

        session_df = _session_buffers[key].copy()

        # Check warmup requirement
        min_warmup = max(cfg.bb_period, cfg.cci_fast, cfg.atr_period)
        if len(session_df) < min_warmup:
            return None

        # Run full pipeline
        processed = _run_pipeline(session_df, cfg)
        if processed.empty:
            return None

        # Check last row for signal
        last_row = processed.iloc[-1]
        if last_row.get("signal_any", False):
            signal = _row_to_signal(last_row, cfg)
            logger.info(
                "Signal detected for %s %s: %s %s @ %.2f",
                symbol,
                timeframe_str,
                signal["direction"],
                signal["signal"],
                signal["price"],
            )
            return signal

        return None

    except Exception:
        logger.exception(
            "Error processing bar for %s %s",
            symbol,
            timeframe_str,
        )
        return None


def reset_session(symbol: str, timeframe_str: str) -> None:
    """Manually reset the session buffer for a given symbol/timeframe pair."""
    key = (symbol, timeframe_str)
    if key in _session_buffers:
        del _session_buffers[key]
        logger.info("Session buffer reset for %s %s", symbol, timeframe_str)


def reset_all_sessions() -> None:
    """Reset all session buffers."""
    _session_buffers.clear()
    logger.info("All session buffers reset")
