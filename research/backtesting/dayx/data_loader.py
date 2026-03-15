"""
Fetch OHLCV data from Alpaca and prepare for Day_X backtest.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .config import DayXConfig

# Load .env from project root
_root = Path(__file__).resolve().parents[3]
_env_path = _root / ".env"
load_dotenv(_env_path)

_cache_dir = _root / "data" / "backtests" / "dayx" / "cache"


def _parse_timeframe(tf_str: str) -> TimeFrame:
    """Convert '1Min', '5Min', etc. to Alpaca TimeFrame."""
    minutes = int(tf_str.replace("Min", ""))
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def fetch_bars(cfg: DayXConfig) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca for the configured symbol/date range.
    Chunks by year to avoid API timeouts on large ranges.

    Returns DataFrame indexed by datetime (ET) with columns:
        open, high, low, close, volume, vwap, trade_count
    """
    import logging
    logger = logging.getLogger(__name__)

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_KEY_ID"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )

    start = datetime.fromisoformat(cfg.start_date)
    end = datetime.fromisoformat(cfg.end_date)
    tf = _parse_timeframe(cfg.timeframe)

    # Chunk by year to avoid API timeouts
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(datetime(chunk_start.year + 1, 1, 1), end)
        logger.info(f"  Fetching {cfg.symbol} {chunk_start.date()} to {chunk_end.date()}...")

        request = StockBarsRequest(
            symbol_or_symbols=cfg.symbol,
            timeframe=tf,
            start=chunk_start,
            end=chunk_end,
        )
        bars = client.get_stock_bars(request)
        chunk_df = bars.df
        if len(chunk_df) > 0:
            chunks.append(chunk_df)

        chunk_start = chunk_end

    df = pd.concat(chunks)

    # If multi-index (symbol, timestamp), drop symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")

    # Deduplicate in case of overlap
    df = df[~df.index.duplicated(keep="first")]

    # Ensure timezone-aware in US/Eastern
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "timestamp"

    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    return df


def _cache_path(cfg: DayXConfig) -> Path:
    return _cache_dir / f"{cfg.symbol}_{cfg.timeframe}_{cfg.start_date}_{cfg.end_date}.parquet"


def fetch_bars_cached(cfg: DayXConfig) -> pd.DataFrame:
    """Fetch bars with parquet caching. Caches raw data before RTH filtering."""
    import logging
    logger = logging.getLogger(__name__)

    cache_file = _cache_path(cfg)
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file.name}")
        return pd.read_parquet(cache_file)

    df = fetch_bars(cfg)
    _cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.info(f"Cached {len(df)} bars to {cache_file.name}")
    return df


def filter_rth(df: pd.DataFrame, cfg: DayXConfig) -> pd.DataFrame:
    """Keep only Regular Trading Hours bars."""
    start_h, start_m = map(int, cfg.rth_start.split(":"))
    end_h, end_m = map(int, cfg.rth_end.split(":"))

    mask = (
        (df.index.hour > start_h) | ((df.index.hour == start_h) & (df.index.minute >= start_m))
    ) & (
        (df.index.hour < end_h) | ((df.index.hour == end_h) & (df.index.minute == 0))
    )
    return df[mask].copy()


def add_session_markers(df: pd.DataFrame, cfg: DayXConfig) -> pd.DataFrame:
    """Add session_date and session_open flag."""
    df["session_date"] = df.index.date

    start_h, start_m = map(int, cfg.rth_start.split(":"))

    # First bar of each session
    df["session_open"] = False
    for date in df["session_date"].unique():
        day_mask = df["session_date"] == date
        day_idx = df[day_mask].index
        if len(day_idx) > 0:
            df.loc[day_idx[0], "session_open"] = True

    return df


def compute_opening_range(df: pd.DataFrame, cfg: DayXConfig) -> pd.DataFrame:
    """
    Compute opening range (high/low of first N bars per session).
    Adds columns: or_high, or_low
    """
    df["or_high"] = np.nan
    df["or_low"] = np.nan

    for date in df["session_date"].unique():
        day_mask = df["session_date"] == date
        day_df = df[day_mask]
        if len(day_df) < cfg.opening_range_bars:
            continue

        or_bars = day_df.iloc[:cfg.opening_range_bars]
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()

        df.loc[day_mask, "or_high"] = or_high
        df.loc[day_mask, "or_low"] = or_low

    return df


def load_data(cfg: DayXConfig) -> pd.DataFrame:
    """Full pipeline: fetch → filter RTH → add markers → opening range."""
    df = fetch_bars_cached(cfg)
    df = filter_rth(df, cfg)
    df = add_session_markers(df, cfg)
    df = compute_opening_range(df, cfg)
    return df
