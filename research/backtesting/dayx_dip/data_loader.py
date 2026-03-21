"""
Fetch OHLCV data from Alpaca and prepare for DayX Dip backtest.
Uses the dayx_dip cache directory.
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

from .config import DayXDipConfig

# Load .env from project root
_root = Path(__file__).resolve().parents[3]
_env_path = _root / ".env"
load_dotenv(_env_path)

_cache_dir = _root / "data" / "backtests" / "dayx_dip" / "cache"


def _parse_timeframe(tf_str: str) -> TimeFrame:
    """Convert '1Min', '3Min', etc. to Alpaca TimeFrame."""
    minutes = int(tf_str.replace("Min", ""))
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def fetch_bars(cfg: DayXDipConfig) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca. Chunks by year to avoid API timeouts."""
    import logging
    logger = logging.getLogger(__name__)

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_KEY_ID"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )

    start = datetime.fromisoformat(cfg.start_date)
    end = datetime.fromisoformat(cfg.end_date)
    tf = _parse_timeframe(cfg.timeframe)

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

    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")

    df = df[~df.index.duplicated(keep="first")]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "timestamp"
    df.columns = [c.lower() for c in df.columns]

    return df


def _cache_path(cfg: DayXDipConfig) -> Path:
    return _cache_dir / f"{cfg.symbol}_{cfg.timeframe}_{cfg.start_date}_{cfg.end_date}.parquet"


def fetch_bars_cached(cfg: DayXDipConfig) -> pd.DataFrame:
    """Fetch bars with parquet caching."""
    import logging
    logger = logging.getLogger(__name__)

    # Also check the dayx cache to reuse already-downloaded data
    dayx_cache = _root / "data" / "backtests" / "dayx" / "cache"
    dayx_file = dayx_cache / f"{cfg.symbol}_{cfg.timeframe}_{cfg.start_date}_{cfg.end_date}.parquet"

    cache_file = _cache_path(cfg)
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file.name}")
        return pd.read_parquet(cache_file)
    if dayx_file.exists():
        logger.info(f"Reusing dayx cache: {dayx_file.name}")
        return pd.read_parquet(dayx_file)

    df = fetch_bars(cfg)
    _cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.info(f"Cached {len(df)} bars to {cache_file.name}")
    return df


def filter_rth(df: pd.DataFrame, cfg: DayXDipConfig) -> pd.DataFrame:
    """Keep only Regular Trading Hours bars."""
    start_h, start_m = map(int, cfg.rth_start.split(":"))
    end_h, end_m = map(int, cfg.rth_end.split(":"))

    mask = (
        (df.index.hour > start_h) | ((df.index.hour == start_h) & (df.index.minute >= start_m))
    ) & (
        (df.index.hour < end_h) | ((df.index.hour == end_h) & (df.index.minute == 0))
    )
    return df[mask].copy()


def add_session_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Add session_date column."""
    df["session_date"] = df.index.date
    return df


def load_data(cfg: DayXDipConfig) -> pd.DataFrame:
    """Full pipeline: fetch → filter RTH → add session markers."""
    df = fetch_bars_cached(cfg)
    df = filter_rth(df, cfg)
    df = add_session_markers(df)
    return df
