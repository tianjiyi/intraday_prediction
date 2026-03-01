"""
Migrate Parquet Cache Files to TimescaleDB

This script reads existing parquet files from the cache directory
and imports them into the TimescaleDB database using COPY for bulk loading.

Usage:
    python -m market_regime.scripts.migrate_parquet [--symbol QQQ] [--force]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from io import StringIO

import pandas as pd
import numpy as np
from sqlalchemy import text

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_regime.database import (
    get_engine,
    get_session,
    test_connection,
    check_timescaledb,
    is_date_ingested,
    log_ingestion
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(__file__).parent.parent / "data"


def get_parquet_files(symbol: str, data_type: str) -> list:
    """
    Find all parquet files for a symbol and data type.

    Returns list of (file_path, date_str) tuples.
    """
    symbol_dir = CACHE_DIR / symbol
    if not symbol_dir.exists():
        logger.warning(f"No cache directory found for {symbol}")
        return []

    pattern = f"{data_type}_*.parquet"
    files = []

    for file_path in symbol_dir.glob(pattern):
        # Extract date from filename: trades_2024-12-20.parquet -> 2024-12-20
        filename = file_path.stem
        date_str = filename.replace(f"{data_type}_", "")
        files.append((file_path, date_str))

    # Sort by date
    files.sort(key=lambda x: x[1])
    return files


def copy_dataframe_to_table(df: pd.DataFrame, table_name: str, columns: list):
    """
    Use PostgreSQL COPY to bulk load a DataFrame into a table.
    This is 10-100x faster than row-by-row inserts.
    """
    engine = get_engine()

    # Create CSV in memory
    buffer = StringIO()
    df[columns].to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
    buffer.seek(0)

    # Use raw connection for COPY
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.copy_from(
            buffer,
            table_name,
            sep='\t',
            null='\\N',
            columns=columns
        )
        raw_conn.commit()
    finally:
        cursor.close()
        raw_conn.close()


def migrate_trades(file_path: Path, symbol: str, date_str: str, force: bool = False) -> int:
    """
    Migrate a single trades parquet file to the database using COPY.

    Returns number of records inserted.
    """
    # Check if already ingested
    if not force and is_date_ingested(symbol, date_str, 'trades'):
        logger.info(f"Skipping {symbol} trades {date_str} (already ingested)")
        return 0

    # Read parquet file
    logger.info(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)

    if df.empty:
        logger.warning(f"Empty parquet file: {file_path}")
        return 0

    # Ensure timestamp column is timezone-aware
    if 'timestamp' not in df.columns:
        logger.error(f"Missing 'timestamp' column in {file_path}")
        return 0

    record_count = len(df)
    logger.info(f"Processing {record_count:,} trades for {symbol} {date_str}...")

    # Prepare DataFrame for COPY
    copy_df = pd.DataFrame()
    copy_df['time'] = df['timestamp']
    copy_df['symbol'] = symbol
    copy_df['price'] = df['price'].astype(float)
    copy_df['size'] = df['size'].astype(int)
    copy_df['exchange'] = df.get('exchange')

    # Handle conditions column - convert to PostgreSQL array format
    if 'conditions' in df.columns:
        def format_conditions(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if isinstance(x, list):
                # Format as PostgreSQL array: {a,b,c}
                return '{' + ','.join(str(v) for v in x) + '}'
            return None
        copy_df['conditions'] = df['conditions'].apply(format_conditions)
    else:
        copy_df['conditions'] = None

    # Delete existing records for this date (if force)
    if force:
        with get_session() as session:
            session.execute(
                text("""
                    DELETE FROM trades
                    WHERE symbol = :symbol
                      AND time >= CAST(:date_str AS date)
                      AND time < CAST(:date_str AS date) + INTERVAL '1 day'
                """),
                {'symbol': symbol, 'date_str': date_str}
            )

    # Bulk load using COPY
    logger.info(f"Bulk loading {record_count:,} trades...")
    copy_dataframe_to_table(
        copy_df,
        'trades',
        ['time', 'symbol', 'price', 'size', 'exchange', 'conditions']
    )

    # Log successful ingestion
    log_ingestion(symbol, date_str, 'trades', record_count, 'parquet_migration')
    logger.info(f"Done: Migrated {record_count:,} trades for {symbol} {date_str}")

    return record_count


def migrate_quotes(file_path: Path, symbol: str, date_str: str, force: bool = False) -> int:
    """
    Migrate a single quotes parquet file to the database using COPY.

    Returns number of records inserted.
    """
    # Check if already ingested
    if not force and is_date_ingested(symbol, date_str, 'quotes'):
        logger.info(f"Skipping {symbol} quotes {date_str} (already ingested)")
        return 0

    # Read parquet file
    logger.info(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)

    if df.empty:
        logger.warning(f"Empty parquet file: {file_path}")
        return 0

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        logger.error(f"Missing 'timestamp' column in {file_path}")
        return 0

    # Filter out rows with null timestamps
    df = df.dropna(subset=['timestamp'])

    record_count = len(df)
    logger.info(f"Processing {record_count:,} quotes for {symbol} {date_str}...")

    # Prepare DataFrame for COPY
    copy_df = pd.DataFrame()
    copy_df['time'] = df['timestamp']
    copy_df['symbol'] = symbol
    copy_df['bid_price'] = df['bid_price'].astype(float) if 'bid_price' in df.columns else None
    copy_df['bid_size'] = df['bid_size'].astype('Int64') if 'bid_size' in df.columns else None
    copy_df['ask_price'] = df['ask_price'].astype(float) if 'ask_price' in df.columns else None
    copy_df['ask_size'] = df['ask_size'].astype('Int64') if 'ask_size' in df.columns else None

    # Delete existing records for this date (if force)
    if force:
        with get_session() as session:
            session.execute(
                text("""
                    DELETE FROM quotes
                    WHERE symbol = :symbol
                      AND time >= CAST(:date_str AS date)
                      AND time < CAST(:date_str AS date) + INTERVAL '1 day'
                """),
                {'symbol': symbol, 'date_str': date_str}
            )

    # Bulk load using COPY
    logger.info(f"Bulk loading {record_count:,} quotes...")
    copy_dataframe_to_table(
        copy_df,
        'quotes',
        ['time', 'symbol', 'bid_price', 'bid_size', 'ask_price', 'ask_size']
    )

    # Log successful ingestion
    log_ingestion(symbol, date_str, 'quotes', record_count, 'parquet_migration')
    logger.info(f"Done: Migrated {record_count:,} quotes for {symbol} {date_str}")

    return record_count


def migrate_symbol(symbol: str, force: bool = False, start_date: str = None, end_date: str = None) -> dict:
    """
    Migrate all parquet files for a symbol.

    Args:
        symbol: Symbol to migrate
        force: Force re-migration of existing data
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns dict with migration statistics.
    """
    stats = {
        'trades_files': 0,
        'trades_records': 0,
        'quotes_files': 0,
        'quotes_records': 0
    }

    def is_in_date_range(date_str: str) -> bool:
        """Check if date_str is within the specified date range."""
        if start_date is None and end_date is None:
            return True
        if start_date and date_str < start_date:
            return False
        if end_date and date_str > end_date:
            return False
        return True

    # Migrate trades
    trades_files = get_parquet_files(symbol, 'trades')
    trades_files = [(f, d) for f, d in trades_files if is_in_date_range(d)]
    logger.info(f"Found {len(trades_files)} trades parquet files for {symbol}" +
                (f" (filtered: {start_date} to {end_date})" if start_date or end_date else ""))

    for file_path, date_str in trades_files:
        try:
            count = migrate_trades(file_path, symbol, date_str, force)
            if count > 0:
                stats['trades_files'] += 1
                stats['trades_records'] += count
        except Exception as e:
            logger.error(f"Error migrating trades {file_path}: {e}")

    # Migrate quotes
    quotes_files = get_parquet_files(symbol, 'quotes')
    quotes_files = [(f, d) for f, d in quotes_files if is_in_date_range(d)]
    logger.info(f"Found {len(quotes_files)} quotes parquet files for {symbol}" +
                (f" (filtered: {start_date} to {end_date})" if start_date or end_date else ""))

    for file_path, date_str in quotes_files:
        try:
            count = migrate_quotes(file_path, symbol, date_str, force)
            if count > 0:
                stats['quotes_files'] += 1
                stats['quotes_records'] += count
        except Exception as e:
            logger.error(f"Error migrating quotes {file_path}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Migrate parquet cache to TimescaleDB')
    parser.add_argument('--symbol', type=str, help='Symbol to migrate (default: all)')
    parser.add_argument('--force', action='store_true', help='Force re-migration of existing data')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Parquet to TimescaleDB Migration (COPY Mode)")
    logger.info("=" * 60)

    # Test database connection
    logger.info("Testing database connection...")
    if not test_connection():
        logger.error("Cannot connect to database. Is TimescaleDB running?")
        logger.error("Start with: docker-compose up -d")
        sys.exit(1)

    if not check_timescaledb():
        logger.error("TimescaleDB extension not installed!")
        sys.exit(1)

    logger.info("Database connection successful")
    logger.info("")

    # Find symbols to migrate
    if args.symbol:
        symbols = [args.symbol]
    else:
        # Find all symbol directories
        symbols = [d.name for d in CACHE_DIR.iterdir() if d.is_dir()]

    if not symbols:
        logger.warning("No symbol directories found in cache")
        sys.exit(0)

    logger.info(f"Migrating symbols: {', '.join(symbols)}")
    logger.info("")

    # Migrate each symbol
    total_stats = {
        'trades_files': 0,
        'trades_records': 0,
        'quotes_files': 0,
        'quotes_records': 0
    }

    # Log date range filter if specified
    if args.start_date or args.end_date:
        logger.info(f"Date range filter: {args.start_date or 'any'} to {args.end_date or 'any'}")
        logger.info("")

    for symbol in symbols:
        logger.info(f"--- Migrating {symbol} ---")
        stats = migrate_symbol(symbol, args.force, args.start_date, args.end_date)

        for key in total_stats:
            total_stats[key] += stats[key]

        logger.info(f"  Trades: {stats['trades_files']} files, {stats['trades_records']:,} records")
        logger.info(f"  Quotes: {stats['quotes_files']} files, {stats['quotes_records']:,} records")
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Migration Complete")
    logger.info("=" * 60)
    logger.info(f"Total trades: {total_stats['trades_files']} files, {total_stats['trades_records']:,} records")
    logger.info(f"Total quotes: {total_stats['quotes_files']} files, {total_stats['quotes_records']:,} records")


if __name__ == "__main__":
    main()
