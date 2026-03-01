"""
Database Query Functions for Market Data

Provides efficient queries for fetching trades and quotes from TimescaleDB
for feature computation (OFI, VPIN, etc.)
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy import text

from .database import get_engine, get_session

logger = logging.getLogger(__name__)


def get_trades(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch trades for a symbol within a time range.

    Args:
        symbol: Trading symbol (e.g., 'QQQ')
        start_time: Start of time range (inclusive)
        end_time: End of time range (exclusive)
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: time, price, size, exchange, conditions
    """
    engine = get_engine()

    query = """
        SELECT time, price, size, exchange, conditions
        FROM trades
        WHERE symbol = :symbol
          AND time >= :start_time
          AND time < :end_time
        ORDER BY time ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(
        text(query),
        engine,
        params={
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time
        }
    )

    logger.info(f"Fetched {len(df):,} trades for {symbol} from {start_time} to {end_time}")
    return df


def get_quotes(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch quotes for a symbol within a time range.

    Args:
        symbol: Trading symbol (e.g., 'QQQ')
        start_time: Start of time range (inclusive)
        end_time: End of time range (exclusive)
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: time, bid_price, bid_size, ask_price, ask_size
    """
    engine = get_engine()

    query = """
        SELECT time, bid_price, bid_size, ask_price, ask_size
        FROM quotes
        WHERE symbol = :symbol
          AND time >= :start_time
          AND time < :end_time
        ORDER BY time ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(
        text(query),
        engine,
        params={
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time
        }
    )

    logger.info(f"Fetched {len(df):,} quotes for {symbol} from {start_time} to {end_time}")
    return df


def get_trades_for_date(symbol: str, trade_date: date) -> pd.DataFrame:
    """
    Fetch all trades for a specific date.

    Args:
        symbol: Trading symbol
        trade_date: Date to fetch

    Returns:
        DataFrame with trade data
    """
    start_time = datetime.combine(trade_date, datetime.min.time())
    end_time = start_time + timedelta(days=1)
    return get_trades(symbol, start_time, end_time)


def get_quotes_for_date(symbol: str, trade_date: date) -> pd.DataFrame:
    """
    Fetch all quotes for a specific date.

    Args:
        symbol: Trading symbol
        trade_date: Date to fetch

    Returns:
        DataFrame with quote data
    """
    start_time = datetime.combine(trade_date, datetime.min.time())
    end_time = start_time + timedelta(days=1)
    return get_quotes(symbol, start_time, end_time)


def get_trades_with_prevailing_quotes(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch trades with the prevailing bid/ask at each trade time.

    Uses ASOF join to get the most recent quote before each trade.
    This is essential for proper trade classification (buy vs sell).

    Args:
        symbol: Trading symbol
        start_time: Start of time range
        end_time: End of time range
        limit: Optional limit

    Returns:
        DataFrame with columns: time, price, size, bid_price, bid_size, ask_price, ask_size
    """
    engine = get_engine()

    # TimescaleDB/PostgreSQL LATERAL join for ASOF semantics
    query = """
        SELECT
            t.time,
            t.price,
            t.size,
            t.exchange,
            q.bid_price,
            q.bid_size,
            q.ask_price,
            q.ask_size
        FROM trades t
        LEFT JOIN LATERAL (
            SELECT bid_price, bid_size, ask_price, ask_size
            FROM quotes
            WHERE symbol = :symbol
              AND time <= t.time
            ORDER BY time DESC
            LIMIT 1
        ) q ON true
        WHERE t.symbol = :symbol
          AND t.time >= :start_time
          AND t.time < :end_time
        ORDER BY t.time ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(
        text(query),
        engine,
        params={
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time
        }
    )

    logger.info(f"Fetched {len(df):,} trades with quotes for {symbol}")
    return df


def get_quote_changes(
    symbol: str,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame:
    """
    Fetch quotes with changes from previous quote.

    This is optimized for OFI calculation - includes:
    - Previous bid/ask prices and sizes
    - Whether bid/ask moved up, down, or stayed same

    Args:
        symbol: Trading symbol
        start_time: Start of time range
        end_time: End of time range

    Returns:
        DataFrame with quote changes
    """
    engine = get_engine()

    query = """
        SELECT
            time,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            LAG(bid_price) OVER (ORDER BY time) as prev_bid_price,
            LAG(bid_size) OVER (ORDER BY time) as prev_bid_size,
            LAG(ask_price) OVER (ORDER BY time) as prev_ask_price,
            LAG(ask_size) OVER (ORDER BY time) as prev_ask_size
        FROM quotes
        WHERE symbol = :symbol
          AND time >= :start_time
          AND time < :end_time
        ORDER BY time ASC
    """

    df = pd.read_sql(
        text(query),
        engine,
        params={
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time
        }
    )

    logger.info(f"Fetched {len(df):,} quote changes for {symbol}")
    return df


def get_available_dates(symbol: str) -> list:
    """
    Get list of dates that have been ingested for a symbol.

    Returns:
        List of date objects
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT DISTINCT date
                FROM ingestion_log
                WHERE symbol = :symbol
                  AND data_type = 'trades'
                ORDER BY date ASC
            """),
            {'symbol': symbol}
        )
        return [row[0] for row in result]


def get_date_stats(symbol: str, trade_date: date) -> dict:
    """
    Get statistics for a specific date.

    Returns:
        Dict with trade_count, quote_count, first_trade, last_trade
    """
    start_time = datetime.combine(trade_date, datetime.min.time())
    end_time = start_time + timedelta(days=1)

    with get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    (SELECT COUNT(*) FROM trades WHERE symbol = :symbol AND time >= :start AND time < :end) as trade_count,
                    (SELECT COUNT(*) FROM quotes WHERE symbol = :symbol AND time >= :start AND time < :end) as quote_count,
                    (SELECT MIN(time) FROM trades WHERE symbol = :symbol AND time >= :start AND time < :end) as first_trade,
                    (SELECT MAX(time) FROM trades WHERE symbol = :symbol AND time >= :start AND time < :end) as last_trade
            """),
            {'symbol': symbol, 'start': start_time, 'end': end_time}
        )
        row = result.fetchone()
        return {
            'trade_count': row[0] or 0,
            'quote_count': row[1] or 0,
            'first_trade': row[2],
            'last_trade': row[3]
        }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from datetime import date

    # Test fetching data
    symbol = 'QQQ'
    test_date = date(2025, 12, 23)

    print(f"\nTesting queries for {symbol} on {test_date}...")

    # Get date stats
    stats = get_date_stats(symbol, test_date)
    print(f"  Trades: {stats['trade_count']:,}")
    print(f"  Quotes: {stats['quote_count']:,}")
    print(f"  First trade: {stats['first_trade']}")
    print(f"  Last trade: {stats['last_trade']}")

    # Get sample trades
    print("\nFetching sample trades...")
    start = datetime.combine(test_date, datetime.min.time())
    end = start + timedelta(hours=1)
    trades = get_trades(symbol, start, end, limit=10)
    print(trades.head())

    # Get sample quotes
    print("\nFetching sample quotes...")
    quotes = get_quotes(symbol, start, end, limit=10)
    print(quotes.head())

    # Get trades with quotes
    print("\nFetching trades with prevailing quotes...")
    trades_with_quotes = get_trades_with_prevailing_quotes(symbol, start, end, limit=10)
    print(trades_with_quotes.head())
