"""
Database Connection and Utilities for TimescaleDB

Provides connection management, session handling, and utility functions
for the market data database.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator
from datetime import datetime, date

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Database configuration (can be overridden via environment variables)
DB_CONFIG = {
    'host': os.getenv('TIMESCALE_HOST', 'localhost'),
    'port': os.getenv('TIMESCALE_PORT', '5432'),
    'database': os.getenv('TIMESCALE_DB', 'market_data'),
    'user': os.getenv('TIMESCALE_USER', 'kronos'),
    'password': os.getenv('TIMESCALE_PASSWORD', 'kronos_dev'),
}


def get_connection_string() -> str:
    """Build PostgreSQL connection string from config."""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


# Global engine instance (lazy initialization)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        connection_string = get_connection_string()
        _engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            echo=False,  # Set to True for SQL debugging
        )
        logger.info(f"Database engine created: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_session() as session:
            session.execute(text("SELECT * FROM trades"))
            session.commit()
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """Test if database connection is working."""
    try:
        with get_session() as session:
            result = session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def check_timescaledb() -> bool:
    """Check if TimescaleDB extension is installed."""
    try:
        with get_session() as session:
            result = session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            )
            version = result.scalar()
            if version:
                logger.info(f"TimescaleDB version: {version}")
                return True
            return False
    except Exception as e:
        logger.error(f"TimescaleDB check failed: {e}")
        return False


def get_data_coverage(symbol: str) -> dict:
    """
    Get data coverage information for a symbol.

    Returns dict with first_date, last_date, days_covered for trades and quotes.
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT data_type, first_date, last_date, days_covered, total_records
                FROM data_coverage
                WHERE symbol = :symbol
            """),
            {'symbol': symbol}
        )
        coverage = {}
        for row in result:
            coverage[row[0]] = {
                'first_date': row[1],
                'last_date': row[2],
                'days_covered': row[3],
                'total_records': row[4]
            }
        return coverage


def is_date_ingested(symbol: str, date_str: str, data_type: str) -> bool:
    """Check if a specific date has been ingested for a symbol."""
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT 1 FROM ingestion_log
                WHERE symbol = :symbol
                  AND date = :date
                  AND data_type = :data_type
            """),
            {'symbol': symbol, 'date': date_str, 'data_type': data_type}
        )
        return result.scalar() is not None


def log_ingestion(
    symbol: str,
    date_str: str,
    data_type: str,
    record_count: int,
    source: str = 'api'
) -> None:
    """Log a successful data ingestion."""
    with get_session() as session:
        session.execute(
            text("""
                INSERT INTO ingestion_log (symbol, date, data_type, record_count, source)
                VALUES (:symbol, :date, :data_type, :record_count, :source)
                ON CONFLICT (symbol, date, data_type)
                DO UPDATE SET
                    record_count = EXCLUDED.record_count,
                    ingested_at = NOW(),
                    source = EXCLUDED.source
            """),
            {
                'symbol': symbol,
                'date': date_str,
                'data_type': data_type,
                'record_count': record_count,
                'source': source
            }
        )


def get_trade_count(symbol: str, start_date: str, end_date: str) -> int:
    """Get total trade count for a symbol and date range."""
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT COUNT(*) FROM trades
                WHERE symbol = :symbol
                  AND time >= :start_date
                  AND time < :end_date::date + INTERVAL '1 day'
            """),
            {'symbol': symbol, 'start_date': start_date, 'end_date': end_date}
        )
        return result.scalar() or 0


def get_quote_count(symbol: str, start_date: str, end_date: str) -> int:
    """Get total quote count for a symbol and date range."""
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT COUNT(*) FROM quotes
                WHERE symbol = :symbol
                  AND time >= :start_date
                  AND time < :end_date::date + INTERVAL '1 day'
            """),
            {'symbol': symbol, 'start_date': start_date, 'end_date': end_date}
        )
        return result.scalar() or 0


# Quick test when module is run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing database connection...")
    if test_connection():
        print("✓ Connection successful")

        if check_timescaledb():
            print("✓ TimescaleDB extension installed")
        else:
            print("✗ TimescaleDB extension not found")

        # Show data coverage if any
        coverage = get_data_coverage('QQQ')
        if coverage:
            print(f"\nData coverage for QQQ:")
            for data_type, info in coverage.items():
                print(f"  {data_type}: {info['first_date']} to {info['last_date']} ({info['days_covered']} days, {info['total_records']} records)")
        else:
            print("\nNo data ingested yet for QQQ")
    else:
        print("✗ Connection failed. Is TimescaleDB running?")
        print("  Start with: docker-compose up -d")
