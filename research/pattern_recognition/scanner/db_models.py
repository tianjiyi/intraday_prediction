"""
SQLAlchemy ORM models for the Scanner module.

Tables:
- DefaultWatchlist: Static list of symbols to always scan
- FloatingWatchlist: Dynamic premarket movers (daily)
- PatternFinding: Detected patterns with breakout status
- PremarketScanLog: Daily premarket scan execution log
- RealtimeScanLog: Real-time scan cycle log
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    Column, Integer, String, Boolean, Text, Date, DateTime,
    DECIMAL, BigInteger, JSON, Index, UniqueConstraint,
    create_engine, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class DefaultWatchlist(Base):
    """
    Static watchlist of symbols to always scan.

    Categories:
    - index_etf: QQQ, SPY, IWM
    - tech_mega: NVDA, AAPL, MSFT, META, GOOGL, AMZN
    - momentum: TSLA, AMD
    """
    __tablename__ = 'default_watchlist'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True)
    category = Column(String(50))
    enabled = Column(Boolean, default=True)
    notes = Column(Text)
    added_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DefaultWatchlist(symbol='{self.symbol}', enabled={self.enabled})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'category': self.category,
            'enabled': self.enabled,
            'notes': self.notes,
            'added_at': str(self.added_at) if self.added_at else None,
        }


class FloatingWatchlist(Base):
    """
    Dynamic watchlist from premarket scanner.

    Populated daily with stocks meeting criteria:
    - Market cap >= $1B
    - Gap >= 5%
    - Unusual volume (>= 2x average)
    """
    __tablename__ = 'floating_watchlist'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    symbol = Column(String(10), nullable=False)
    gap_percent = Column(DECIMAL(8, 4))
    premarket_volume = Column(BigInteger)
    avg_volume = Column(BigInteger)
    volume_ratio = Column(DECIMAL(8, 4))
    market_cap = Column(DECIMAL(16, 2))
    prev_close = Column(DECIMAL(12, 4))
    premarket_high = Column(DECIMAL(12, 4))
    premarket_low = Column(DECIMAL(12, 4))
    premarket_open = Column(DECIMAL(12, 4))
    reason = Column(Text)
    scanned_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('date', 'symbol', name='uq_floating_watchlist_date_symbol'),
        Index('idx_floating_watchlist_date', 'date'),
        Index('idx_floating_watchlist_symbol', 'symbol', 'date'),
    )

    def __repr__(self):
        return f"<FloatingWatchlist(date={self.date}, symbol='{self.symbol}', gap={self.gap_percent}%)>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': str(self.date),
            'symbol': self.symbol,
            'gap_percent': float(self.gap_percent) if self.gap_percent else None,
            'premarket_volume': self.premarket_volume,
            'avg_volume': self.avg_volume,
            'volume_ratio': float(self.volume_ratio) if self.volume_ratio else None,
            'market_cap': float(self.market_cap) if self.market_cap else None,
            'prev_close': float(self.prev_close) if self.prev_close else None,
            'premarket_high': float(self.premarket_high) if self.premarket_high else None,
            'reason': self.reason,
            'scanned_at': str(self.scanned_at) if self.scanned_at else None,
        }


class PatternFinding(Base):
    """
    Detected ascending triangle patterns.

    Stored as time-series data in TimescaleDB hypertable.
    Includes breakout tracking (pending/success/failure/expired).
    """
    __tablename__ = 'pattern_findings'

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(10), primary_key=True)
    timeframe = Column(String(10), primary_key=True)
    pattern_type = Column(String(50), nullable=False, default='ascending_triangle')
    resistance_level = Column(DECIMAL(12, 4))
    support_slope = Column(DECIMAL(12, 8))
    support_intercept = Column(DECIMAL(12, 4))
    confidence = Column(DECIMAL(5, 4))
    pattern_start_idx = Column(Integer)
    pattern_end_idx = Column(Integer)
    pattern_start_time = Column(DateTime(timezone=True))
    pattern_end_time = Column(DateTime(timezone=True))
    breakout_status = Column(String(20), default='pending')
    breakout_price = Column(DECIMAL(12, 4))
    breakout_time = Column(DateTime(timezone=True))
    bars_to_breakout = Column(Integer)
    compression_ratio = Column(DECIMAL(5, 4))
    pattern_height = Column(DECIMAL(12, 4))
    peaks_count = Column(Integer)
    valleys_count = Column(Integer)
    pattern_data = Column(JSONB)

    __table_args__ = (
        Index('idx_pattern_findings_symbol_time', 'symbol', text('time DESC')),
        Index('idx_pattern_findings_type', 'pattern_type', text('time DESC')),
        Index('idx_pattern_findings_status', 'breakout_status', text('time DESC')),
    )

    def __repr__(self):
        return (f"<PatternFinding(symbol='{self.symbol}', time={self.time}, "
                f"confidence={self.confidence}, status='{self.breakout_status}')>")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': str(self.time),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'pattern_type': self.pattern_type,
            'resistance_level': float(self.resistance_level) if self.resistance_level else None,
            'support_slope': float(self.support_slope) if self.support_slope else None,
            'confidence': float(self.confidence) if self.confidence else None,
            'breakout_status': self.breakout_status,
            'breakout_price': float(self.breakout_price) if self.breakout_price else None,
            'bars_to_breakout': self.bars_to_breakout,
            'peaks_count': self.peaks_count,
            'valleys_count': self.valleys_count,
        }

    @classmethod
    def from_pattern(cls, pattern, scan_time: datetime = None):
        """
        Create PatternFinding from AscendingTrianglePattern.

        Args:
            pattern: AscendingTrianglePattern instance
            scan_time: Override detection time (default: now)

        Returns:
            PatternFinding instance
        """
        return cls(
            time=scan_time or datetime.utcnow(),
            symbol=pattern.ticker,
            timeframe=pattern.timeframe,
            pattern_type='ascending_triangle',
            resistance_level=Decimal(str(pattern.resistance_level)),
            support_slope=Decimal(str(pattern.support_slope)),
            support_intercept=Decimal(str(pattern.support_intercept)),
            confidence=Decimal(str(pattern.confidence)),
            pattern_start_idx=pattern.start_index,
            pattern_end_idx=pattern.end_index,
            pattern_start_time=pattern.start_timestamp,
            pattern_end_time=pattern.end_timestamp,
            breakout_status=pattern.breakout_status or 'pending',
            breakout_price=Decimal(str(pattern.breakout_price)) if pattern.breakout_price else None,
            breakout_time=pattern.breakout_timestamp,
            bars_to_breakout=pattern.bars_to_breakout,
            compression_ratio=Decimal(str(pattern.compression_ratio)) if pattern.compression_ratio else None,
            pattern_height=Decimal(str(pattern.pattern_height)) if pattern.pattern_height else None,
            peaks_count=len(pattern.peaks) if pattern.peaks else None,
            valleys_count=len(pattern.valleys) if pattern.valleys else None,
            pattern_data=pattern.to_dict() if hasattr(pattern, 'to_dict') else None,
        )


class PremarketScanLog(Base):
    """
    Log of daily premarket scan executions.
    """
    __tablename__ = 'premarket_scan_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    scan_start = Column(DateTime(timezone=True))
    scan_end = Column(DateTime(timezone=True))
    symbols_scanned = Column(Integer, default=0)
    symbols_passed = Column(Integer, default=0)
    status = Column(String(20))  # 'completed', 'failed', 'partial', 'running'
    error_msg = Column(Text)
    config_snapshot = Column(JSONB)

    def __repr__(self):
        return f"<PremarketScanLog(date={self.date}, status='{self.status}', passed={self.symbols_passed})>"


class RealtimeScanLog(Base):
    """
    Log of real-time scan cycles during RTH.
    """
    __tablename__ = 'realtime_scan_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_time = Column(DateTime(timezone=True), default=datetime.utcnow)
    symbols_scanned = Column(Integer, default=0)
    patterns_found = Column(Integer, default=0)
    scan_duration_ms = Column(Integer)
    status = Column(String(20))  # 'completed', 'failed', 'timeout'
    error_msg = Column(Text)

    def __repr__(self):
        return f"<RealtimeScanLog(time={self.scan_time}, patterns={self.patterns_found})>"


class StockFundamentals(Base):
    """
    Stock fundamental data from Yahoo Finance.

    Refreshed weekly/monthly for market cap filtering and analysis.
    Extensible for additional fundamentals (float, sector, industry, etc.)
    """
    __tablename__ = 'stock_fundamentals'

    symbol = Column(String(10), primary_key=True)
    market_cap = Column(DECIMAL(18, 2))  # Market capitalization in dollars
    shares_outstanding = Column(BigInteger)  # Total shares outstanding
    float_shares = Column(BigInteger)  # Tradable float
    sector = Column(String(100))  # e.g., "Technology"
    industry = Column(String(100))  # e.g., "Semiconductors"
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index('idx_fundamentals_market_cap', 'market_cap'),
        Index('idx_fundamentals_sector', 'sector'),
    )

    def __repr__(self):
        mcap_b = float(self.market_cap) / 1e9 if self.market_cap else 0
        return f"<StockFundamentals(symbol='{self.symbol}', market_cap=${mcap_b:.2f}B)>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'market_cap': float(self.market_cap) if self.market_cap else None,
            'shares_outstanding': self.shares_outstanding,
            'float_shares': self.float_shares,
            'sector': self.sector,
            'industry': self.industry,
            'updated_at': str(self.updated_at) if self.updated_at else None,
        }


# Database connection utilities
_engine = None
_SessionLocal = None


def get_engine(connection_string: str = None):
    """
    Get or create SQLAlchemy engine.

    Args:
        connection_string: PostgreSQL connection string
            Default: postgresql://kronos:kronos_dev@localhost:5432/market_data

    Returns:
        SQLAlchemy engine
    """
    global _engine

    if _engine is None:
        if connection_string is None:
            import os
            host = os.getenv('TIMESCALE_HOST', 'localhost')
            port = os.getenv('TIMESCALE_PORT', '5432')
            database = os.getenv('TIMESCALE_DB', 'market_data')
            user = os.getenv('TIMESCALE_USER', 'kronos')
            password = os.getenv('TIMESCALE_PASSWORD', 'kronos_dev')
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        _engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )

    return _engine


def get_session():
    """
    Get database session.

    Returns:
        SQLAlchemy session
    """
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False
        )

    return _SessionLocal()


def init_db(connection_string: str = None):
    """
    Initialize database tables.

    Creates all tables if they don't exist.
    """
    engine = get_engine(connection_string)
    Base.metadata.create_all(bind=engine)
