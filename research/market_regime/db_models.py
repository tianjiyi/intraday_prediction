"""
SQLAlchemy Models for Market Data

Defines ORM models for trades, quotes, features, and ingestion log tables.
These models can be used for type-safe queries and data insertion.
"""

from datetime import datetime
from typing import Optional, List
from decimal import Decimal

from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Date, DECIMAL, ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Trade(Base):
    """
    Trade record from Polygon tick data.

    Represents a single trade execution with price, size, and metadata.
    """
    __tablename__ = 'trades'

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String(10), primary_key=True, nullable=False)
    price = Column(DECIMAL(12, 4), primary_key=True, nullable=False)
    size = Column(Integer, primary_key=True, nullable=False)
    exchange = Column(String(10), nullable=True)
    conditions = Column(ARRAY(String), nullable=True)

    def __repr__(self):
        return f"<Trade({self.symbol} @ {self.time}: {self.size} @ ${self.price})>"

    def to_dict(self) -> dict:
        return {
            'time': self.time.isoformat() if self.time else None,
            'symbol': self.symbol,
            'price': float(self.price) if self.price else None,
            'size': self.size,
            'exchange': self.exchange,
            'conditions': self.conditions
        }


class Quote(Base):
    """
    Quote record (NBBO) from Polygon tick data.

    Represents best bid/ask at a given timestamp.
    """
    __tablename__ = 'quotes'

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String(10), primary_key=True, nullable=False)
    bid_price = Column(DECIMAL(12, 4), nullable=True)
    bid_size = Column(Integer, nullable=True)
    ask_price = Column(DECIMAL(12, 4), nullable=True)
    ask_size = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<Quote({self.symbol} @ {self.time}: {self.bid_price}/{self.ask_price})>"

    def to_dict(self) -> dict:
        return {
            'time': self.time.isoformat() if self.time else None,
            'symbol': self.symbol,
            'bid_price': float(self.bid_price) if self.bid_price else None,
            'bid_size': self.bid_size,
            'ask_price': float(self.ask_price) if self.ask_price else None,
            'ask_size': self.ask_size
        }

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid_price and self.ask_price:
            return (float(self.bid_price) + float(self.ask_price)) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid_price and self.ask_price:
            return float(self.ask_price) - float(self.bid_price)
        return None


class Feature(Base):
    """
    Computed features for ML training.

    Stores OFI, VPIN, HMM state, and other derived metrics
    at various timeframe resolutions.
    """
    __tablename__ = 'features'

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String(10), primary_key=True, nullable=False)
    timeframe = Column(Integer, primary_key=True, nullable=False)  # minutes
    ofi = Column(DECIMAL(16, 4), nullable=True)
    vpin = Column(DECIMAL(8, 4), nullable=True)
    hmm_state = Column(Integer, nullable=True)
    close = Column(DECIMAL(12, 4), nullable=True)
    volume = Column(BigInteger, nullable=True)

    def __repr__(self):
        return f"<Feature({self.symbol} {self.timeframe}m @ {self.time}: HMM={self.hmm_state}, VPIN={self.vpin})>"

    def to_dict(self) -> dict:
        return {
            'time': self.time.isoformat() if self.time else None,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'ofi': float(self.ofi) if self.ofi else None,
            'vpin': float(self.vpin) if self.vpin else None,
            'hmm_state': self.hmm_state,
            'close': float(self.close) if self.close else None,
            'volume': self.volume
        }


class IngestionLog(Base):
    """
    Tracks which dates have been ingested for each symbol.

    Used to avoid re-ingesting data and track data coverage.
    """
    __tablename__ = 'ingestion_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    data_type = Column(String(20), nullable=False)  # 'trades' or 'quotes'
    record_count = Column(Integer, nullable=False)
    ingested_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    source = Column(String(50), nullable=True)

    def __repr__(self):
        return f"<IngestionLog({self.symbol} {self.data_type} {self.date}: {self.record_count} records)>"


# Utility functions for bulk operations

def trades_from_dataframe(df, symbol: str) -> List[dict]:
    """
    Convert a pandas DataFrame to list of trade dicts for bulk insert.

    Expected columns: timestamp, price, size, [exchange], [conditions]
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            'time': row['timestamp'],
            'symbol': symbol,
            'price': row['price'],
            'size': row['size'],
            'exchange': row.get('exchange'),
            'conditions': row.get('conditions')
        })
    return records


def quotes_from_dataframe(df, symbol: str) -> List[dict]:
    """
    Convert a pandas DataFrame to list of quote dicts for bulk insert.

    Expected columns: timestamp, bid_price, bid_size, ask_price, ask_size
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            'time': row['timestamp'],
            'symbol': symbol,
            'bid_price': row.get('bid_price'),
            'bid_size': row.get('bid_size'),
            'ask_price': row.get('ask_price'),
            'ask_size': row.get('ask_size')
        })
    return records
