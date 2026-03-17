"""SQLAlchemy 2.0 ORM models for the OMS database."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String, Text, Boolean, Integer, DateTime, Numeric, ForeignKey, Index,
    UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    secret_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    base_url: Mapped[str] = mapped_column(
        String(200), default="https://paper-api.alpaca.markets"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    allowed_sources: Mapped[dict] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    risk_limits: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False
    )
    alpaca_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Order details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)
    action: Mapped[str] = mapped_column(String(10), nullable=False)  # open/close/cancel
    order_type: Mapped[str] = mapped_column(String(20), default="market")
    qty: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    notional: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    limit_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    filled_qty: Mapped[float] = mapped_column(Numeric(12, 4), default=0)
    filled_avg_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    time_in_force: Mapped[str] = mapped_column(String(10), default="day")

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Source + tags
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    tags: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    bracket_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    parent_order_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("orders.id"), nullable=True
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    filled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("idx_orders_account_status", "account_id", "status"),
        Index("idx_orders_symbol_time", "symbol", "created_at"),
        Index("idx_orders_source_time", "source", "created_at"),
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # long/short
    qty: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    current_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    market_value: Mapped[float] = mapped_column(Numeric(14, 2), default=0)
    entry_order_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("orders.id"), nullable=True
    )
    tags: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("account_id", "symbol", name="uq_position_account_symbol"),
        Index("idx_positions_account", "account_id"),
    )


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    qty: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    realized_pnl_pct: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    entry_order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False
    )
    exit_order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False
    )
    exit_reason: Mapped[str] = mapped_column(String(50), nullable=False)
    tags: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    hold_duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_trades_account_time", "account_id", "closed_at"),
        Index("idx_trades_closed_at", "closed_at"),
    )


class WebhookLog(Base):
    __tablename__ = "webhook_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    source: Mapped[str] = mapped_column(String(50), default="")
    action: Mapped[str] = mapped_column(String(10), default="")
    symbol: Mapped[str] = mapped_column(String(20), default="")
    idempotency_key: Mapped[str] = mapped_column(String(200), default="")
    request_body: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    response_status: Mapped[str] = mapped_column(String(20), default="")
    response_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    order_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("idx_webhook_logs_time", "received_at"),
    )


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    # Use (snapshot_at, id) composite PK so TimescaleDB can partition on snapshot_at
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, default=_utcnow, nullable=False
    )
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False
    )
    equity: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    market_value: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    daily_pnl: Mapped[float] = mapped_column(Numeric(12, 2), default=0)

    __table_args__ = (
        Index("idx_equity_account_time", "account_id", "snapshot_at"),
    )
