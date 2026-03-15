"""
SQLAlchemy 2.0 ORM models for the Agent Memory System.

Tables:
  - strategies:       Strategy definitions and parameters
  - signals:          Trading signals with indicator snapshots
  - decisions:        Explicit user decisions from chat
  - user_preferences: User settings (risk, watchlist, notifications)
  - chat_messages:    Persistent chat history
  - agent_memories:   Vector store for RAG (pgvector)
  - enriched_news:    High-impact news with embeddings for theme analysis
  - themes:           Persistent market themes (AI-identified)
  - theme_history:    Theme lifecycle snapshots over time
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String, Text, Boolean, Integer, DateTime, Numeric, ForeignKey, Index,
    UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ============================================================
# Tier A: Structured Long-Term Memory
# ============================================================

class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    applicable_symbols = mapped_column(ARRAY(String), default=list)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    signals: Mapped[list["Signal"]] = relationship(back_populates="strategy")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_strategy_name_version"),
        Index("idx_strategies_enabled", "is_enabled", postgresql_where=text("is_enabled = true")),
    )

    def __repr__(self):
        return f"<Strategy {self.name} v{self.version}>"


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe_minutes: Mapped[int] = mapped_column(Integer, default=1)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)  # long, short, hold, exit
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    kronos_p_up: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    kronos_exp_return: Mapped[float | None] = mapped_column(Numeric(8, 6), nullable=True)
    indicators: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    percentiles: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    daily_context: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    market_regime: Mapped[str | None] = mapped_column(String(30), nullable=True)
    llm_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    strategy: Mapped[Strategy | None] = relationship(back_populates="signals")

    __table_args__ = (
        Index("idx_signals_symbol_time", "symbol", created_at.desc()),
        Index("idx_signals_strategy", "strategy_id",
              postgresql_where=text("strategy_id IS NOT NULL")),
    )

    def __repr__(self):
        return f"<Signal {self.signal_type} {self.symbol} conf={self.confidence}>"


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    decision_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_rule: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    source: Mapped[str] = mapped_column(String(30), default="chat")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decisions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("idx_decisions_active", "is_active",
              postgresql_where=text("is_active = true")),
    )

    def __repr__(self):
        return f"<Decision active={self.is_active} '{self.decision_text[:40]}...'>"


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("category", "key", name="uq_pref_category_key"),
    )

    def __repr__(self):
        return f"<UserPreference {self.category}.{self.key}>"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_chat_session", "session_id", "created_at"),
        Index("idx_chat_time", created_at.desc()),
    )

    def __repr__(self):
        return f"<ChatMessage {self.role} session={str(self.session_id)[:8]}>"


# ============================================================
# Tier B: Unstructured Memory (pgvector RAG)
# ============================================================

class AgentMemory(Base):
    __tablename__ = "agent_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(384), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # Types: experience, principle, review, macro_judgment, lesson, pattern
    source: Mapped[str] = mapped_column(String(30), default="chat")
    # Sources: chat, backtest, manual, analysis
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=True
    )
    market_regime: Mapped[str | None] = mapped_column(String(30), nullable=True)
    importance_score: Mapped[float] = mapped_column(Numeric(3, 2), default=0.50)
    decay_weight: Mapped[float] = mapped_column(Numeric(5, 4), default=1.0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        # HNSW index created via raw SQL in init_db() for pgvector
        Index("idx_memories_type", "memory_type"),
        Index("idx_memories_symbol", "symbol", postgresql_where=text("symbol IS NOT NULL")),
        Index("idx_memories_importance", importance_score.desc()),
    )

    def __repr__(self):
        return f"<AgentMemory {self.memory_type} imp={self.importance_score} '{self.content[:30]}...'>"


# ============================================================
# Tier C: Theme Intelligence
# ============================================================

class EnrichedNews(Base):
    __tablename__ = "enriched_news"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding = mapped_column(Vector(384), nullable=True)
    sentiment: Mapped[str] = mapped_column(String(20), default="neutral")
    sectors = mapped_column(ARRAY(String), default=list)
    tickers = mapped_column(ARRAY(String), default=list)
    impact_score: Mapped[float] = mapped_column(Numeric(6, 2), default=0.0)
    source: Mapped[str] = mapped_column(String(30), default="alpaca")
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_enriched_news_time", timestamp.desc()),
        Index("idx_enriched_news_impact", impact_score.desc()),
        Index("idx_enriched_news_source", "source"),
    )

    def __repr__(self):
        return f"<EnrichedNews {self.source} score={self.impact_score} '{self.headline[:40]}...'>"


class Theme(Base):
    __tablename__ = "themes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    lifecycle_stage: Mapped[str] = mapped_column(
        String(20), default="emerging"  # emerging, hot, cooling, faded
    )
    confidence: Mapped[float] = mapped_column(Numeric(3, 2), default=0.50)
    related_tickers = mapped_column(ARRAY(String), default=list)
    related_sectors = mapped_column(ARRAY(String), default=list)
    news_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    history: Mapped[list["ThemeHistory"]] = relationship(back_populates="theme")

    __table_args__ = (
        Index("idx_themes_stage", "lifecycle_stage"),
        Index("idx_themes_updated", last_updated.desc()),
    )

    def __repr__(self):
        return f"<Theme '{self.name}' stage={self.lifecycle_stage} conf={self.confidence}>"


class ThemeHistory(Base):
    __tablename__ = "theme_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    theme_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("themes.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    lifecycle_stage: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(3, 2), default=0.50)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    news_count: Mapped[int] = mapped_column(Integer, default=0)

    theme: Mapped[Theme] = relationship(back_populates="history")

    __table_args__ = (
        Index("idx_theme_history_theme_time", "theme_id", snapshot_at.desc()),
    )

    def __repr__(self):
        return f"<ThemeHistory theme={str(self.theme_id)[:8]} stage={self.lifecycle_stage}>"
