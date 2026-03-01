"""
Async SQLAlchemy engine and session management for PostgreSQL.

Connection is configured via environment variables:
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import os
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _build_database_url() -> str:
    """Build async PostgreSQL URL from environment variables."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "agent_memory")
    user = os.environ.get("POSTGRES_USER", "kronos")
    password = os.environ.get("POSTGRES_PASSWORD", "kronos_dev")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


async def create_db_engine(config: dict = None) -> AsyncEngine:
    """Create and return the async engine (singleton)."""
    global _engine
    if _engine is not None:
        return _engine

    url = _build_database_url()
    logger.info(f"Creating database engine: {url.split('@')[1]}")  # log host only, no creds

    _engine = create_async_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    return _engine


def get_async_session_factory(engine: AsyncEngine = None) -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    eng = engine or _engine
    if eng is None:
        raise RuntimeError("Database engine not created. Call create_db_engine() first.")

    _session_factory = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def init_db(engine: AsyncEngine = None):
    """Create all tables (development mode). Use Alembic for production migrations."""
    from .models import Base

    eng = engine or _engine
    if eng is None:
        raise RuntimeError("Database engine not created.")

    async with eng.begin() as conn:
        # Enable pgvector extension
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created/verified")


async def test_connection(engine: AsyncEngine = None) -> bool:
    """Test database connectivity. Returns True if successful."""
    eng = engine or _engine
    if eng is None:
        return False

    try:
        async with eng.connect() as conn:
            result = await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection test: OK")
        return True
    except Exception as e:
        logger.warning(f"Database connection test failed: {e}")
        return False


async def close_engine():
    """Dispose the engine and clean up connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        logger.info("Database engine closed")
    _engine = None
    _session_factory = None
