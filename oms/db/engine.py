"""Async PostgreSQL engine and session management for OMS database."""

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


async def create_db_engine(database_url: str, force: bool = False) -> AsyncEngine:
    """Create and return the async engine (singleton)."""
    global _engine
    if _engine is not None and not force:
        return _engine

    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception:
            pass

    # Log host only, not credentials
    safe_url = database_url.split("@")[-1] if "@" in database_url else database_url
    logger.info(f"OMS: Creating database engine: {safe_url}")

    _engine = create_async_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    return _engine


def get_session_factory(engine: AsyncEngine = None) -> async_sessionmaker[AsyncSession]:
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
    """Create all tables on startup."""
    from .models import Base
    import sqlalchemy as sa

    eng = engine or _engine
    if eng is None:
        raise RuntimeError("Database engine not created.")

    # Each step in its own transaction — PostgreSQL aborts all commands
    # after a failed statement within the same transaction.

    # Step 1: Optional extension (separate transaction so failure doesn't break table creation)
    try:
        async with eng.begin() as conn:
            await conn.execute(sa.text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
    except Exception as e:
        logger.warning(f"OMS: uuid-ossp extension not available: {e}")

    # Step 2: Create tables (must succeed)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("OMS: Database tables created/verified")

    # Step 3: Optional TimescaleDB hypertable (separate transaction)
    try:
        async with eng.begin() as conn:
            await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            await conn.execute(sa.text(
                "SELECT create_hypertable('equity_snapshots', 'snapshot_at', "
                "if_not_exists => TRUE)"
            ))
            logger.info("OMS: TimescaleDB hypertable created for equity_snapshots")
    except Exception as e:
        logger.info(f"OMS: TimescaleDB not available, using regular table ({e})")


async def close_engine():
    """Dispose the engine and clean up connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        logger.info("OMS: Database engine closed")
    _engine = None
    _session_factory = None
