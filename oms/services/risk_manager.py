"""Pre-trade risk checks for incoming webhook orders."""

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oms.db.models import Account, Order, Position, Trade

logger = logging.getLogger(__name__)


class RiskCheckResult:
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    @staticmethod
    def ok():
        return RiskCheckResult(True)

    @staticmethod
    def fail(reason: str):
        return RiskCheckResult(False, reason)


class RiskManager:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def check_idempotency(self, idempotency_key: str) -> tuple[bool, Order | None]:
        """
        Check if an order with this idempotency_key already exists.
        Returns (is_duplicate, existing_order).
        If the existing order was rejected/canceled, delete it to allow retry.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Order).where(Order.idempotency_key == idempotency_key)
            )
            existing = result.scalar_one_or_none()
            if not existing:
                return False, None
            if existing.status in ("rejected", "canceled"):
                # Allow retry — delete the old rejected/canceled order
                await session.delete(existing)
                await session.commit()
                return False, None
            return True, existing

    async def check_source_allowed(self, account: Account, source: str) -> RiskCheckResult:
        """Check if the source is allowed for this account."""
        allowed = account.allowed_sources
        if not allowed or len(allowed) == 0:
            return RiskCheckResult.ok()  # Empty = all sources allowed
        if source in allowed:
            return RiskCheckResult.ok()
        return RiskCheckResult.fail(f"Source '{source}' not in allowed sources: {allowed}")

    async def check_daily_loss(self, account_id, risk_limits: dict) -> RiskCheckResult:
        """Check if daily realized loss exceeds limit."""
        max_daily_loss = risk_limits.get("max_daily_loss")
        if not max_daily_loss:
            return RiskCheckResult.ok()

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        async with self._session_factory() as session:
            result = await session.execute(
                select(func.sum(Trade.realized_pnl)).where(
                    Trade.account_id == account_id,
                    Trade.closed_at >= today_start,
                )
            )
            daily_pnl = result.scalar() or 0

        if daily_pnl < -abs(max_daily_loss):
            return RiskCheckResult.fail(
                f"Daily loss limit exceeded: ${daily_pnl:.2f} (limit: -${max_daily_loss})"
            )
        return RiskCheckResult.ok()

    async def check_max_positions(self, account_id, risk_limits: dict) -> RiskCheckResult:
        """Check if max concurrent positions exceeded."""
        max_positions = risk_limits.get("max_positions")
        if not max_positions:
            return RiskCheckResult.ok()

        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count(Position.id)).where(Position.account_id == account_id)
            )
            count = result.scalar() or 0

        if count >= max_positions:
            return RiskCheckResult.fail(
                f"Max positions exceeded: {count} (limit: {max_positions})"
            )
        return RiskCheckResult.ok()

    async def run_all_checks(
        self,
        account: Account,
        source: str,
        action: str,
    ) -> RiskCheckResult:
        """Run all applicable risk checks for an order."""
        if action != "open":
            return RiskCheckResult.ok()  # Only check risk for new positions

        # Source allowed
        check = await self.check_source_allowed(account, source)
        if not check.passed:
            return check

        # Daily loss
        check = await self.check_daily_loss(account.id, account.risk_limits)
        if not check.passed:
            return check

        # Max positions
        check = await self.check_max_positions(account.id, account.risk_limits)
        if not check.passed:
            return check

        return RiskCheckResult.ok()
