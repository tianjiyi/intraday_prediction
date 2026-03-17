"""Background position sync and equity snapshots from Alpaca."""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oms.db.models import Account, Position, Trade, Order, EquitySnapshot
from oms.services.alpaca_executor import AlpacaExecutor

logger = logging.getLogger(__name__)


class PositionTracker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        alpaca_executor: AlpacaExecutor,
        position_sync_interval: int = 30,
        equity_snapshot_interval: int = 300,
    ):
        self._session_factory = session_factory
        self._executor = alpaca_executor
        self._position_sync_interval = position_sync_interval
        self._equity_snapshot_interval = equity_snapshot_interval
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """Start background sync tasks."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._position_sync_loop()),
            asyncio.create_task(self._equity_snapshot_loop()),
        ]
        logger.info(
            f"OMS: Position tracker started "
            f"(sync={self._position_sync_interval}s, snapshot={self._equity_snapshot_interval}s)"
        )

    async def stop(self):
        """Stop background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("OMS: Position tracker stopped")

    # ------------------------------------------------------------------
    # Position sync loop
    # ------------------------------------------------------------------

    async def _position_sync_loop(self):
        while self._running:
            try:
                await self._sync_all_accounts()
            except Exception as e:
                logger.error(f"OMS: Position sync error: {e}")
            await asyncio.sleep(self._position_sync_interval)

    async def _sync_all_accounts(self):
        async with self._session_factory() as session:
            result = await session.execute(
                select(Account).where(Account.is_active == True)
            )
            accounts = list(result.scalars().all())

        for account in accounts:
            try:
                await self._sync_account_positions(account)
            except Exception as e:
                logger.warning(f"OMS: Position sync failed for '{account.name}': {e}")

    async def _sync_account_positions(self, account: Account):
        """Sync positions from Alpaca to local DB. Detect closed positions → create trades."""
        alpaca_positions = await self._executor.get_positions(account)
        alpaca_symbols = {p["symbol"] for p in alpaca_positions}

        async with self._session_factory() as session:
            # Get existing DB positions for this account
            result = await session.execute(
                select(Position).where(Position.account_id == account.id)
            )
            db_positions = {p.symbol: p for p in result.scalars().all()}

            # Update or create positions from Alpaca
            for ap in alpaca_positions:
                if ap["symbol"] in db_positions:
                    # Update existing
                    pos = db_positions[ap["symbol"]]
                    pos.qty = ap["qty"]
                    pos.current_price = ap["current_price"]
                    pos.unrealized_pnl = ap["unrealized_pnl"]
                    pos.market_value = ap["market_value"]
                    pos.entry_price = ap["entry_price"]
                    pos.side = ap["side"]
                    session.add(pos)
                else:
                    # New position — try to find the entry order
                    entry_order_result = await session.execute(
                        select(Order)
                        .where(
                            Order.account_id == account.id,
                            Order.symbol == ap["symbol"],
                            Order.action == "open",
                            Order.status.in_(["accepted", "filled", "submitted"]),
                        )
                        .order_by(Order.created_at.desc())
                        .limit(1)
                    )
                    entry_order = entry_order_result.scalar_one_or_none()

                    new_pos = Position(
                        account_id=account.id,
                        symbol=ap["symbol"],
                        side=ap["side"],
                        qty=ap["qty"],
                        entry_price=ap["entry_price"],
                        current_price=ap["current_price"],
                        unrealized_pnl=ap["unrealized_pnl"],
                        market_value=ap["market_value"],
                        entry_order_id=entry_order.id if entry_order else None,
                        tags=entry_order.tags if entry_order else None,
                    )
                    session.add(new_pos)
                    logger.info(f"OMS: New position detected: {ap['symbol']} on '{account.name}'")

            # Detect closed positions (in DB but not on Alpaca)
            for symbol, pos in db_positions.items():
                if symbol not in alpaca_symbols:
                    await self._record_trade(session, account, pos)
                    await session.delete(pos)
                    logger.info(f"OMS: Position closed: {symbol} on '{account.name}'")

            await session.commit()

    async def _record_trade(self, session: AsyncSession, account: Account, pos: Position):
        """Create a round-trip trade record when a position is closed."""
        # Use current_price as exit price (last known from Alpaca)
        exit_price = float(pos.current_price)
        entry_price = float(pos.entry_price)
        qty = float(pos.qty)

        if pos.side == "long":
            realized_pnl = (exit_price - entry_price) * qty
        else:
            realized_pnl = (entry_price - exit_price) * qty

        realized_pnl_pct = realized_pnl / (entry_price * qty) if entry_price * qty > 0 else 0

        # Find the entry order (best effort)
        entry_order_id = pos.entry_order_id

        # Find the most recent close order for this symbol
        exit_order_result = await session.execute(
            select(Order)
            .where(
                Order.account_id == account.id,
                Order.symbol == pos.symbol,
                Order.action == "close",
            )
            .order_by(Order.created_at.desc())
            .limit(1)
        )
        exit_order = exit_order_result.scalar_one_or_none()

        # Determine exit reason from the exit order or default
        exit_reason = "unknown"
        if exit_order:
            exit_order_id = exit_order.id
            if exit_order.tags and exit_order.tags.get("reason"):
                exit_reason = exit_order.tags["reason"]
            else:
                exit_reason = "manual"
        else:
            exit_order_id = entry_order_id  # fallback

        if not entry_order_id or not exit_order_id:
            logger.warning(
                f"OMS: Could not find entry/exit orders for trade {pos.symbol}. "
                f"Skipping trade record."
            )
            return

        hold_seconds = int((datetime.now(timezone.utc) - pos.opened_at).total_seconds())

        trade = Trade(
            account_id=account.id,
            symbol=pos.symbol,
            side=pos.side,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl=round(realized_pnl, 2),
            realized_pnl_pct=round(realized_pnl_pct, 4),
            entry_order_id=entry_order_id,
            exit_order_id=exit_order_id,
            exit_reason=exit_reason,
            tags=pos.tags,
            hold_duration_seconds=hold_seconds,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
        )
        session.add(trade)

    # ------------------------------------------------------------------
    # Equity snapshot loop
    # ------------------------------------------------------------------

    async def _equity_snapshot_loop(self):
        while self._running:
            try:
                await self._snapshot_all_accounts()
            except Exception as e:
                logger.error(f"OMS: Equity snapshot error: {e}")
            await asyncio.sleep(self._equity_snapshot_interval)

    async def _snapshot_all_accounts(self):
        async with self._session_factory() as session:
            result = await session.execute(
                select(Account).where(Account.is_active == True)
            )
            accounts = list(result.scalars().all())

        for account in accounts:
            try:
                await self._snapshot_account(account)
            except Exception as e:
                logger.warning(f"OMS: Equity snapshot failed for '{account.name}': {e}")

    async def _snapshot_account(self, account: Account):
        info = await self._executor.get_account_info(account)

        # Calculate daily P&L (equity - cash = market value, approximate)
        daily_pnl = info.get("daily_pnl", 0)

        snapshot = EquitySnapshot(
            account_id=account.id,
            equity=info["equity"],
            cash=info["cash"],
            market_value=info.get("market_value", info["equity"] - info["cash"]),
            daily_pnl=daily_pnl,
        )

        async with self._session_factory() as session:
            session.add(snapshot)
            await session.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_positions(self, account_name: str | None = None) -> list[dict]:
        """Get positions from DB."""
        async with self._session_factory() as session:
            query = select(Position)
            if account_name:
                query = query.join(Account).where(Account.name == account_name)
            result = await session.execute(query)
            positions = result.scalars().all()
            return [
                {
                    "id": str(p.id),
                    "account_id": str(p.account_id),
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": float(p.qty),
                    "entry_price": float(p.entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pnl),
                    "market_value": float(p.market_value),
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                }
                for p in positions
            ]

    async def get_trades(
        self,
        account_name: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get completed trades from DB."""
        async with self._session_factory() as session:
            query = select(Trade).order_by(Trade.closed_at.desc()).limit(limit)
            if account_name:
                query = query.join(Account).where(Account.name == account_name)
            if symbol:
                query = query.where(Trade.symbol == symbol.upper())
            result = await session.execute(query)
            trades = result.scalars().all()
            return [
                {
                    "id": str(t.id),
                    "account_id": str(t.account_id),
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": float(t.qty),
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "realized_pnl": float(t.realized_pnl),
                    "realized_pnl_pct": float(t.realized_pnl_pct),
                    "exit_reason": t.exit_reason,
                    "hold_duration_seconds": t.hold_duration_seconds,
                    "opened_at": t.opened_at.isoformat(),
                    "closed_at": t.closed_at.isoformat(),
                    "tags": t.tags,
                }
                for t in trades
            ]
