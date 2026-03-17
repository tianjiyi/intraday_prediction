"""Core order lifecycle: validate → execute → track."""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oms.db.models import Account, Order
from oms.schemas import WebhookRequest, OrderResponse, WebhookResponse
from oms.services.account_manager import AccountManager
from oms.services.alpaca_executor import AlpacaExecutor
from oms.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        account_manager: AccountManager,
        alpaca_executor: AlpacaExecutor,
        risk_manager: RiskManager,
        webhook_secret: str,
    ):
        self._session_factory = session_factory
        self._account_manager = account_manager
        self._executor = alpaca_executor
        self._risk_manager = risk_manager
        self._webhook_secret = webhook_secret

    async def handle_webhook(self, req: WebhookRequest) -> WebhookResponse:
        """Process an incoming webhook request end-to-end."""

        # 1. Validate token (skip if no secret configured — dev mode)
        if self._webhook_secret and req.token != self._webhook_secret:
            logger.warning(f"OMS: Invalid webhook token from source='{req.source}'")
            return WebhookResponse(status="rejected", reason="Invalid token")

        # 2. Check idempotency
        is_dup, existing = await self._risk_manager.check_idempotency(req.idempotency_key)
        if is_dup and existing:
            logger.info(f"OMS: Duplicate idempotency_key='{req.idempotency_key}', returning existing order")
            return WebhookResponse(
                status="duplicate",
                order=OrderResponse.model_validate(existing),
            )

        # 3. Resolve account
        account = await self._account_manager.resolve_account(req.account)
        if not account:
            return await self._create_rejected_order(
                req, None, f"Account '{req.account or 'default'}' not found or inactive"
            )

        # 4. Risk checks
        risk_result = await self._risk_manager.run_all_checks(
            account=account,
            source=req.source,
            action=req.action,
        )
        if not risk_result.passed:
            return await self._create_rejected_order(req, account, risk_result.reason)

        # 5. Dispatch by action
        if req.action == "open":
            return await self._handle_open(req, account)
        elif req.action == "close":
            return await self._handle_close(req, account)
        elif req.action == "cancel":
            return await self._handle_cancel(req, account)
        else:
            return WebhookResponse(status="rejected", reason=f"Unknown action: {req.action}")

    async def _handle_open(self, req: WebhookRequest, account: Account) -> WebhookResponse:
        """Place a new order."""
        if not req.symbol or not req.side:
            return await self._create_rejected_order(req, account, "symbol and side required for open")

        # Create order record
        order = Order(
            idempotency_key=req.idempotency_key,
            account_id=account.id,
            symbol=req.symbol.upper(),
            side=req.side,
            action="open",
            order_type=req.order_type,
            qty=req.qty,
            notional=req.notional,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            time_in_force=req.time_in_force,
            status="pending",
            source=req.source,
            tags=req.tags,
            bracket_config=req.bracket.model_dump() if req.bracket else None,
            metadata_=req.metadata,
        )

        async with self._session_factory() as session:
            session.add(order)
            await session.commit()
            await session.refresh(order)

        # Submit to Alpaca
        try:
            result = await self._executor.submit_order(
                account=account,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                qty=float(order.qty) if order.qty else None,
                notional=float(order.notional) if order.notional else None,
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                time_in_force=order.time_in_force,
                bracket=req.bracket.model_dump() if req.bracket else None,
            )

            # Update order with Alpaca response
            async with self._session_factory() as session:
                await session.execute(
                    update(Order).where(Order.id == order.id).values(
                        alpaca_order_id=result["alpaca_order_id"],
                        status=result["status"],
                        filled_qty=result["filled_qty"],
                        filled_avg_price=result["filled_avg_price"],
                        submitted_at=result["submitted_at"],
                        filled_at=datetime.now(timezone.utc) if result["status"] == "filled" else None,
                    )
                )
                await session.commit()

                # Reload for response
                res = await session.execute(select(Order).where(Order.id == order.id))
                order = res.scalar_one()

            logger.info(
                f"OMS: Order {order.id} submitted → alpaca_id={result['alpaca_order_id']}, "
                f"status={result['status']}"
            )
            return WebhookResponse(
                status="accepted",
                order=OrderResponse.model_validate(order),
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"OMS: Order submission failed: {error_msg}")

            async with self._session_factory() as session:
                await session.execute(
                    update(Order).where(Order.id == order.id).values(
                        status="rejected",
                        reject_reason=error_msg,
                    )
                )
                await session.commit()

                res = await session.execute(select(Order).where(Order.id == order.id))
                order = res.scalar_one()

            return WebhookResponse(
                status="rejected",
                order=OrderResponse.model_validate(order),
                reason=error_msg,
            )

    async def _handle_close(self, req: WebhookRequest, account: Account) -> WebhookResponse:
        """Close an entire position."""
        if not req.symbol:
            return await self._create_rejected_order(req, account, "symbol required for close")

        order = Order(
            idempotency_key=req.idempotency_key,
            account_id=account.id,
            symbol=req.symbol.upper(),
            side=None,
            action="close",
            order_type="market",
            status="pending",
            source=req.source,
            tags=req.tags,
            metadata_=req.metadata,
        )

        async with self._session_factory() as session:
            session.add(order)
            await session.commit()
            await session.refresh(order)

        try:
            result = await self._executor.close_position(account, req.symbol.upper())

            async with self._session_factory() as session:
                await session.execute(
                    update(Order).where(Order.id == order.id).values(
                        alpaca_order_id=result.get("alpaca_order_id"),
                        status="submitted",
                        submitted_at=result["submitted_at"],
                    )
                )
                await session.commit()

                res = await session.execute(select(Order).where(Order.id == order.id))
                order = res.scalar_one()

            return WebhookResponse(
                status="accepted",
                order=OrderResponse.model_validate(order),
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"OMS: Close position failed: {error_msg}")

            async with self._session_factory() as session:
                await session.execute(
                    update(Order).where(Order.id == order.id).values(
                        status="rejected",
                        reject_reason=error_msg,
                    )
                )
                await session.commit()

                res = await session.execute(select(Order).where(Order.id == order.id))
                order = res.scalar_one()

            return WebhookResponse(
                status="rejected",
                order=OrderResponse.model_validate(order),
                reason=error_msg,
            )

    async def _handle_cancel(self, req: WebhookRequest, account: Account) -> WebhookResponse:
        """Cancel a pending order."""
        if not req.order_id:
            return WebhookResponse(status="rejected", reason="order_id required for cancel")

        # Find the order
        async with self._session_factory() as session:
            result = await session.execute(
                select(Order).where(Order.id == req.order_id)
            )
            order = result.scalar_one_or_none()

        if not order:
            return WebhookResponse(status="rejected", reason=f"Order {req.order_id} not found")

        if not order.alpaca_order_id:
            return WebhookResponse(status="rejected", reason="Order has no Alpaca ID to cancel")

        try:
            await self._executor.cancel_order(account, order.alpaca_order_id)

            async with self._session_factory() as session:
                await session.execute(
                    update(Order).where(Order.id == order.id).values(status="canceled")
                )
                await session.commit()

                res = await session.execute(select(Order).where(Order.id == order.id))
                order = res.scalar_one()

            return WebhookResponse(
                status="accepted",
                order=OrderResponse.model_validate(order),
            )

        except Exception as e:
            return WebhookResponse(status="rejected", reason=str(e))

    async def _create_rejected_order(
        self, req: WebhookRequest, account: Account | None, reason: str
    ) -> WebhookResponse:
        """Create a rejected order record for audit trail."""
        logger.warning(f"OMS: Order rejected: {reason}")

        order = Order(
            idempotency_key=req.idempotency_key,
            account_id=account.id if account else None,
            symbol=req.symbol.upper() if req.symbol else "",
            side=req.side,
            action=req.action,
            order_type=req.order_type,
            qty=req.qty,
            notional=req.notional,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            status="rejected",
            reject_reason=reason,
            source=req.source,
            tags=req.tags,
            metadata_=req.metadata,
        )

        # Only persist if we have an account_id (FK required)
        if account:
            async with self._session_factory() as session:
                session.add(order)
                await session.commit()
                await session.refresh(order)

            return WebhookResponse(
                status="rejected",
                order=OrderResponse.model_validate(order),
                reason=reason,
            )

        return WebhookResponse(status="rejected", reason=reason)

    async def list_orders(
        self,
        account_name: str | None = None,
        status: str | None = None,
        source: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[Order]:
        """List orders with optional filters."""
        async with self._session_factory() as session:
            query = select(Order).order_by(Order.created_at.desc()).limit(limit)

            if status:
                query = query.where(Order.status == status)
            if source:
                query = query.where(Order.source == source)
            if symbol:
                query = query.where(Order.symbol == symbol.upper())
            if account_name:
                acct = await self._account_manager.get_account_by_name(account_name)
                if acct:
                    query = query.where(Order.account_id == acct.id)

            result = await session.execute(query)
            return list(result.scalars().all())
