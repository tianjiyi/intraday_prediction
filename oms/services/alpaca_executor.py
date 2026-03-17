"""Alpaca TradingClient wrapper — one client per account, lazily cached."""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)

from oms.db.models import Account

logger = logging.getLogger(__name__)


class AlpacaExecutor:
    def __init__(self, account_manager):
        self._account_manager = account_manager
        self._clients: dict[UUID, TradingClient] = {}

    def _get_client(self, account: Account) -> TradingClient:
        """Get or create a TradingClient for the given account."""
        if account.id in self._clients:
            return self._clients[account.id]

        api_key, secret_key = self._account_manager.get_credentials(account)
        is_paper = "paper" in account.base_url

        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=is_paper,
        )
        self._clients[account.id] = client
        logger.info(f"OMS: Alpaca client created for account '{account.name}' (paper={is_paper})")
        return client

    async def submit_order(
        self,
        account: Account,
        symbol: str,
        side: str,
        order_type: str = "market",
        qty: float | None = None,
        notional: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day",
        bracket: dict | None = None,
    ) -> dict:
        """
        Submit an order to Alpaca. Returns a dict with order details.
        Raises on Alpaca API errors.
        """
        client = self._get_client(account)

        alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        alpaca_tif = _map_tif(time_in_force)

        # Build order request
        order_kwargs = {
            "symbol": symbol,
            "side": alpaca_side,
            "time_in_force": alpaca_tif,
        }

        if qty is not None:
            order_kwargs["qty"] = qty
        elif notional is not None:
            order_kwargs["notional"] = notional

        # Bracket order
        if bracket and (bracket.get("stop_loss") or bracket.get("take_profit")):
            order_kwargs["order_class"] = OrderClass.BRACKET
            if bracket.get("take_profit"):
                order_kwargs["take_profit"] = TakeProfitRequest(
                    limit_price=bracket["take_profit"]
                )
            if bracket.get("stop_loss"):
                order_kwargs["stop_loss"] = StopLossRequest(
                    stop_price=bracket["stop_loss"]
                )

        # Create appropriate request type
        if order_type == "market":
            request = MarketOrderRequest(**order_kwargs)
        elif order_type == "limit":
            order_kwargs["limit_price"] = limit_price
            request = LimitOrderRequest(**order_kwargs)
        elif order_type == "stop":
            order_kwargs["stop_price"] = stop_price
            request = StopOrderRequest(**order_kwargs)
        elif order_type == "stop_limit":
            order_kwargs["limit_price"] = limit_price
            order_kwargs["stop_price"] = stop_price
            request = StopLimitOrderRequest(**order_kwargs)
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        logger.info(f"OMS: Submitting {order_type} {side} {qty or notional} {symbol} to '{account.name}'")

        # Alpaca's submit_order is synchronous
        response = client.submit_order(request)

        return {
            "alpaca_order_id": str(response.id),
            "status": str(response.status.value) if response.status else "submitted",
            "filled_qty": float(response.filled_qty) if response.filled_qty else 0,
            "filled_avg_price": float(response.filled_avg_price) if response.filled_avg_price else None,
            "submitted_at": datetime.now(timezone.utc),
        }

    async def cancel_order(self, account: Account, alpaca_order_id: str) -> dict:
        """Cancel an order on Alpaca."""
        client = self._get_client(account)
        client.cancel_order_by_id(alpaca_order_id)
        logger.info(f"OMS: Canceled order {alpaca_order_id} on '{account.name}'")
        return {"status": "canceled"}

    async def close_position(self, account: Account, symbol: str) -> dict:
        """Close entire position in a symbol."""
        client = self._get_client(account)
        response = client.close_position(symbol)
        logger.info(f"OMS: Closed position {symbol} on '{account.name}'")
        return {
            "alpaca_order_id": str(response.id) if hasattr(response, "id") else None,
            "status": "submitted",
            "submitted_at": datetime.now(timezone.utc),
        }

    async def get_account_info(self, account: Account) -> dict:
        """Get account equity, cash, buying power."""
        client = self._get_client(account)
        info = client.get_account()
        return {
            "equity": float(info.equity),
            "cash": float(info.cash),
            "buying_power": float(info.buying_power),
            "portfolio_value": float(info.portfolio_value) if info.portfolio_value else float(info.equity),
        }

    async def get_positions(self, account: Account) -> list[dict]:
        """Get all open positions from Alpaca."""
        client = self._get_client(account)
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "side": "long" if float(p.qty) > 0 else "short",
                "qty": abs(float(p.qty)),
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pnl": float(p.unrealized_pl),
                "market_value": float(p.market_value),
            }
            for p in positions
        ]

    async def check_connectivity(self, account: Account) -> bool:
        """Test if we can connect to Alpaca with this account."""
        try:
            client = self._get_client(account)
            client.get_account()
            return True
        except Exception as e:
            logger.warning(f"OMS: Connectivity check failed for '{account.name}': {e}")
            return False


def _map_tif(tif: str) -> TimeInForce:
    mapping = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }
    return mapping.get(tif, TimeInForce.DAY)
