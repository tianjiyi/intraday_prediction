"""Pydantic request/response models for the OMS webhook API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Webhook request
# ---------------------------------------------------------------------------

class BracketParams(BaseModel):
    stop_loss: float | None = None
    take_profit: float | None = None


class WebhookRequest(BaseModel):
    """Universal webhook payload — accepted from all source systems."""

    token: str
    idempotency_key: str
    source: str  # freeform: "platform", "tradingview", "external", etc.
    account: str | None = None  # account name; None → default account
    action: str  # "open", "close", "cancel", "buy", "sell"

    # Order fields (required for action=open)
    symbol: str = ""
    side: str | None = None  # "buy" or "sell" (flexible for TradingView)
    order_type: str = "market"  # "market", "limit", "stop", "stop_limit"
    limit_price: float | None = None
    stop_price: float | None = None
    qty: float | str | None = None  # Accept string from TradingView placeholders
    notional: float | None = None
    time_in_force: str = "day"  # "day", "gtc", "ioc", "fok"

    @field_validator("qty", mode="before")
    @classmethod
    def coerce_qty(cls, v):
        if v is None or v == "":
            return None
        return float(v)

    # Bracket order
    bracket: BracketParams | None = None

    # Freeform tags for filtering/reporting
    tags: dict[str, Any] | None = None

    # Additional context (stored, not used for execution)
    metadata: dict[str, Any] | None = None

    # For cancel action
    order_id: str | None = None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class OrderResponse(BaseModel):
    id: UUID
    idempotency_key: str
    account_id: UUID
    alpaca_order_id: str | None = None
    symbol: str
    side: str | None = None
    action: str
    order_type: str
    qty: float | None = None
    notional: float | None = None
    limit_price: float | None = None
    filled_qty: float = 0
    filled_avg_price: float | None = None
    status: str
    reject_reason: str | None = None
    source: str
    tags: dict | None = None
    created_at: datetime
    submitted_at: datetime | None = None
    filled_at: datetime | None = None

    class Config:
        from_attributes = True


class AccountCreate(BaseModel):
    name: str
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    is_default: bool = False
    allowed_sources: list[str] = Field(default_factory=list)
    risk_limits: dict[str, Any] = Field(default_factory=dict)


class AccountUpdate(BaseModel):
    is_active: bool | None = None
    is_default: bool | None = None
    allowed_sources: list[str] | None = None
    risk_limits: dict[str, Any] | None = None


class AccountResponse(BaseModel):
    id: UUID
    name: str
    base_url: str
    is_active: bool
    is_default: bool
    allowed_sources: list
    risk_limits: dict
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    database: str
    accounts: list[dict]


class WebhookResponse(BaseModel):
    status: str  # "accepted", "rejected", "duplicate"
    order: OrderResponse | None = None
    reason: str | None = None
