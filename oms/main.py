"""
OMS — Order Management Service
Standalone FastAPI microservice for order execution on Alpaca.

Usage:
    python oms/main.py
    # or
    uvicorn oms.main:app --host 0.0.0.0 --port 8100
"""

import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure project root is on sys.path for imports
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from oms.config import Settings, load_settings
from oms.db.engine import create_db_engine, get_session_factory, init_db, close_engine
from oms.schemas import (
    WebhookRequest, WebhookResponse, OrderResponse,
    AccountCreate, AccountUpdate, AccountResponse, HealthResponse,
)
from oms.services.account_manager import AccountManager
from oms.services.alpaca_executor import AlpacaExecutor
from oms.services.order_service import OrderService
from oms.services.position_tracker import PositionTracker
from oms.services.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

settings: Optional[Settings] = None
order_service: Optional[OrderService] = None
account_manager: Optional[AccountManager] = None
alpaca_executor: Optional[AlpacaExecutor] = None
position_tracker: Optional[PositionTracker] = None

logger = logging.getLogger("oms")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, order_service, account_manager, alpaca_executor, position_tracker

    # Load env from project root
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    port = int(os.environ.get("PORT", settings.port))
    logger.info(f"OMS starting on {settings.host}:{port}")

    # Database (retry on startup for Cloud SQL socket readiness)
    logger.info(f"OMS: Connecting to database...")
    import asyncio
    engine = None
    for attempt in range(5):
        try:
            engine = await create_db_engine(settings.database_url)
            await init_db(engine)
            logger.info("OMS: Database connected")
            break
        except Exception as e:
            logger.warning(f"OMS: DB connection attempt {attempt+1}/5 failed: {e}")
            if attempt < 4:
                await asyncio.sleep(3)
            else:
                raise
    session_factory = get_session_factory(engine)

    # Services
    account_manager = AccountManager(session_factory, settings.oms_encryption_key)
    alpaca_executor = AlpacaExecutor(account_manager)
    risk_manager = RiskManager(session_factory)
    order_service = OrderService(
        session_factory=session_factory,
        account_manager=account_manager,
        alpaca_executor=alpaca_executor,
        risk_manager=risk_manager,
        webhook_secret=settings.oms_webhook_secret,
    )

    # Background tasks
    position_tracker = PositionTracker(
        session_factory=session_factory,
        alpaca_executor=alpaca_executor,
        position_sync_interval=settings.position_sync_interval,
        equity_snapshot_interval=settings.equity_snapshot_interval,
    )
    await position_tracker.start()

    logger.info("OMS ready")
    yield

    # Shutdown
    await position_tracker.stop()
    await close_engine()
    logger.info("OMS shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OMS — Order Management Service",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Rate limiting (webhook endpoint only)
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple token-bucket rate limiter for /webhook. 10 requests/minute per IP."""

    def __init__(self, app, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path != "/webhook" or request.method != "POST":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Prune old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self._window
        ]

        if len(self._requests[client_ip]) >= self._max:
            logger.warning(f"OMS: Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Max 10 requests per minute."},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


def _require_api_key(x_oms_api_key: str = Header(None)):
    if not settings or not settings.oms_api_key:
        return  # No API key configured = open access
    if x_oms_api_key != settings.oms_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Webhook endpoint (primary integration point)
# ---------------------------------------------------------------------------

@app.post("/webhook", response_model=WebhookResponse)
async def webhook(req: WebhookRequest):
    """Universal webhook — accepts orders from any source system."""
    if not order_service:
        raise HTTPException(503, "OMS not initialized")

    start = time.monotonic()
    resp = await order_service.handle_webhook(req)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Log webhook request/response
    try:
        from oms.db.engine import get_session_factory
        from oms.db.models import WebhookLog

        req_body = req.model_dump()
        req_body.pop("token", None)  # Don't log secrets

        log = WebhookLog(
            source=req.source,
            action=req.action,
            symbol=req.symbol or "",
            idempotency_key=req.idempotency_key,
            request_body=req_body,
            response_status=resp.status,
            response_reason=resp.reason,
            order_id=resp.order.id if resp.order else None,
            duration_ms=elapsed_ms,
        )
        sf = get_session_factory()
        async with sf() as session:
            session.add(log)
            await session.commit()
    except Exception as e:
        logger.warning(f"OMS: Failed to log webhook: {e}")

    return resp


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    accounts_status = []
    if account_manager:
        accounts = await account_manager.get_all_accounts()
        for acct in accounts:
            connected = await alpaca_executor.check_connectivity(acct)
            accounts_status.append({
                "name": acct.name,
                "connected": connected,
                "is_default": acct.is_default,
            })

    return HealthResponse(
        status="ok",
        database="connected",
        accounts=accounts_status,
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

@app.get("/orders", response_model=list[OrderResponse])
async def list_orders(
    x_oms_api_key: str = Header(None),
    account: str | None = Query(None),
    status: str | None = Query(None),
    source: str | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(50, le=200),
):
    _require_api_key(x_oms_api_key)
    if not order_service:
        raise HTTPException(503, "OMS not initialized")
    orders = await order_service.list_orders(
        account_name=account, status=status, source=source, symbol=symbol, limit=limit
    )
    return [OrderResponse.model_validate(o) for o in orders]


@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: UUID, x_oms_api_key: str = Header(None)):
    _require_api_key(x_oms_api_key)
    if not order_service:
        raise HTTPException(503, "OMS not initialized")

    from sqlalchemy import select
    from oms.db.engine import get_session_factory
    from oms.db.models import Order

    sf = get_session_factory()
    async with sf() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(404, "Order not found")
    return OrderResponse.model_validate(order)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@app.get("/positions")
async def list_positions(
    x_oms_api_key: str = Header(None),
    account: str | None = Query(None),
):
    """Current open positions (synced from Alpaca)."""
    _require_api_key(x_oms_api_key)
    if not position_tracker:
        raise HTTPException(503, "OMS not initialized")
    return await position_tracker.get_positions(account_name=account)


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@app.get("/trades")
async def list_trades(
    x_oms_api_key: str = Header(None),
    account: str | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(50, le=200),
):
    """Completed round-trip trades."""
    _require_api_key(x_oms_api_key)
    if not position_tracker:
        raise HTTPException(503, "OMS not initialized")
    return await position_tracker.get_trades(
        account_name=account, symbol=symbol, limit=limit
    )


# ---------------------------------------------------------------------------
# Webhook Logs
# ---------------------------------------------------------------------------

@app.get("/webhook-logs")
async def list_webhook_logs(
    x_oms_api_key: str = Header(None),
    source: str | None = Query(None),
    limit: int = Query(100, le=500),
):
    """Webhook request history."""
    _require_api_key(x_oms_api_key)
    from sqlalchemy import select
    from oms.db.models import WebhookLog

    sf = get_session_factory()
    async with sf() as session:
        query = select(WebhookLog).order_by(WebhookLog.received_at.desc()).limit(limit)
        if source:
            query = query.where(WebhookLog.source == source)
        result = await session.execute(query)
        logs = result.scalars().all()

    return [
        {
            "id": str(log.id),
            "received_at": log.received_at.isoformat(),
            "source": log.source,
            "action": log.action,
            "symbol": log.symbol,
            "idempotency_key": log.idempotency_key,
            "request_body": log.request_body,
            "response_status": log.response_status,
            "response_reason": log.response_reason,
            "order_id": str(log.order_id) if log.order_id else None,
            "duration_ms": log.duration_ms,
        }
        for log in logs
    ]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the OMS dashboard."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>OMS Dashboard</h1><p>dashboard.html not found</p>")


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

@app.get("/accounts", response_model=list[AccountResponse])
async def list_accounts(x_oms_api_key: str = Header(None)):
    _require_api_key(x_oms_api_key)
    if not account_manager:
        raise HTTPException(503, "OMS not initialized")
    accounts = await account_manager.get_all_accounts()
    return [AccountResponse.model_validate(a) for a in accounts]


@app.post("/accounts", response_model=AccountResponse, status_code=201)
async def create_account(body: AccountCreate, x_oms_api_key: str = Header(None)):
    _require_api_key(x_oms_api_key)
    if not account_manager:
        raise HTTPException(503, "OMS not initialized")

    try:
        account = await account_manager.create_account(
            name=body.name,
            api_key=body.api_key,
            secret_key=body.secret_key,
            base_url=body.base_url,
            is_default=body.is_default,
            allowed_sources=body.allowed_sources,
            risk_limits=body.risk_limits,
        )
        return AccountResponse.model_validate(account)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.patch("/accounts/{account_id}", response_model=AccountResponse)
async def update_account(account_id: UUID, body: AccountUpdate, x_oms_api_key: str = Header(None)):
    _require_api_key(x_oms_api_key)
    if not account_manager:
        raise HTTPException(503, "OMS not initialized")

    kwargs = body.model_dump(exclude_none=True)
    if not kwargs:
        raise HTTPException(400, "No fields to update")

    account = await account_manager.update_account(account_id, **kwargs)
    if not account:
        raise HTTPException(404, "Account not found")
    return AccountResponse.model_validate(account)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _settings = load_settings()
    # Cloud Run sets PORT env var; fall back to config
    port = int(os.environ.get("PORT", _settings.port))
    uvicorn.run(
        "oms.main:app",
        host=_settings.host,
        port=port,
        reload=False,
        log_level=_settings.log_level.lower(),
    )
