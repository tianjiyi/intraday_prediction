# Order Management Service (OMS) — Standalone Microservice

## Context
The DayX intraday strategy generates real-time signals but has no execution capability. We need an order management service that is **fully decoupled** from any strategy/signal logic. The OMS is a standalone microservice that:
- Receives order requests via **webhook** (the only integration point)
- Places orders on Alpaca paper trading accounts
- Tracks order lifecycle, positions, and performance
- Supports multiple source systems (platform, TradingView, anything with HTTP) and multiple Alpaca accounts

**Key principle**: Strategy/signal logic stays in the source system. OMS only knows how to execute orders and track results. Both the intraday platform and TradingView PineScript integrate via the same webhook interface.

## Architecture

```
┌─────────────────┐     POST /webhook     ┌───────────────┐     Orders API     ┌─────────────┐
│  Intraday       │ ──────────────────────>│               │ ─────────────────> │   Alpaca    │
│  Platform       │                        │     OMS       │                    │   Paper     │
└─────────────────┘                        │  Microservice │ <───────────────── │   Account(s)│
                                           │               │   Fills/Status     └─────────────┘
┌─────────────────┐     POST /webhook     │  (FastAPI)    │
│  TradingView    │ ──────────────────────>│               │──> PostgreSQL (oms db)
│  PineScript     │                        │  Port 8100    │    + TimescaleDB
└─────────────────┘                        └───────────────┘
                                                  ↑
┌─────────────────┐     POST /webhook            │
│  Any Future     │ ─────────────────────────────┘
│  System         │
└─────────────────┘
```

### Standalone FastAPI app
- Top-level directory: `oms/` (sibling to `platform/`, `research/`, etc.)
- Own `main.py`, `requirements.txt`, `Dockerfile`
- Runs on port 8100 (configurable)
- **PostgreSQL** — separate `oms` database on the existing PostgreSQL server (already running for `agent_memory`)
- **TimescaleDB** hypertable for `equity_snapshots` (time-series optimized queries)
- Can run independently via `python oms/main.py`
- Can be containerized separately

### Why standalone
- Clear separation: OMS knows nothing about DayX, CCI, VWAP, or any strategy
- Same webhook interface for all clients — platform, TradingView, future systems
- Can restart/deploy OMS without touching the trading platform
- Testable in isolation with curl/Postman
- Own database (`oms`) keeps order data separate from agent memory

## Directory Structure

```
oms/
  main.py                 # FastAPI app entry point
  config.py               # Settings (from env vars + config.yaml)
  requirements.txt        # Minimal deps: fastapi, uvicorn, alpaca-py, sqlalchemy
  Dockerfile              # Standalone container
  config.yaml             # OMS-specific config
  db/
    engine.py             # Async PostgreSQL engine + session factory (separate `oms` database)
    models.py             # ORM models (Account, Order, Position, Trade, EquitySnapshot)
    migrations.py         # Auto-create tables + TimescaleDB hypertable on startup
  services/
    order_service.py      # Core order lifecycle (validate → execute → track)
    alpaca_executor.py    # Alpaca TradingClient wrapper (one per account)
    account_manager.py    # Multi-account credential management
    risk_manager.py       # Pre-trade risk checks
    position_tracker.py   # Sync positions from Alpaca, build round-trip trades
    performance.py        # Metrics calculation (Sharpe, win rate, P&L)
  schemas.py              # Pydantic request/response models
```

## Universal Webhook Schema

**Single endpoint**: `POST /webhook`

All source systems send the same JSON:

```json
{
  "token": "shared-secret-here",
  "idempotency_key": "platform-DayX_1Min-QQQ-buy_dip-1711036800",
  "source": "platform",
  "account": "dayx-paper-1",
  "action": "open",

  "symbol": "QQQ",
  "side": "buy",
  "order_type": "limit",
  "limit_price": 485.50,
  "qty": 20,
  "time_in_force": "day",

  "bracket": {
    "stop_loss": 483.05,
    "take_profit": 489.20
  },

  "tags": {
    "strategy": "DayX_Optuna_1Min",
    "signal": "buy_dip",
    "timeframe": "1Min",
    "confidence": 0.82
  },

  "metadata": {
    "bar_time": "2026-03-16T10:30:00-04:00"
  }
}
```

### Field definitions

| Field | Required | Description |
|-------|----------|-------------|
| `token` | yes | Shared secret for auth (validated against `OMS_WEBHOOK_SECRET` env var) |
| `idempotency_key` | yes | Unique key to prevent duplicate orders. Convention: `{source}-{strategy}-{symbol}-{signal}-{unix_ts}` |
| `source` | yes | Identifier of the sending system (freeform string, stored for tracking) |
| `account` | no | Account name to route to. If omitted, uses default account. |
| `action` | yes | `open` (new position), `close` (close existing), `cancel` (cancel pending order) |
| `symbol` | yes | Ticker symbol |
| `side` | yes for open | `buy` or `sell` |
| `order_type` | no | `market` (default), `limit`, `stop`, `stop_limit` |
| `limit_price` | if limit | Limit price |
| `stop_price` | if stop | Stop trigger price |
| `qty` | no | Share quantity. If omitted, risk manager calculates from account equity + `max_position_pct` |
| `notional` | no | Dollar amount (alternative to qty). Alpaca supports fractional via notional. |
| `time_in_force` | no | `day` (default), `gtc`, `ioc`, `fok` |
| `bracket` | no | If present, submits Alpaca bracket order with stop_loss and/or take_profit legs |
| `bracket.stop_loss` | no | Stop loss price for bracket |
| `bracket.take_profit` | no | Take profit price for bracket |
| `tags` | no | Freeform key-value pairs stored with the order for filtering/reporting. Strategy, signal type, timeframe, etc. |
| `metadata` | no | Additional context stored but not used for execution |

### Close action
```json
{
  "token": "...",
  "idempotency_key": "platform-DayX_1Min-QQQ-close-1711040400",
  "source": "platform",
  "account": "dayx-paper-1",
  "action": "close",
  "symbol": "QQQ"
}
```
Closes the entire position in `symbol` via market order.

### Cancel action
```json
{
  "token": "...",
  "idempotency_key": "platform-cancel-order-abc123",
  "source": "platform",
  "action": "cancel",
  "order_id": "uuid-of-order-to-cancel"
}
```

## TradingView PineScript Integration

TradingView webhooks POST to `https://{oms-host}:8100/webhook`. The PineScript `alert_message` outputs our JSON directly:

```pine
if canEnter and sig_buy_dip
    strategy.entry("BuyDip", strategy.long)
    alert('{"token":"{{secret}}","idempotency_key":"tv-DayX1Min-' + syminfo.ticker + '-buy_dip-' + str.tostring(timenow) + '","source":"tradingview","account":"dayx-paper-1","action":"open","symbol":"' + syminfo.ticker + '","side":"buy","order_type":"limit","limit_price":' + str.tostring(close) + ',"qty":20,"bracket":{"stop_loss":' + str.tostring(stop_px) + ',"take_profit":' + str.tostring(t1_px) + '},"tags":{"strategy":"DayX_Optuna_1Min","signal":"buy_dip"}}', alert.freq_once_per_bar)
```

## Platform Integration

The intraday platform sends webhooks from its signal detection callback. In `platform/main.py`, after a signal is detected:

```python
# In websocket_callback, after signal_process_bar() returns a signal:
if signal and app_config.get("oms", {}).get("webhook_url"):
    import httpx
    payload = {
        "token": os.environ.get("OMS_WEBHOOK_SECRET"),
        "idempotency_key": f"platform-{signal['signal']}-{sym}-{signal['time']}",
        "source": "platform",
        "account": app_config["oms"].get("default_account", "default"),
        "action": "open",
        "symbol": sym,
        "side": "buy" if signal["direction"] == "long" else "sell",
        "order_type": "limit",
        "limit_price": signal["price"],
        "bracket": {
            "stop_loss": signal["stop"],
            "take_profit": signal["target1"],
        },
        "tags": {
            "strategy": f"DayX_{tf_str}",
            "signal": signal["signal"],
            "timeframe": tf_str,
        },
    }
    asyncio.create_task(httpx.AsyncClient().post(
        app_config["oms"]["webhook_url"], json=payload
    ))
```

## Database (PostgreSQL — separate `oms` database)

Uses the same PostgreSQL server as `agent_memory` but a separate database named `oms`. Reuses the existing connection pattern from `platform/db/engine.py` with OMS-specific env vars.

Env vars: `OMS_POSTGRES_DB=oms` (host/port/user/password shared with existing PG).

### accounts
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | default uuid_generate_v4() |
| name | VARCHAR(100) UNIQUE | e.g. "dayx-paper-1" |
| api_key_encrypted | TEXT | Fernet-encrypted |
| secret_key_encrypted | TEXT | Fernet-encrypted |
| base_url | VARCHAR(200) | Default: `https://paper-api.alpaca.markets` |
| is_active | BOOLEAN | |
| is_default | BOOLEAN | Used when webhook omits `account` |
| allowed_sources | JSONB | `["platform", "tradingview"]` or empty = all |
| risk_limits | JSONB | `{"max_daily_loss": 500, "max_positions": 3, "max_position_pct": 0.25}` |
| created_at | TIMESTAMPTZ | |

### orders
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| idempotency_key | VARCHAR(200) UNIQUE | |
| account_id | UUID FK → accounts | |
| alpaca_order_id | VARCHAR(100) | Alpaca's order ID |
| symbol | VARCHAR(20) | |
| side | VARCHAR(10) | buy/sell |
| action | VARCHAR(10) | open/close/cancel |
| order_type | VARCHAR(20) | market/limit/stop/stop_limit |
| qty | NUMERIC(12,4) | |
| notional | NUMERIC(12,2) | |
| limit_price | NUMERIC(12,4) | |
| stop_price | NUMERIC(12,4) | |
| filled_qty | NUMERIC(12,4) | Default 0 |
| filled_avg_price | NUMERIC(12,4) | |
| time_in_force | VARCHAR(10) | |
| status | VARCHAR(20) | pending/submitted/accepted/filled/partial_fill/canceled/rejected |
| reject_reason | TEXT | |
| source | VARCHAR(50) | |
| tags | JSONB | Strategy, signal, timeframe, etc. |
| bracket_config | JSONB | Stop loss / take profit prices |
| parent_order_id | UUID FK (self) | For bracket legs |
| metadata | JSONB | |
| created_at | TIMESTAMPTZ | |
| submitted_at | TIMESTAMPTZ | |
| filled_at | TIMESTAMPTZ | |
| **Indexes** | | (account_id, status), (symbol, created_at), (source, created_at) |

### positions (synced from Alpaca)
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| account_id | UUID FK | |
| symbol | VARCHAR(20) | |
| side | VARCHAR(10) | long/short |
| qty | NUMERIC(12,4) | |
| entry_price | NUMERIC(12,4) | avg_entry_price from Alpaca |
| current_price | NUMERIC(12,4) | |
| unrealized_pnl | NUMERIC(12,2) | |
| market_value | NUMERIC(14,2) | |
| entry_order_id | UUID FK | |
| tags | JSONB | Inherited from entry order |
| opened_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |
| **UNIQUE** | (account_id, symbol) | |

### trades (completed round-trips)
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| account_id | UUID FK | |
| symbol | VARCHAR(20) | |
| side | VARCHAR(10) | |
| qty | NUMERIC(12,4) | |
| entry_price | NUMERIC(12,4) | |
| exit_price | NUMERIC(12,4) | |
| realized_pnl | NUMERIC(12,2) | |
| realized_pnl_pct | NUMERIC(8,4) | |
| entry_order_id | UUID FK | |
| exit_order_id | UUID FK | |
| exit_reason | VARCHAR(50) | stop_loss/take_profit/manual/eod_flatten |
| tags | JSONB | |
| hold_duration_seconds | INTEGER | |
| opened_at | TIMESTAMPTZ | |
| closed_at | TIMESTAMPTZ | |
| **Indexes** | | (account_id, closed_at), (closed_at) for tags->>strategy queries |

### equity_snapshots (TimescaleDB hypertable)
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| account_id | UUID FK | |
| equity | NUMERIC(14,2) | |
| cash | NUMERIC(14,2) | |
| market_value | NUMERIC(14,2) | |
| daily_pnl | NUMERIC(12,2) | |
| snapshot_at | TIMESTAMPTZ | Hypertable partition column |

On startup: `SELECT create_hypertable('equity_snapshots', 'snapshot_at', if_not_exists => TRUE)` — enables TimescaleDB time-series compression and fast range queries for equity curves. Falls back gracefully to regular table if TimescaleDB extension not available.

## API Endpoints

### Webhook (the primary integration point)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/webhook` | Universal order webhook — open, close, or cancel |

### Management & Monitoring (for dashboard/debugging)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health + account connectivity |
| GET | `/orders` | List orders (filters: account, status, source, symbol, tags, date range) |
| GET | `/orders/{id}` | Single order detail |
| GET | `/positions` | Current open positions across accounts |
| GET | `/trades` | Completed round-trips (filters: account, tags.strategy, date range) |
| GET | `/accounts` | Account summaries (equity, open positions, daily P&L) |
| POST | `/accounts` | Add Alpaca paper account |
| PATCH | `/accounts/{id}` | Update account settings/risk limits |
| GET | `/performance` | Aggregate metrics (Sharpe, win rate, P&L by strategy/account/source) |
| GET | `/performance/{account_id}` | Per-account performance |
| GET | `/equity-curve/{account_id}` | Equity time-series |

Auth: All endpoints require `X-OMS-API-Key` header (except `/webhook` which uses `token` in body, and `/health` which is unauthenticated).

## Risk Manager (pre-trade checks)

Every webhook order passes through before execution:
1. **Token validation** — reject if token doesn't match `OMS_WEBHOOK_SECRET`
2. **Idempotency** — if `idempotency_key` exists and order isn't rejected/canceled, return existing order
3. **Account resolution** — find account by `account` name (or default). Must be `is_active`.
4. **Source allowed** — if account has `allowed_sources`, check `source` is in list
5. **Daily loss limit** — sum today's realized P&L vs `risk_limits.max_daily_loss`
6. **Max positions** — count open positions vs `risk_limits.max_positions`
7. **Position sizing** — if `qty` omitted, calculate from `max_position_pct * equity / price`
8. **Market hours** — warn (log) if outside 09:30-16:00 ET but don't block (Alpaca paper allows extended hours)

Rejections stored as orders with `status=rejected` and `reject_reason`.

## Config

`oms/config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8100

position_sync_interval: 30    # seconds
equity_snapshot_interval: 300  # seconds (5 min)

logging:
  level: INFO
```

Environment variables (`.env`):
```
# OMS Database (same PG server as agent_memory, separate DB)
OMS_POSTGRES_DB=oms
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=kronos
POSTGRES_PASSWORD=kronos_dev

# OMS Auth
OMS_WEBHOOK_SECRET=your_webhook_secret
OMS_API_KEY=your_dashboard_api_key
OMS_ENCRYPTION_KEY=your_fernet_key_for_account_credentials
```

## Key Files to Reference

- `platform/services/trade_context_service.py` — pattern for Alpaca client init with graceful degradation
- `platform/services/signal_service.py:222` — `_row_to_signal()` output dict that platform will convert to webhook payload
- `platform/main.py:288` — signal detection callback where platform webhook POST will be added
- `platform/services/websocket_manager.py` — existing Alpaca credential patterns (`ALPACA_KEY_ID`, etc.)
- `research/backtesting/dayx/strategy.py` — reference for position management concepts (bracket, trailing, EOD flatten)

## Implementation Phases

### Phase 1 — Scaffold + Webhook + Single Order
- Create `oms/` directory structure
- Pydantic schemas (`schemas.py`)
- PostgreSQL DB engine (async, separate `oms` database) + models + auto-create tables on startup
- TimescaleDB hypertable for equity_snapshots (graceful fallback if extension not installed)
- Account manager (add account, encrypt credentials)
- Alpaca executor (submit market/limit order)
- `POST /webhook` — validate → risk check → execute → store
- `GET /health`, `GET /orders`
- Test: curl a webhook → order appears on Alpaca paper dashboard

### Phase 2 — Bracket Orders + Position Sync
- Bracket order support (Alpaca `OrderClass.BRACKET`)
- Background task: poll Alpaca positions every 30s → sync to `positions` table
- Background task: poll Alpaca account every 5min → `equity_snapshots`
- `GET /positions`
- Handle close action (market sell entire position)

### Phase 3 — Round-Trip Trades + Risk
- Detect position close → create `trades` record (entry + exit = round-trip)
- Full risk manager (daily loss, max positions, position sizing)
- Multi-account support (route by `account` field)
- `POST /accounts`, `PATCH /accounts/{id}`, `GET /accounts`

### Phase 4 — Performance + Platform Integration
- Performance calculator (win rate, avg W/L, Sharpe, drawdown, per-strategy)
- `GET /performance`, `GET /equity-curve/{account_id}`, `GET /trades`
- Add webhook POST to platform's signal callback in `main.py`
- Test end-to-end: platform signal → webhook → Alpaca order

### Phase 5 — TradingView + Polish
- Document PineScript alert_message templates
- Test TradingView webhook → Alpaca order flow
- Dockerfile for standalone deployment
- Add to `docker-compose.yml` as separate service

## Verification

1. `python oms/main.py` — starts on port 8100, creates tables in `oms` PostgreSQL DB
2. `POST /accounts` — add Alpaca paper account credentials
3. `curl -X POST http://localhost:8100/webhook -d '{"token":"...","action":"open","symbol":"QQQ","side":"buy","order_type":"market","qty":1}'` → order placed on Alpaca
4. `GET /orders` → see order with status=filled
5. `GET /positions` → see QQQ position
6. Webhook with `action=close` → position closed, trade recorded
7. `GET /trades` → round-trip with realized P&L
8. `GET /performance` → win rate, Sharpe calculated
9. Platform signal fires → webhook sent → order placed (end-to-end)
10. TradingView alert → webhook received → order placed
