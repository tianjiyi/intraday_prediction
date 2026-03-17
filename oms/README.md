# OMS — Order Management Service

Standalone FastAPI microservice for order execution, position tracking, and risk management on Alpaca Markets.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Cloud Run                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  FastAPI (uvicorn)                       │ │
│  │                                                         │ │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │ Webhook  │  │  Order       │  │  Position        │  │ │
│  │  │ Endpoint │─▶│  Service     │  │  Tracker (bg)    │  │ │
│  │  └──────────┘  │              │  │  - sync 30s      │  │ │
│  │                │  ┌─────────┐ │  │  - snapshot 5m   │  │ │
│  │  ┌──────────┐  │  │  Risk   │ │  └────────┬─────────┘  │ │
│  │  │ REST API │  │  │ Manager │ │           │             │ │
│  │  │ /orders  │  │  └─────────┘ │           │             │ │
│  │  │ /trades  │  └──────┬───────┘           │             │ │
│  │  │ /accts   │         │                   │             │ │
│  │  └──────────┘         ▼                   ▼             │ │
│  │               ┌───────────────┐  ┌──────────────────┐   │ │
│  │               │   Alpaca      │  │  Account         │   │ │
│  │               │   Executor    │  │  Manager         │   │ │
│  │               │   (cached     │  │  (Fernet encrypt)│   │ │
│  │               │    clients)   │  └──────────────────┘   │ │
│  │               └───────┬───────┘                         │ │
│  └───────────────────────┼─────────────────────────────────┘ │
│                          │                                    │
└──────────────────────────┼────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌───────────┐  ┌───────────┐  ┌───────────┐
     │  Alpaca   │  │  Cloud    │  │ Dashboard │
     │  Paper /  │  │  SQL      │  │  (HTML)   │
     │  Live API │  │  Postgres │  │  :8080/   │
     └───────────┘  └───────────┘  └───────────┘
```

## Deployment (GCP Cloud Run + Cloud SQL)

**Live URL**: `https://oms-481888403086.us-east1.run.app`

| Component       | Service                                             | Region   |
|-----------------|-----------------------------------------------------|----------|
| App             | Cloud Run `oms`                                     | us-east1 |
| Database        | Cloud SQL PostgreSQL 16 `oms-db` (db-f1-micro)      | us-east1 |
| Container Image | `us-east1-docker.pkg.dev/intraday-osm/oms/oms:latest` | us-east1 |
| GCP Project     | `intraday-osm`                                      | —        |

### Environment Variables (set on Cloud Run)

| Variable             | Purpose                                |
|----------------------|----------------------------------------|
| `POSTGRES_HOST`      | Cloud SQL public IP                    |
| `POSTGRES_PORT`      | 5432                                   |
| `POSTGRES_USER`      | postgres                               |
| `POSTGRES_PASSWORD`  | Database password                      |
| `OMS_POSTGRES_DB`    | oms                                    |
| `OMS_WEBHOOK_SECRET` | Token for webhook authentication       |
| `OMS_API_KEY`        | API key for protected endpoints        |
| `OMS_ENCRYPTION_KEY` | Fernet key for credential encryption   |
| `PORT`               | 8080 (Cloud Run default)               |

### Deploy Commands

```bash
# Build & push container
gcloud builds submit --tag us-east1-docker.pkg.dev/intraday-osm/oms/oms:latest \
  --project=intraday-osm

# Deploy to Cloud Run
gcloud run deploy oms \
  --image=us-east1-docker.pkg.dev/intraday-osm/oms/oms:latest \
  --region=us-east1 --project=intraday-osm
```

### Local Development

```bash
# Activate venv from project root
. venv/Scripts/activate

# Start with local Postgres
cd oms && python main.py
# → http://localhost:8100
```

---

## API Endpoints

### Authentication

Protected endpoints require `X-OMS-API-Key` header. If `OMS_API_KEY` is not set, all endpoints are open.

The webhook endpoint uses a `token` field in the request body (checked against `OMS_WEBHOOK_SECRET`).

### Public Endpoints

#### `GET /health`
Service health check. No auth required.

**Response**:
```json
{
  "status": "ok",
  "database": "connected",
  "accounts": [
    { "name": "ai-signals-paper", "connected": true, "is_default": true }
  ]
}
```

#### `GET /`
Dashboard (HTML). No auth required.

---

### Webhook

#### `POST /webhook`
Primary integration point for TradingView, platform signals, or manual orders.

**Rate limited**: 10 requests/minute per IP.

**Request**:
```json
{
  "token": "your_webhook_secret",
  "idempotency_key": "unique-key-123",
  "source": "tradingview",
  "account": null,
  "action": "open",
  "symbol": "AAPL",
  "side": "buy",
  "order_type": "market",
  "qty": 10,
  "time_in_force": "day",
  "bracket": {
    "stop_loss": 145.00,
    "take_profit": 160.00
  },
  "tags": { "strategy": "momentum" }
}
```

| Field             | Type                                   | Required | Default   |
|-------------------|----------------------------------------|----------|-----------|
| `token`           | string                                 | Yes      | —         |
| `idempotency_key` | string                                 | Yes      | —         |
| `source`          | string                                 | Yes      | —         |
| `account`         | string \| null                         | No       | default   |
| `action`          | `"open"` \| `"close"` \| `"cancel"`   | Yes      | —         |
| `symbol`          | string                                 | Yes*     | —         |
| `side`            | `"buy"` \| `"sell"`                    | Yes*     | —         |
| `order_type`      | `"market"` \| `"limit"` \| `"stop"` \| `"stop_limit"` | No | `"market"` |
| `qty`             | float                                  | No†      | —         |
| `notional`        | float                                  | No†      | —         |
| `limit_price`     | float                                  | No       | —         |
| `stop_price`      | float                                  | No       | —         |
| `time_in_force`   | `"day"` \| `"gtc"` \| `"ioc"` \| `"fok"` | No    | `"day"`   |
| `bracket`         | object                                 | No       | —         |
| `tags`            | object                                 | No       | —         |
| `order_id`        | string                                 | No‡      | —         |

\* Required for `action=open`
† One of `qty` or `notional` required for `action=open`
‡ Required for `action=cancel`

**Response**:
```json
{
  "status": "accepted",
  "order": {
    "id": "uuid",
    "idempotency_key": "unique-key-123",
    "account_id": "uuid",
    "alpaca_order_id": "alpaca-uuid",
    "symbol": "AAPL",
    "side": "buy",
    "action": "open",
    "order_type": "market",
    "qty": 10.0,
    "status": "submitted",
    "source": "tradingview",
    "created_at": "2026-03-17T10:30:00Z"
  },
  "reason": null
}
```

**Status values**: `accepted`, `rejected`, `duplicate`

**Processing pipeline**:
1. Validate token against `OMS_WEBHOOK_SECRET`
2. Check idempotency (return existing if duplicate)
3. Resolve account by name or use default
4. Run risk checks (source whitelist, daily loss limit, max positions)
5. Submit to Alpaca
6. Log to `webhook_logs` table

---

### Orders (API-key protected)

#### `GET /orders`
List orders with optional filters.

| Query Param | Type   | Default | Max |
|-------------|--------|---------|-----|
| `account`   | string | all     | —   |
| `status`    | string | all     | —   |
| `source`    | string | all     | —   |
| `symbol`    | string | all     | —   |
| `limit`     | int    | 50      | 200 |

#### `GET /orders/{order_id}`
Get single order by UUID.

---

### Positions (API-key protected)

#### `GET /positions`
Current open positions (synced from Alpaca every 30s).

| Query Param | Type   | Default |
|-------------|--------|---------|
| `account`   | string | all     |

**Response**:
```json
[
  {
    "id": "uuid",
    "account_id": "uuid",
    "symbol": "AAPL",
    "side": "long",
    "qty": 10.0,
    "entry_price": 150.25,
    "current_price": 151.00,
    "unrealized_pnl": 7.50,
    "market_value": 1510.00,
    "opened_at": "2026-03-17T10:30:00Z"
  }
]
```

---

### Trades (API-key protected)

#### `GET /trades`
Completed round-trip trades (auto-recorded when positions close).

| Query Param | Type   | Default | Max |
|-------------|--------|---------|-----|
| `account`   | string | all     | —   |
| `symbol`    | string | all     | —   |
| `limit`     | int    | 50      | 200 |

**Response**:
```json
[
  {
    "id": "uuid",
    "symbol": "AAPL",
    "side": "long",
    "qty": 10.0,
    "entry_price": 150.25,
    "exit_price": 155.00,
    "realized_pnl": 47.50,
    "realized_pnl_pct": 0.0316,
    "exit_reason": "manual",
    "hold_duration_seconds": 7200,
    "opened_at": "2026-03-17T10:30:00Z",
    "closed_at": "2026-03-17T12:30:00Z"
  }
]
```

---

### Webhook Logs (API-key protected)

#### `GET /webhook-logs`
Webhook request/response audit trail.

| Query Param | Type   | Default | Max |
|-------------|--------|---------|-----|
| `source`    | string | all     | —   |
| `limit`     | int    | 100     | 500 |

---

### Accounts (API-key protected)

#### `GET /accounts`
List all active accounts.

#### `POST /accounts`
Create a new trading account.

```json
{
  "name": "ai-signals-paper",
  "api_key": "ALPACA_KEY",
  "secret_key": "ALPACA_SECRET",
  "base_url": "https://paper-api.alpaca.markets",
  "is_default": true,
  "allowed_sources": ["tradingview", "platform"],
  "risk_limits": {
    "max_position_pct": 25,
    "max_daily_loss_pct": 5
  }
}
```

#### `PATCH /accounts/{account_id}`
Update account settings (is_active, is_default, allowed_sources, risk_limits).

---

## Database Schema

**Engine**: PostgreSQL 16 (Cloud SQL) with asyncpg driver.

### Tables

#### `accounts`
| Column                | Type          | Notes                        |
|-----------------------|---------------|------------------------------|
| `id`                  | UUID (PK)     | uuid4                        |
| `name`                | VARCHAR(100)  | UNIQUE                       |
| `api_key_encrypted`   | TEXT          | Fernet-encrypted             |
| `secret_key_encrypted`| TEXT          | Fernet-encrypted             |
| `base_url`            | VARCHAR(200)  |                              |
| `is_active`           | BOOLEAN       | default true                 |
| `is_default`          | BOOLEAN       | default false                |
| `allowed_sources`     | JSONB         | [] = all sources allowed     |
| `risk_limits`         | JSONB         | {max_daily_loss, max_positions} |
| `created_at`          | TIMESTAMPTZ   | UTC                          |

#### `orders`
| Column            | Type          | Notes                              |
|-------------------|---------------|-------------------------------------|
| `id`              | UUID (PK)     |                                     |
| `idempotency_key` | VARCHAR(200)  | UNIQUE                              |
| `account_id`      | UUID (FK)     | → accounts.id                       |
| `alpaca_order_id`  | VARCHAR(100) | Alpaca-assigned ID                  |
| `symbol`          | VARCHAR(20)   |                                     |
| `side`            | VARCHAR(10)   | buy / sell                          |
| `action`          | VARCHAR(10)   | open / close / cancel               |
| `order_type`      | VARCHAR(20)   | market / limit / stop / stop_limit  |
| `qty`             | NUMERIC(12,4) |                                     |
| `notional`        | NUMERIC(12,2) |                                     |
| `limit_price`     | NUMERIC(12,4) |                                     |
| `stop_price`      | NUMERIC(12,4) |                                     |
| `filled_qty`      | NUMERIC(12,4) | default 0                           |
| `filled_avg_price` | NUMERIC(12,4)|                                     |
| `time_in_force`   | VARCHAR(10)   |                                     |
| `status`          | VARCHAR(20)   | pending / submitted / filled / rejected / canceled |
| `reject_reason`   | TEXT          |                                     |
| `source`          | VARCHAR(50)   |                                     |
| `tags`            | JSONB         |                                     |
| `bracket_config`  | JSONB         |                                     |
| `parent_order_id` | UUID (FK)     | → orders.id                         |
| `metadata`        | JSONB         |                                     |
| `created_at`      | TIMESTAMPTZ   |                                     |
| `submitted_at`    | TIMESTAMPTZ   |                                     |
| `filled_at`       | TIMESTAMPTZ   |                                     |

**Indexes**: (account_id, status), (symbol, created_at), (source, created_at)

#### `positions`
| Column          | Type          | Notes                    |
|-----------------|---------------|--------------------------|
| `id`            | UUID (PK)     |                          |
| `account_id`    | UUID (FK)     | → accounts.id            |
| `symbol`        | VARCHAR(20)   | UNIQUE with account_id   |
| `side`          | VARCHAR(10)   | long / short             |
| `qty`           | NUMERIC(12,4) |                          |
| `entry_price`   | NUMERIC(12,4) |                          |
| `current_price` | NUMERIC(12,4) |                          |
| `unrealized_pnl`| NUMERIC(12,2) |                          |
| `market_value`  | NUMERIC(14,2) |                          |
| `entry_order_id`| UUID (FK)     | → orders.id              |
| `tags`          | JSONB         |                          |
| `opened_at`     | TIMESTAMPTZ   |                          |
| `updated_at`    | TIMESTAMPTZ   |                          |

#### `trades`
| Column                 | Type          | Notes               |
|------------------------|---------------|----------------------|
| `id`                   | UUID (PK)     |                      |
| `account_id`           | UUID (FK)     | → accounts.id        |
| `symbol`               | VARCHAR(20)   |                      |
| `side`                 | VARCHAR(10)   | long / short         |
| `qty`                  | NUMERIC(12,4) |                      |
| `entry_price`          | NUMERIC(12,4) |                      |
| `exit_price`           | NUMERIC(12,4) |                      |
| `realized_pnl`         | NUMERIC(12,2) |                      |
| `realized_pnl_pct`     | NUMERIC(8,4)  |                      |
| `entry_order_id`       | UUID (FK)     | → orders.id          |
| `exit_order_id`        | UUID (FK)     | → orders.id          |
| `exit_reason`          | VARCHAR(50)   | manual / tp / sl / unknown |
| `tags`                 | JSONB         |                      |
| `hold_duration_seconds`| INTEGER       |                      |
| `opened_at`            | TIMESTAMPTZ   |                      |
| `closed_at`            | TIMESTAMPTZ   |                      |

**Indexes**: (account_id, closed_at), (closed_at)

#### `webhook_logs`
| Column            | Type          | Notes                   |
|-------------------|---------------|-------------------------|
| `id`              | UUID (PK)     |                         |
| `received_at`     | TIMESTAMPTZ   | indexed                 |
| `source`          | VARCHAR(50)   |                         |
| `action`          | VARCHAR(10)   |                         |
| `symbol`          | VARCHAR(20)   |                         |
| `idempotency_key` | VARCHAR(200)  |                         |
| `request_body`    | JSONB         | token stripped          |
| `response_status` | VARCHAR(20)   |                         |
| `response_reason` | TEXT          |                         |
| `order_id`        | UUID          |                         |
| `duration_ms`     | INTEGER       |                         |

#### `equity_snapshots`
| Column        | Type          | Notes                              |
|---------------|---------------|------------------------------------|
| `snapshot_at` | TIMESTAMPTZ   | Composite PK (TimescaleDB partition)|
| `id`          | UUID          | Composite PK                       |
| `account_id`  | UUID (FK)     | → accounts.id                      |
| `equity`      | NUMERIC(14,2) |                                    |
| `cash`        | NUMERIC(14,2) |                                    |
| `market_value`| NUMERIC(14,2) |                                    |
| `daily_pnl`   | NUMERIC(12,2) |                                    |

**Index**: (account_id, snapshot_at). Uses TimescaleDB hypertable if available.

---

## Services

### OrderService (`services/order_service.py`)
Core order lifecycle: validate → risk check → execute → record.

### AccountManager (`services/account_manager.py`)
Multi-account credential management with Fernet encryption. Encrypts API keys at rest, decrypts on demand.

### AlpacaExecutor (`services/alpaca_executor.py`)
Alpaca TradingClient wrapper. Caches clients per account. Supports market/limit/stop/stop_limit/bracket orders.

### RiskManager (`services/risk_manager.py`)
Pre-trade risk checks: idempotency, source whitelist, daily loss limit, max open positions.

### PositionTracker (`services/position_tracker.py`)
Background async tasks:
- **Position sync** (every 30s): polls Alpaca, updates `positions` table, creates `trades` records when positions close.
- **Equity snapshots** (every 5m): records account equity/cash/market_value to `equity_snapshots`.

---

## Project Structure

```
oms/
├── main.py                     # FastAPI app, endpoints, lifespan, middleware
├── config.py                   # Settings from env vars + config.yaml
├── config.yaml                 # Non-secret configuration
├── schemas.py                  # Pydantic request/response models
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image (python:3.12-slim)
├── .dockerignore               # Build exclusions
├── dashboard.html              # Web dashboard (vanilla HTML/CSS/JS)
├── db/
│   ├── __init__.py
│   ├── engine.py               # Async engine, session factory, init_db
│   └── models.py               # SQLAlchemy ORM models
└── services/
    ├── __init__.py
    ├── account_manager.py      # Credential encryption & account CRUD
    ├── alpaca_executor.py      # Alpaca API client wrapper
    ├── order_service.py        # Webhook handling & order lifecycle
    ├── position_tracker.py     # Background sync & snapshots
    └── risk_manager.py         # Pre-trade risk checks
```

---

## TradingView Integration

**Webhook URL**: `https://oms-481888403086.us-east1.run.app/webhook`

Example TradingView alert message (JSON):
```json
{
  "token": "your_webhook_secret",
  "idempotency_key": "tv-{{timenow}}-{{ticker}}",
  "source": "tradingview",
  "action": "open",
  "symbol": "{{ticker}}",
  "side": "buy",
  "qty": 10,
  "order_type": "market",
  "time_in_force": "day",
  "tags": { "strategy": "{{strategy.order.comment}}" }
}
```
