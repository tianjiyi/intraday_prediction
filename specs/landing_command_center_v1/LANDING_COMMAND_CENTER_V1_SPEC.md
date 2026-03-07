# Landing Command Center V1 Spec

Version: 1.0
Date: 2026-03-07
Status: Approved for implementation
Owner: Product/Trading

## 1. Objective
Replace current dashboard landing page with a day-trading command center that surfaces the most important factors affecting intraday decisions.

Primary goals:
- Immediate awareness of risk regime and sentiment
- Fast visibility into next 72h calendar/event risk
- Clear view of movers/losers and active market themes
- TradingView-like news flow for rapid headline triage
- Actionable state for chart + AI assistant handoff

## 2. Scope
In scope:
- Replace current `/` landing page UX
- Add command-center components and data wiring
- Reuse existing news monitor + scoring services where possible
- Add APIs for market pulse, events, movers, themes
- Integrate with existing chart page and AI chat context

Out of scope (V1):
- Auto-trading execution
- Portfolio/order management UI
- Complex personalization/profile management

## 3. Information Architecture
The new landing page is composed of these sections, top-to-bottom:

1. Market Pulse (hero strip)
2. Macro Tape (core market instruments)
3. Catalyst Clock (next 72h)
4. Movers and Losers (dual table)
5. Hot Themes (topic cards)
6. News Flow (TradingView-style: list + detail + watchlist)
7. Trade Context Snapshot

## 4. UX Requirements

### 4.1 Market Pulse
Display:
- Risk mode: `risk_on | risk_off | mixed`
- Aggregate sentiment score (-10 to +10)
- Volatility state (normal/elevated/high)
- 1-line change summary: "What changed in last 60 minutes"

Behavior:
- Updates every 60s
- Color coding:
  - green for risk_on
  - red for risk_off
  - amber for mixed/event-risk

### 4.1.1 Market Pulse calculation rules (V1)

Volatility state:
- Primary symbol source: `VIX` (if available from provider)
- Fallback symbol priority: `VIXY` -> `UVXY`
- Classification:
  - `high`: level >= 22 OR 1-day pct change >= +8%
  - `elevated`: 16 <= level < 22 OR 1-day pct change >= +4%
  - `normal`: otherwise

Sentiment score (`-10` to `+10`):
- `sentiment = 0.7 * news_sentiment + 0.3 * breadth_sentiment`
- `news_sentiment`: normalized from scored-news bullish vs bearish imbalance
- `breadth_sentiment`: normalized from advance/decline ratio of configured universe

Risk mode score (`-100` to `+100`):
- `risk_score = 100 * (0.35*vol_component + 0.20*macro_component + 0.25*breadth_component + 0.20*news_component)`
- Map to label:
  - `risk_on` if score >= +20
  - `risk_off` if score <= -20
  - `mixed` otherwise

Macro component proxies (Alpaca-friendly):
- Dollar proxy: `UUP`
- Rates proxy: `TLT` (or `IEF`)
- Equity breadth: `SPY`, `QQQ`, plus configured watchlist

### 4.2 Catalyst Clock (72h)
Display:
- Events from now to +72h
- Time, name, severity, expected affected sectors
- Countdown timer for each event

Required event classes:
- Macro calendar (CPI, PPI, NFP, FOMC, Fed speakers)
- Major earnings
- Geopolitical high-impact developments (when available)

Behavior:
- Default filter: `severity >= medium`
- Explicit "High Impact" tag for event-risk windows

### 4.2.1 Catalyst data sources (V1) — IMPLEMENTED

Alpaca note:
- Alpaca supports market session calendar/clock and news, but does not provide a full macro + earnings catalyst calendar for CPI/FOMC/NFP + broad earnings scheduling.

Provider (implemented):
- **Primary: `Benzinga` economics calendar** (`GET /api/v2.1/calendar/economics`)
  - Provides CPI, PPI, NFP, PCE, FOMC, Fed speakers, Jobless Claims, Michigan Sentiment, etc.
  - Auth: API key via `token` query parameter
  - Filter by: `date_from`, `date_to`, `importance` (1-3), `country` (3-letter code)
  - Benzinga importance mapping: 3→high, 2→medium, 1→low
  - Times returned in US/Eastern, normalized to UTC by `CatalystCalendarService`
- Benzinga earnings calendar (`/api/v2.1/calendar/earnings`) exists but returns empty on current plan — deferred to V2.
- TradingEconomics and FMP remain as future alternatives if Benzinga is dropped.

Fallback behavior (implemented):
- If Benzinga API is unavailable or `BENZINGA_API_KEY` is not set, the service returns inferred catalyst candidates from the news monitor's scored buffer.
- Inferred catalysts are detected by keyword matching (CPI, PPI, NFP, FOMC, Fed rate, PCE, GDP, jobless claims, earnings) against recent headlines.
- When both sources return data, events are merged with `source = mixed`.
- UI shows `inferred` badge on news-derived events.

Normalization (implemented):
- All events mapped into `CatalystEvent`:
  - `id`, `type`, `title`, `time` (UTC ISO), `impact`, `detail`, `source`, `category`, `countdown_seconds`, `consensus`, `prior`, `actual`
- Impact mapping: Benzinga importance 3→high, 2→medium, 1→low
- Detail string built from: period, prior value, consensus value, actual value
- Countdown seconds computed live from event time vs current UTC

Implementation files:
- Backend: `platform/services/catalyst_calendar_service.py`
- Config: `platform/config.yaml` → `landing.catalyst.*`
- Env var: `BENZINGA_API_KEY`
- Frontend component: `platform/frontend/src/components/landing/CatalystClock.tsx`
- Frontend types: `platform/frontend/src/types/landing.ts` → `CatalystEvent`, `CatalystClockResponse`

### 4.2 Macro Tape (core instruments)
Display a compact tape for:
- `VIX` (fallback `VIXY` if unavailable)
- `Gold` proxy: `GLD`
- `Oil` proxy: `USO`
- `SPY`
- `QQQ`
- `IWM`

Per item fields:
- label
- actual symbol used
- last price
- 1-day percent change
- source timestamp

Behavior:
- Refresh every 30-60 seconds
- Green/red color by pct change
- Click on equity ETFs (`SPY`, `QQQ`, `IWM`) opens chart page
- Show fallback marker when VIX proxy is used

### 4.3 Movers / Losers
Display:
- Two compact lists:
  - Top gainers
  - Top losers
- Columns: symbol, % move, relative volume, sector

Constraints:
- Liquid symbols only
- Configurable symbol universe (default: US large/mid cap + watchlist)

Interactions:
- Clicking symbol navigates to `/chart/:symbol`

### 4.4 Hot Themes
Display cards for active topics (e.g., HBM/Memory, Crude Oil/War, AI infra).

Each card includes:
- Theme name
- Momentum score
- Sentiment
- Top linked symbols
- "Active" status in last 24h

Behavior:
- Rank by weighted news impact + mention velocity + breadth
- Keep to top 4-6 themes

### 4.5 News Flow (TradingView style)
Layout:
- Left: feed list
- Center: selected article detail
- Right: watchlist / quick stats

Feed item fields:
- timestamp
- source
- headline
- impact tier
- symbols
- theme tags

Default mode:
- `Critical only` enabled on page load

Saved views:
- My Pick
- Macro
- Geopolitics
- Semiconductors

### 4.6 Trade Context Snapshot
Display:
- Intraday regime: trend/range/volatile
- Day trading VWAP state (above/below, sigma position)
- Nearest S/R zones
- Event-risk warning (if high-impact event near)

Purpose:
- Provide immediate context before moving to chart or asking AI

## 5. Backend API Contracts

### 5.1 GET /api/landing/market-pulse
Response:
```json
{
  "risk_mode": "risk_off",
  "risk_score": -34.7,
  "sentiment_score": -4.2,
  "volatility_state": "elevated",
  "volatility_source": "VIXY",
  "volatility_level": 21.4,
  "volatility_change_1d_pct": 6.9,
  "components": {
    "vol_component": -0.62,
    "macro_component": -0.31,
    "breadth_component": -0.12,
    "news_component": -0.44
  },
  "proxies": {
    "dollar_symbol": "UUP",
    "rates_symbol": "TLT",
    "breadth_universe_size": 8
  },
  "change_summary": "Oil +2.1%, VIX +1.8, war headlines accelerating",
  "updated_at": "2026-03-07T18:00:00Z"
}
```

### 5.2 GET /api/landing/catalyst-clock?hours=72
Response (implemented):
```json
{
  "window_hours": 72,
  "events": [
    {
      "id": "bz_68ba91852cf5150001d6ea58",
      "type": "economic",
      "title": "CPI (YoY)",
      "time": "2026-03-11T12:30:00+00:00",
      "impact": "high",
      "detail": "Period: Feb | Prior: 2.400%",
      "source": "benzinga",
      "category": "Inflation",
      "countdown_seconds": 172800,
      "consensus": "",
      "prior": "2.400",
      "actual": ""
    }
  ],
  "source": "benzinga",
  "provider_status": "ok",
  "updated_at": "2026-03-07T18:00:00+00:00"
}
```

Data-source contract (implemented):
- Top-level `source`: `benzinga | inferred_news | mixed | unavailable`
- Top-level `provider_status`: `ok | degraded | unavailable`
- Per-event `source`: `benzinga | inferred_news`
- `mixed` indicates merged events from Benzinga + inferred news fallback.
- Inferred events have `id` prefix `inf_`, Benzinga events have `bz_`.

### 5.3 GET /api/landing/macro-tape
Response:
```json
{
  "items": [
    {"label":"VIX","symbol":"VIXY","price":21.34,"pct_1d":6.9,"is_fallback":true},
    {"label":"Gold","symbol":"GLD","price":257.46,"pct_1d":-1.09,"is_fallback":false},
    {"label":"Oil","symbol":"USO","price":82.10,"pct_1d":2.35,"is_fallback":false},
    {"label":"SPY","symbol":"SPY","price":623.88,"pct_1d":-0.15,"is_fallback":false},
    {"label":"QQQ","symbol":"QQQ","price":599.75,"pct_1d":-0.09,"is_fallback":false},
    {"label":"IWM","symbol":"IWM","price":250.89,"pct_1d":-2.29,"is_fallback":false}
  ],
  "updated_at": "2026-03-07T18:00:00Z"
}
```

VIX source priority:
- try `VIX`
- fallback `VIXY`
- fallback `UVXY`

### 5.4 GET /api/landing/movers?limit=10
Response:
```json
{
  "gainers": [{"symbol":"XYZ","pct":3.2,"rel_volume":1.8,"sector":"Energy"}],
  "losers": [{"symbol":"ABC","pct":-2.9,"rel_volume":2.1,"sector":"Technology"}],
  "updated_at": "2026-03-07T18:00:00Z"
}
```

### 5.5 GET /api/landing/themes?limit=6
Response:
```json
{
  "themes": [
    {
      "name": "Crude Oil / Geopolitics",
      "momentum_score": 82.3,
      "sentiment": "bearish",
      "symbols": ["XOM","CVX","USO"],
      "active": true
    }
  ],
  "updated_at": "2026-03-07T18:00:00Z"
}
```

### 5.5 Existing endpoints reused
- `/api/news/feed`
- `/api/news/critical`
- `/api/news/trending-sectors`
- `/api/news/break-impact`

## 6. Frontend Component Plan
New page:
- `platform/frontend/src/pages/LandingPage.tsx`

New components:
- `components/landing/MarketPulseStrip.tsx`
- `components/landing/MacroTape.tsx`
- `components/landing/CatalystClock.tsx`
- `components/landing/MoversPanel.tsx`
- `components/landing/HotThemesPanel.tsx`
- `components/landing/NewsFlowPanel.tsx`
- `components/landing/TradeContextSnapshot.tsx`

Routing:
- `/` -> `LandingPage`
- `/chart/:symbol` unchanged

## 7. State and Data Layer
Add `landingStore` (Zustand) with:
- `marketPulse`
- `macroTape`
- `events72h`
- `movers`
- `themes`
- `selectedNewsItem`
- loading/error states per block

Refresh cadence:
- Market pulse: 60s
- Macro tape: 30-60s
- Catalyst clock: 5m
- Movers: 60s
- Themes: 5m
- News feed: WS + periodic sync

Catalyst provider config (implemented):
- Config block in `platform/config.yaml`:
  - `landing.catalyst.provider`: `benzinga` (only Benzinga implemented in V1)
  - `landing.catalyst.cache_ttl`: `300` (5 minutes)
  - `landing.catalyst.country`: `USA` (3-letter code, maps to Benzinga `country` param)
  - `landing.catalyst.min_impact`: `medium` (maps to Benzinga importance >= 2)
- Required env var:
  - `BENZINGA_API_KEY` (e.g., `bz.xxxxx`)

## 8. Integration with AI Assistant
When user opens chat from landing page, include compact landing context:
- market pulse summary
- top 3 high-impact upcoming events
- top 3 active themes
- nearest S/R + VWAP state when symbol selected

Assistant instructions:
- Prioritize catalysts and event risk in answers
- Avoid generic responses not tied to current landing context

## 9. Performance and Reliability
- Landing first meaningful content < 2.0s on local network
- Individual panels degrade gracefully if one API fails
- Cache short-lived responses server-side for heavy computations

Market pulse source fallback behavior:
- Try `VIX` first.
- If unavailable from Alpaca feed/entitlements, fallback to `VIXY`, then `UVXY`.
- Always return `volatility_source` to make the source explicit in UI.

## 10. Acceptance Criteria
AC-1: `/` renders new command center and old dashboard is removed/replaced.

AC-2: User can identify risk mode, macro tape values (VIX/Gold/Oil/SPY/QQQ/IWM), top events, movers, and themes within one screen without scrolling.

AC-3: Clicking a mover/loser symbol opens chart page correctly.

AC-4: News flow supports critical-only mode and article detail view.

AC-5: Landing context can be consumed by AI assistant for prompt grounding.

AC-6: No major layout break on desktop and mobile breakpoints.

AC-7: Catalyst Clock returns at least one valid source mode (`benzinga`, `inferred_news`, or `mixed`) and exposes `source`/`provider_status` in payload. PASSED — Benzinga economics calendar live with inferred-news fallback.

## 11. Rollout Plan
Phase A:
- Skeleton layout + static placeholders + routing switch

Phase B:
- Wire APIs and stores + live refresh

Phase C:
- Polish interactions (saved views, compact mode, mobile optimization)

## 12. Implementation Checklist
- [ ] Create new landing page and route it to `/`
- [ ] Add landing API endpoints in backend
- [ ] Add landing store and API client
- [ ] Build seven landing components (including Macro Tape)
- [ ] Implement Market Pulse formula and component-level diagnostics
- [ ] Implement VIX source fallback (`VIX -> VIXY -> UVXY`) with source field in response
- [ ] Implement `/api/landing/macro-tape` for VIX/Gold/Oil/SPY/QQQ/IWM
- [ ] Integrate existing news endpoints into News Flow panel
- [ ] Add click-through from movers/themes to chart
- [ ] Add AI context bridge from landing state
- [ ] Validate responsive behavior and loading/error states
- [x] Implement `CatalystCalendarService` with Benzinga economics calendar (`platform/services/catalyst_calendar_service.py`)
- [x] Add catalyst source/status fields to `/api/landing/catalyst-clock`
- [x] Implement inferred-news fallback for catalyst clock (keyword matching from news monitor)
- [x] Wire CatalystClock frontend component to fetch and render real events with countdown timers
- [x] Add `BENZINGA_API_KEY` env var and `landing.catalyst.*` config block
