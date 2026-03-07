# Day Trading Mode V1 Spec

Version: 1.0  
Date: 2026-03-06  
Scope: Platform charting + backend payload enrichment

## 1. Goal
Implement **Day Trading Mode** foundation for future automation.

V1 includes only:
1. Mode gating for minute timeframes (`1m`, `5m`, `15m`)
2. Session VWAP calculation in frontend chart layer (TradingView-style)
3. VWAP bands (`+/-1sigma`, `+/-2sigma`) from frontend chart layer
4. `chart_state` contract so assistant sees the same indicators user sees

Excluded from V1:
- 3PM / 0DTE logic
- CCI entry logic
- Auto-trade execution logic
- Backend-generated VWAP/Bollinger as authoritative indicators

## 2. Functional Requirements

### FR-1 Timeframe gating
- Day Trading Mode is `enabled` only when timeframe is `1`, `5`, or `15` minutes.
- For `30m`, `D`, `W`: mode must be `disabled` and no VWAP-band overlays emitted.

### FR-2 Session reset
- VWAP accumulators reset at new trading day/session boundary.
- Use market session semantics currently used in app (RTH context for stocks).

### FR-3 VWAP formula (must match Pine behavior)
Use these exact running variables per bar:
- `src = (high + low + close) / 3` (HLC3)
- `cumV += volume`
- `cumPV += src * volume`
- `cumPV2 += src^2 * volume`
- `vwap = cumPV / cumV`
- `variance = cumPV2 / cumV - vwap^2`
- `std = sqrt(max(variance, 0))`

Bands:
- `upper1 = vwap + std` (`+1sigma`)
- `lower1 = vwap - std` (`-1sigma`)
- `upper2 = vwap + 2*std` (`+2sigma`)
- `lower2 = vwap - 2*std` (`-2sigma`)

Implementation authority:
- This VWAP package is computed in frontend chart logic and used for rendering.
- Assistant context must use this same frontend-computed VWAP package.

### FR-4 Output consistency
- Emit VWAP package only when mode is enabled and sufficient bar data exists.
- Values must be numeric and JSON-serializable.

## 3. Backend Changes

## 3.1 Assistant context contract (from frontend)
Backend receives chart indicator state from frontend and injects it into assistant context.

Add to chat request payload:

```json
"chart_state": {
  "mode": "day_trading",
  "enabled": true,
  "timeframe_minutes": 1,
  "indicators": {
    "vwap": {
      "value": 0.0,
      "std": 0.0,
      "upper1": 0.0,
      "lower1": 0.0,
      "upper2": 0.0,
      "lower2": 0.0
    }
  }
}
```

Behavior:
- `enabled=true` only for `1/5/15`.
- If disabled, include object with `enabled=false` and `vwap=null` for contract stability.

## 3.2 Location
- Parse `chart_state` in chat API request model and pass to LLM context builder.
- Do not treat backend `prediction_service` VWAP/Bollinger as authoritative for assistant decisions.

## 3.3 Determinism
- Ensure same input bars => same VWAP outputs.
- Clamp variance to non-negative to avoid float noise issues.

## 4. Frontend Changes

### FR-5 Overlay rendering
- In `TradingChart` overlay logic, consume frontend-computed day-trading VWAP package.
- Plot lines:
  - VWAP (orange)
  - `+1sigma/-1sigma`
  - `+2sigma/-2sigma`
- Show only when:
  - Day Trading Mode enabled, and
  - timeframe in `1/5/15`.

### FR-6 Toggle UX
- Add a UI toggle in toolbar/store for Day Trading Mode (default `on` for minute TF).
- Auto-disable visuals when timeframe is not minute-mode.

### FR-7 Type contracts
Update TS types (`types/market.ts` and chat request types) to include:

```ts
interface DayTradingVwap {
  value: number
  std: number
  upper1: number
  lower1: number
  upper2: number
  lower2: number
}

interface DayTradingChartState {
  enabled: boolean
  timeframe_minutes: number
  vwap: DayTradingVwap | null
}
```

Attach to frontend chart state and chat payload model.

## 5. API Contract

Affected endpoint:
- `POST /api/chat` (and any future assistant endpoint)

Request includes consistent `chart_state.day_trading` fields from frontend.

## 6. Acceptance Criteria

### AC-1 Formula parity
For the same intraday bar sequence, frontend VWAP outputs match Pine-style VWAP and sigma bands within float tolerance (`1e-6` intermediate, rounded display acceptable).

### AC-2 Timeframe gating
- `1m/5m/15m`: `day_trading.enabled == true`
- `30m/D/W`: `day_trading.enabled == false`, no overlay plotted

### AC-3 Chart behavior
When enabled:
- VWAP and all 4 bands appear and update as new bars arrive.
When disabled:
- All VWAP-band overlays are hidden/cleared.

### AC-4 Contract stability
No frontend errors when switching timeframe repeatedly; `chart_state` payload shape remains stable.

### AC-5 Assistant parity
Assistant analysis uses frontend-provided VWAP/bands context when available.

## 7. Implementation Checklist

Backend
- [ ] Extend chat request model with `chart_state.day_trading`
- [ ] Inject `chart_state.day_trading` into LLM prompt/context
- [ ] Keep backend VWAP/BB non-authoritative for assistant

Frontend
- [ ] Implement session VWAP accumulator function in chart layer
- [ ] Extend chat request types with `chart_state.day_trading`
- [ ] Add Day Trading Mode toggle state in `uiStore`
- [ ] Render VWAP + band overlays in chart component
- [ ] Clear overlays when mode disabled or non-minute timeframe
- [ ] Send `chart_state.day_trading` with chat requests

Validation
- [ ] Unit test VWAP math function with fixed candles
- [ ] Manual test switching `1m -> 5m -> 15m -> 30m -> 1m`
- [ ] Verify no console errors and no stale overlay artifacts
- [ ] Verify assistant response changes when `chart_state.day_trading` is provided/omitted

## 8. Notes for Claude
- Keep V1 strictly to VWAP + bands.  
- Do not add entry signals, CCI triggers, or 3PM strategy branches in this phase.  
- Prefer small, isolated changes to payload + chart overlay plumbing.

---

## 9. V1.1 Support/Resistance Zones Plan

Goal:
- Add intraday support/resistance zones for `1m/5m/15m` that are visible on chart and available to assistant via `chart_state`.

### 9.1 Scope (V1.1)
In scope:
1. Zone detection from three sources:
   - Pivot clusters
   - Prior day levels (`PDH`, `PDL`, `PDC`)
   - Opening range levels (`OR15`, optional `OR30`)
2. Zone rendering in chart UI
3. Zone payload in `chart_state` for assistant context

Out of scope:
- Auto-trade execution
- Multi-timeframe zone fusion beyond `1m/5m/15m`
- Machine-learned zone scoring

### 9.2 Timeframe gating
- Zones are enabled only when timeframe is `1`, `5`, or `15` minutes.
- Disabled for `30m`, `D`, `W`.

### 9.3 Zone data contract

```json
{
  "zones": [
    {
      "id": "zone_abc123",
      "type": "support",
      "source": "pivot_cluster",
      "low": 602.10,
      "high": 602.60,
      "mid": 602.35,
      "strength": 0.78,
      "touch_count": 4,
      "last_touch_ts": "2026-03-06T18:21:00Z",
      "status": "tested",
      "timeframe_minutes": 1
    }
  ]
}
```

Enums:
- `type`: `support | resistance`
- `source`: `pivot_cluster | prior_day | opening_range`
- `status`: `intact | tested | broken`

### 9.4 Detection rules (default)

1. Pivot detection:
- Fractal pivot with `left=3`, `right=3` bars.
- Pivot high candidates -> resistance clusters.
- Pivot low candidates -> support clusters.

2. Cluster merge threshold:
- Merge pivots into same zone if distance <= `0.08%` of price (default).
- Keep rolling window of recent pivots (default: last 300 bars).

3. Zone width:
- Pivot cluster width = `max(0.05% of price, 0.15 * ATR(14))`.
- Prior day/opening range width = `0.04% of price` on each side (narrow band).

4. Strength score (0-1):
- Weighted components:
  - touch count (40%)
  - rejection magnitude from zone (30%)
  - recency decay (20%)
  - source weight (10%)
- Source weight defaults:
  - `pivot_cluster: 1.0`
  - `prior_day: 0.9`
  - `opening_range: 0.8`

5. Status rules:
- `intact`: no confirmed close beyond zone boundary.
- `tested`: at least 1 touch/rejection recorded.
- `broken`: close beyond zone boundary by > `0.03%` for `2` consecutive bars.

### 9.5 Prior day + opening range rules

Prior day:
- Create zones for `PDH`, `PDL`, `PDC` at session start.

Opening range:
- Compute first 15-minute high/low (`OR15`) after RTH open.
- Optional `OR30` behind flag; default off in V1.1.

### 9.6 Frontend rendering requirements
- Render zones as translucent rectangles:
  - support: green tint
  - resistance: red tint
- Show compact label: `{source} | S:{strength}`.
- Cap visible zones to top `6` by strength to avoid clutter.
- Add toolbar toggle: `Zones`.

### 9.7 Assistant integration
- Add zones to chat request payload under:
  - `chart_state.day_trading.zones`.
- Assistant prompt should reference:
  - nearest support zone,
  - nearest resistance zone,
  - zone status (`intact/tested/broken`).

### 9.8 Acceptance criteria (V1.1)

AC-1:
- Zones appear only on `1m/5m/15m`.

AC-2:
- At least prior day and OR15 zones are visible after session open.

AC-3:
- Pivot cluster zones update incrementally without full redraw lag.

AC-4:
- `chart_state.day_trading.zones` is present and schema-valid in chat payload.

AC-5:
- Assistant mentions nearest support/resistance zone when asked for levels.

### 9.9 Implementation checklist (V1.1)

Frontend
- [ ] Add zone computation utility (pivots, merge, scoring, status)
- [ ] Add zone rendering layer in TradingChart
- [ ] Add `Zones` toggle in ui store + toolbar
- [ ] Extend day-trading chart state with `zones[]`
- [ ] Send zones in chat request payload

Backend/LLM
- [ ] Extend chat request model to accept zones (if strict typing added)
- [ ] Inject zones into LLM context
- [ ] Ensure fallback behavior when no zones exist

Validation
- [ ] Manual test on `1m/5m/15m` with live/replay data
- [ ] Verify no stale zones after timeframe switch
- [ ] Verify zone cap and rendering performance
