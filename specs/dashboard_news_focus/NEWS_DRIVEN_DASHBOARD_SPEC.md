# News-Driven Dashboard and Sector Impact Focus

Version: 1.0
Date: 2026-03-05
Owner: Product/Trading
Target System: `platform/` (FastAPI + React)

## 1. Objective
Build a dashboard workflow that:
1. Surfaces trending sectors based on recent, high-impact news.
2. Forces AI analysis to focus on market-moving catalysts (not low-signal noise).
3. Converts incoming news into actionable market impact context for intraday decisions.

Outcomes:
- Faster identification of market regime/catalyst shifts.
- Better alignment between chart decisions and macro/news risk.
- Reduced AI drift into generic commentary.

## 2. Scope
In scope:
- Backend news scoring and sector aggregation.
- New API endpoints for trending sectors and critical news stream.
- Frontend dashboard components for sector heatmap/list + break-impact panel.
- AI prompt/context changes to prioritize critical news.
- Configurable impact model and source weighting.

Out of scope (phase 1):
- Fully automated trading execution.
- External paid NLP services beyond existing Gemini usage.
- Historical model retraining based on the new scores.

## 3. User Stories
1. As a trader, I can see top trending sectors in the last 1h/6h/24h with impact and direction.
2. As a trader, I can click a sector and have dashboard/news/chat context filtered to that sector.
3. As a trader, I can toggle “Critical Only” to focus on high-impact catalysts.
4. As a trader, I can ask the AI agent for analysis and get responses centered on break-impact events.
5. As a trader, I can view why a news item is high impact and what invalidates its thesis.

## 4. Functional Requirements

### 4.1 News Normalization
FR-1: Normalize every news item into a canonical schema:
- `id`, `source`, `headline`, `summary`, `url`, `created_at`, `symbols[]`, `raw_payload`
- `category`: `macro | earnings | policy | geopolitics | company_specific | market_structure`
- `sector_tags[]`: one or more sectors
- `sentiment`: `bullish | bearish | neutral`
- `sentiment_strength`: float [0, 1]
- `market_breadth`: `single_stock | sector | broad_market`
- `horizon`: `immediate | intraday | multi_day`

FR-2: Deduplicate cross-source duplicates using normalized headline hash + time proximity window.

### 4.2 Impact Scoring Engine
FR-3: Compute `impact_score` (0-100) per item.
Recommended formula:
- `impact_score = 100 * (w_source*S + w_recency*R + w_breadth*B + w_surprise*U + w_sentiment*T + w_linkage*L)`
Where each component is [0,1].

FR-4: Default weight config (editable in config):
- `w_source=0.20`
- `w_recency=0.20`
- `w_breadth=0.20`
- `w_surprise=0.15`
- `w_sentiment=0.10`
- `w_linkage=0.15`

FR-5: Include explanation fields:
- `impact_reasons[]` (e.g., “Fed policy headline”, “Broad market breadth”, “High recency”).
- `confidence_score` (0-1) for classifier confidence.

FR-6: Impact tiers:
- `critical`: >= 75
- `high`: 60-74
- `medium`: 40-59
- `low`: < 40

### 4.3 Sector Trend Aggregation
FR-7: Aggregate by sector for rolling windows (`1h`, `6h`, `24h`):
- `sector_impact_sum`
- `sector_impact_avg`
- `critical_count`
- `high_count`
- `dominant_sentiment`
- `top_headline`
- `last_updated`

FR-8: Trending rank defaults to:
`rank_score = sector_impact_sum + 10*critical_count + 4*high_count`.

FR-9: Sector taxonomy (phase 1):
- Technology, Semiconductors, Financials, Energy, Healthcare, Industrials, Consumer Discretionary, Consumer Staples, Utilities, Real Estate, Communication Services, Materials, Crypto

### 4.4 Critical News Queue
FR-10: Maintain two queues:
- `critical_queue` (impact >= threshold)
- `background_queue` (all else)

FR-11: Provide decay model over time:
- Exponential decay for stale items; configurable half-life by category.

FR-12: If duplicate catalyst appears from higher-quality source, refresh score and timestamp.

### 4.5 API Requirements
Add/extend FastAPI endpoints:

FR-13: `GET /api/news/trending-sectors?window=6h&limit=8&critical_only=false`
Response:
- `window`
- `sectors[]` with rank, impact metrics, sentiment, top_headline

FR-14: `GET /api/news/critical?limit=30&sector=Technology`
Response:
- list of normalized, scored, deduped critical items

FR-15: `GET /api/news/impact-summary?symbol=QQQ`
Response:
- top catalysts
- affected sectors
- risk-on/off bias
- invalidation checkpoints

FR-16: Existing news feed endpoint should include new fields (`impact_score`, `impact_tier`, `sector_tags`, `impact_reasons`).

### 4.6 AI Agent Focus Requirements
FR-17: AI prompt context must inject top N critical events first (default N=5), before general context.

FR-18: AI response policy when critical events exist:
Must output sections in order:
1. `Catalyst`
2. `Transmission Path`
3. `Affected Sectors`
4. `Market Impact Horizon`
5. `Key Levels / Instruments`
6. `Invalidation Conditions`

FR-19: If no critical event exists, AI must explicitly state:
- “No major market-moving catalyst detected in configured window.”

FR-20: AI should deprioritize low-impact items unless user explicitly asks for full feed.

FR-21: Add safeguard prompt rule:
- Reject generic opinions not tied to at least one scored catalyst.

### 4.7 Frontend Dashboard Requirements
FR-22: Add `Trending Sectors` component on dashboard top section:
- list/heatmap card with sector, direction, impact, #critical, #high, top headline.

FR-23: Add `Break Impact` panel:
- “Why it matters now”
- “Likely winners/losers by sector”
- “What to monitor next (SPY/QQQ/VIX/UST10Y/DXY/Oil)”
- “What invalidates this view”

FR-24: Add filters:
- `Window`: 1h / 6h / 24h
- `Critical only` toggle
- Sector multi-select

FR-25: Clicking a sector updates:
- news panel filter
- AI chat default context
- optional chart symbol watchlist suggestions

### 4.8 Config Requirements
FR-26: Extend config with:
- source weights and trust table
- scoring weights
- critical threshold
- decay half-lives by category
- sector taxonomy map

FR-27: All thresholds and weights hot-reload on app restart (no code edit needed).

## 5. Non-Functional Requirements
NFR-1: End-to-end score + aggregation update latency < 2 seconds per polling cycle.
NFR-2: API p95 response time < 300ms for trending sectors endpoint.
NFR-3: System should tolerate source outages; continue with available sources.
NFR-4: Full observability logs for scoring decisions and dropped/duplicate items.
NFR-5: Backward compatible with existing frontend endpoints.

## 6. Data Model Additions
Suggested table fields (existing storage can be adapted):

`news_items` (or equivalent):
- `impact_score` numeric
- `impact_tier` text
- `sector_tags` jsonb/text[]
- `category` text
- `market_breadth` text
- `horizon` text
- `sentiment_strength` numeric
- `confidence_score` numeric
- `impact_reasons` jsonb
- `dedupe_key` text

`sector_trends` (materialized/derived or cache):
- `window`
- `sector`
- `rank_score`
- `impact_sum`
- `impact_avg`
- `critical_count`
- `high_count`
- `dominant_sentiment`
- `top_headline`
- `updated_at`

## 7. Integration Plan (for Claude)

### Backend
1. Update `platform/services/news_monitor_service.py`:
- normalize + classify + score pipeline
- queue separation and decay
- sector aggregation cache

2. Add helper modules:
- `platform/services/news_impact_service.py` (scoring and explanations)
- `platform/services/sector_trend_service.py` (aggregation/ranking)

3. Extend `platform/main.py` endpoints for trending sectors/critical feed/impact summary.

4. Update existing `/api/news/feed` payload to include score metadata.

### AI Layer
5. Update `platform/services/llm_service.py`:
- prepend critical catalyst context
- enforce response schema when critical events exist
- enforce “no major catalyst” fallback message

### Frontend
6. Add components:
- `TrendingSectorsPanel`
- `BreakImpactPanel`

7. Add store and API wiring:
- `newsStore` fields for sector trends and critical-only mode
- new client API methods for added endpoints

8. Add UI interactions:
- sector click filter propagation to news + chat context

## 8. Acceptance Criteria
AC-1: Dashboard shows top sectors with correct ranking within selected window.
AC-2: At least 90% of displayed critical items have `impact_score >= threshold` and valid sector tags.
AC-3: AI response references at least one critical catalyst when such catalyst exists.
AC-4: “Critical only” toggle excludes medium/low items across dashboard and chat context.
AC-5: System remains operational when one source (e.g., Twitter) is down.

## 9. Test Requirements

### Unit Tests
- score calculation correctness for controlled inputs.
- impact tier boundary tests.
- dedupe behavior for near-identical headlines.
- sector aggregation and ranking correctness.

### Integration Tests
- source fetch -> normalize -> score -> queue -> API response.
- sector filter propagation from UI selection to API request payload.
- LLM prompt generation includes critical queue block.

### Regression Tests
- existing `/api/news` and `/api/news/feed` consumers do not break.
- existing chat endpoint continues functioning without new fields.

## 10. Rollout Phases
Phase 1 (MVP):
- scoring, trending sectors endpoint, critical queue, basic UI panels.

Phase 2:
- richer explainability, invalidation auto-generation, improved sector-symbol graph.

Phase 3:
- adaptive thresholds by market volatility regime and post-event impact tracking.

## 11. Risks and Mitigations
- Risk: Over-scoring noisy headlines.
  Mitigation: source trust weights + dedupe + strict critical threshold.

- Risk: AI still drifts to generic output.
  Mitigation: strict response schema and prompt guardrails; reject if no catalyst cited.

- Risk: Latency from too much processing.
  Mitigation: cached aggregation and incremental updates only.

## 12. Open Questions
1. Preferred default window for trading desk: `1h` or `6h`?
2. Should sector direction be weighted by market cap of affected symbols?
3. Do we want a manual analyst override for impact tiers?

## 13. Deliverables Checklist for Implementer
- [ ] Backend scoring module
- [ ] Sector aggregation module
- [ ] New endpoints in FastAPI
- [ ] Frontend panels + filters
- [ ] AI prompt update with critical-catalyst focus
- [ ] Config additions documented
- [ ] Tests (unit + integration)
- [ ] Migration note for any schema changes
