export interface FearGreed {
  score: number | null
  label: string | null
  previous_close?: number | null
  change?: number | null
  source: string
  error?: string
  updated_at: string
}

export interface PutCallRatio {
  ratio: number | null
  previous?: number | null
  change?: number | null
  trade_date?: string | null
  source: string
  error?: string
  updated_at: string
}

export interface MarketPulse {
  risk_mode: 'risk_on' | 'risk_off' | 'mixed'
  risk_score: number
  sentiment_score: number
  volatility_state: 'normal' | 'elevated' | 'high'
  volatility_source: string
  volatility_level: number
  volatility_change_1d_pct: number
  fear_greed: FearGreed | null
  put_call_ratio: PutCallRatio | null
  components: {
    vol_component: number
    macro_component: number
    breadth_component: number
    news_component: number
  }
  proxies: {
    dollar_symbol: string
    rates_symbol: string
    breadth_universe_size: number
  }
  change_summary: string
  updated_at: string
}

export interface MacroTapeItem {
  label: string
  symbol: string
  price: number
  pct_1d: number
  is_fallback: boolean
}

export interface MacroTapeResponse {
  items: MacroTapeItem[]
  updated_at: string
}

export interface Mover {
  symbol: string
  price: number
  pct_change: number
  volume: number
  rel_volume: number
  sector: string
}

export interface Theme {
  name: string
  momentum: 'rising' | 'falling' | 'stable'
  sentiment: 'bullish' | 'bearish' | 'neutral'
  impact_score: number
  top_headline: string
  item_count: number
}

export interface CatalystEvent {
  id: string
  type: 'economic' | 'earnings' | 'geopolitical' | 'placeholder'
  title: string
  time: string
  status?: 'past' | 'upcoming'
  impact: 'high' | 'medium' | 'low'
  detail?: string
  source?: 'benzinga' | 'inferred_news'
  category?: string
  countdown_seconds?: number
  consensus?: string
  prior?: string
  actual?: string
}

export interface MoversResponse {
  gainers: Mover[]
  losers: Mover[]
  updated_at: string
}

export interface ThemesResponse {
  themes: Theme[]
  updated_at: string
}

export interface CatalystClockResponse {
  window_hours: number
  events: CatalystEvent[]
  source?: 'benzinga' | 'inferred_news' | 'mixed' | 'unavailable'
  provider_status?: 'ok' | 'degraded' | 'unavailable'
  placeholder?: boolean
  updated_at: string
}

export interface SRZone {
  kind: 'support' | 'resistance'
  low: number
  high: number
  mid: number
  strength: 'strong' | 'medium' | 'weak'
  status: 'active' | 'tested'
  touch_count: number
}

export interface TradeContext {
  symbol: string
  timeframe: string
  state: 'ok' | 'not_applicable' | 'unavailable'
  reason?: string
  intraday_regime?: 'trend' | 'range' | 'volatile' | 'transition'
  regime_confidence?: number
  vwap_state?: {
    relation: 'above' | 'below' | 'crossing'
    sigma_position: 'inside_1sigma' | 'between_1_2sigma' | 'outside_2sigma'
    distance_to_vwap: number
    distance_to_vwap_pct: number
  }
  sr_zones?: {
    nearest_support: SRZone[]
    nearest_resistance: SRZone[]
  }
  event_risk?: {
    status: 'imminent' | 'upcoming' | 'none'
    next_event?: string
    impact?: string
    time?: string
    countdown_seconds?: number
  }
  summary?: string
  updated_at: string
}
