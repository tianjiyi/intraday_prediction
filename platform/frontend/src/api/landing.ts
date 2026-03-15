import { apiFetch } from './client'
import type {
  MarketPulse,
  MacroTapeResponse,
  MoversResponse,
  ThemesResponse,
  CatalystClockResponse,
  TradeContext,
} from '../types/landing'

export function fetchMarketPulse() {
  return apiFetch<MarketPulse>('/api/landing/market-pulse')
}

export function fetchMacroTape() {
  return apiFetch<MacroTapeResponse>('/api/landing/macro-tape')
}

export function fetchMovers(limit = 10) {
  return apiFetch<MoversResponse>(`/api/landing/movers?limit=${limit}`)
}

export function fetchThemes(limit = 10, locale = 'en') {
  return apiFetch<ThemesResponse>(`/api/landing/themes?limit=${limit}&locale=${locale}`)
}

export function fetchCatalystClock(hours = 72, locale = 'en') {
  return apiFetch<CatalystClockResponse>(`/api/landing/catalyst-clock?hours=${hours}&locale=${locale}`)
}

export interface ThemeAnalysis {
  theme_name: string
  lifecycle_stage: string
  analysis: string
  related_tickers: string[]
  related_news_count: number
  generated_at: number
  cached_until: number
  locale: string
}

export function fetchThemeAnalysis(themeId: string, locale = 'en') {
  return apiFetch<ThemeAnalysis>(`/api/landing/themes/${encodeURIComponent(themeId)}/analysis?locale=${locale}`)
}

export function fetchTradeContext(symbol = 'QQQ', timeframe = '1m') {
  return apiFetch<TradeContext>(`/api/landing/trade-context?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`)
}
