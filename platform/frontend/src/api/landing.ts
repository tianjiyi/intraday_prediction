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

export function fetchCatalystClock(hours = 72) {
  return apiFetch<CatalystClockResponse>(`/api/landing/catalyst-clock?hours=${hours}`)
}

export function fetchTradeContext(symbol = 'QQQ', timeframe = '1m') {
  return apiFetch<TradeContext>(`/api/landing/trade-context?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`)
}
