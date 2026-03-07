import { apiFetch } from './client'
import type {
  MarketPulse,
  MacroTapeResponse,
  MoversResponse,
  ThemesResponse,
  CatalystClockResponse,
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

export function fetchThemes(limit = 6) {
  return apiFetch<ThemesResponse>(`/api/landing/themes?limit=${limit}`)
}

export function fetchCatalystClock(hours = 72) {
  return apiFetch<CatalystClockResponse>(`/api/landing/catalyst-clock?hours=${hours}`)
}
