import { apiFetch } from './client'
import type { InitialDataResponse } from '../types/market'

export async function fetchInitialData(
  symbol: string,
  timeframe: number
): Promise<InitialDataResponse> {
  return apiFetch<InitialDataResponse>(
    `/api/initial_data?symbol=${symbol}&timeframe=${timeframe}`
  )
}

export async function generatePrediction(): Promise<InitialDataResponse> {
  return apiFetch<InitialDataResponse>('/api/generate_prediction')
}

export async function startStream(symbol: string, timeframe: number) {
  return apiFetch('/api/start_stream', {
    method: 'POST',
    body: JSON.stringify({ symbol, timeframe: String(timeframe) }),
  })
}

export async function stopStream() {
  return apiFetch('/api/stop_stream', { method: 'POST' })
}

export async function fetchHealth() {
  return apiFetch<{ status: string; services: Record<string, boolean> }>(
    '/api/health'
  )
}
