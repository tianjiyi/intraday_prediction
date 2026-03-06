import { apiFetch } from './client'
import type { NewsFeedResponse, SectorTrend, BreakImpactResponse } from '../types/news'

export async function fetchNewsFeed(
  category = 'all',
  limit = 50
): Promise<NewsFeedResponse> {
  return apiFetch<NewsFeedResponse>(
    `/api/news/feed?category=${category}&limit=${limit}`
  )
}

export async function fetchTrendingSectors(
  window = '6h',
  limit = 8,
  criticalOnly = false
): Promise<{ window: string; sectors: SectorTrend[]; timestamp: string }> {
  return apiFetch(
    `/api/news/trending-sectors?window=${window}&limit=${limit}&critical_only=${criticalOnly}`
  )
}

export async function fetchCriticalNews(
  sector?: string,
  limit = 30
): Promise<{ items: object[]; count: number; timestamp: string }> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (sector) params.set('sector', sector)
  return apiFetch(`/api/news/critical?${params}`)
}

export async function fetchBreakImpact(): Promise<BreakImpactResponse> {
  return apiFetch<BreakImpactResponse>('/api/news/break-impact')
}
