import { apiFetch } from './client'
import type { NewsFeedResponse, SectorSummary } from '../types/news'

export async function fetchNewsFeed(
  category = 'all',
  limit = 50
): Promise<NewsFeedResponse> {
  return apiFetch<NewsFeedResponse>(
    `/api/news/feed?category=${category}&limit=${limit}`
  )
}

export async function fetchTrendingSectors(): Promise<{
  sectors: SectorSummary[]
}> {
  return apiFetch('/api/trending_sectors')
}
