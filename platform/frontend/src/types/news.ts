export interface NewsItem {
  id: string
  headline: string
  summary: string
  source: string
  author: string
  created_at: string
  url: string
  category: string
  symbols: string[]
  sentiment: string
  likes?: number
  retweets?: number
  views?: number
  probability?: number
  volume?: number
  liquidity?: number
}

export interface SectorSummary {
  name: string
  summary: string
  sentiment: 'bullish' | 'bearish' | 'neutral'
  top_tickers: string[]
  momentum: number
}

export interface NewsFeedResponse {
  items: NewsItem[]
  count: number
  unread_count: number
  timestamp: string
}
