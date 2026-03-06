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
  // Scored fields (from impact scoring pipeline)
  impact_score?: number
  impact_tier?: string
  sector_tags?: string[]
  impact_reasons?: string[]
  sentiment_strength?: number
  market_breadth?: string
  horizon?: string
  decayed_score?: number
  images?: string[]
}

export interface SectorTrend {
  sector: string
  window: string
  rank_score: number
  impact_sum: number
  impact_avg: number
  critical_count: number
  high_count: number
  dominant_sentiment: string
  top_headline: string
  item_count: number
  last_updated: string
}

export interface BreakImpactResponse {
  analysis: string
  generated_at: string
  critical_count: number
}

export interface NewsFeedResponse {
  items: NewsItem[]
  count: number
  unread_count: number
  timestamp: string
}
