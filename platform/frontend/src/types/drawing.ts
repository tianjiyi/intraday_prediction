export type DrawingType = 'hline' | 'trendline' | 'zone'
export type DrawingTool = 'cursor' | 'hline' | 'trendline' | 'zone'

export interface ChartDrawing {
  id: string
  type: DrawingType
  // hline
  price?: number
  // trendline
  startTime?: number  // unix seconds
  startPrice?: number
  endTime?: number
  endPrice?: number
  // zone
  priceHigh?: number
  priceLow?: number
  // common
  color: string
  label?: string
  source: 'user' | 'ai'
}
