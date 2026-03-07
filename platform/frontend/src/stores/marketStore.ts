import { create } from 'zustand'
import type { Candle, Prediction, DayTradingVwap } from '../types/market'

interface MarketState {
  symbol: string
  timeframe: number
  historicalData: Candle[]
  prediction: Prediction | null
  isStreaming: boolean
  wsConnected: boolean
  dayTradingVwap: DayTradingVwap | null

  setSymbol: (symbol: string) => void
  setTimeframe: (tf: number) => void
  setHistoricalData: (data: Candle[]) => void
  setPrediction: (pred: Prediction) => void
  setStreaming: (streaming: boolean) => void
  setWsConnected: (connected: boolean) => void
  updateBar: (bar: Candle) => void
  addBar: (bar: Candle) => void
  setDayTradingVwap: (vwap: DayTradingVwap | null) => void
}

export const useMarketStore = create<MarketState>((set) => ({
  symbol: 'QQQ',
  timeframe: 1,
  historicalData: [],
  prediction: null,
  isStreaming: false,
  wsConnected: false,
  dayTradingVwap: null,

  setSymbol: (symbol) => set({ symbol }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setHistoricalData: (historicalData) => set({ historicalData }),
  setPrediction: (prediction) => set({ prediction }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  setWsConnected: (wsConnected) => set({ wsConnected }),

  setDayTradingVwap: (dayTradingVwap) => set({ dayTradingVwap }),

  updateBar: (bar) =>
    set((s) => {
      const data = [...s.historicalData]
      if (data.length > 0 && data[data.length - 1].time === bar.time) {
        data[data.length - 1] = bar
      } else {
        data.push(bar)
      }
      return { historicalData: data }
    }),

  addBar: (bar) =>
    set((s) => {
      const data = [...s.historicalData]
      if (data.length > 0 && data[data.length - 1].time === bar.time) {
        // Replace existing bar with completed bar data
        data[data.length - 1] = bar
      } else {
        data.push(bar)
      }
      return { historicalData: data }
    }),
}))
