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
  /** When true, ignore incoming bar updates (symbol switch in progress) */
  _loading: boolean

  setSymbol: (symbol: string) => void
  setTimeframe: (tf: number) => void
  setHistoricalData: (data: Candle[]) => void
  setPrediction: (pred: Prediction | null) => void
  setStreaming: (streaming: boolean) => void
  setWsConnected: (connected: boolean) => void
  updateBar: (bar: Candle) => void
  addBar: (bar: Candle) => void
  setDayTradingVwap: (vwap: DayTradingVwap | null) => void
  setLoading: (v: boolean) => void
}

export const useMarketStore = create<MarketState>((set) => ({
  symbol: 'QQQ',
  timeframe: 1,
  historicalData: [],
  prediction: null,
  isStreaming: false,
  wsConnected: false,
  dayTradingVwap: null,
  _loading: false,

  setSymbol: (symbol) => set({ symbol }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setHistoricalData: (historicalData) => set({ historicalData, _loading: false }),
  setPrediction: (prediction) => set({ prediction }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  setWsConnected: (wsConnected) => set({ wsConnected }),
  setLoading: (_loading) => set({ _loading }),

  setDayTradingVwap: (dayTradingVwap) => set({ dayTradingVwap }),

  updateBar: (bar) =>
    set((s) => {
      if (s._loading) return s
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
      if (s._loading) return s
      const data = [...s.historicalData]
      if (data.length > 0 && data[data.length - 1].time === bar.time) {
        data[data.length - 1] = bar
      } else {
        data.push(bar)
      }
      return { historicalData: data }
    }),
}))
