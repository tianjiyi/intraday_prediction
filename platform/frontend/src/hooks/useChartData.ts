import { useEffect, useRef } from 'react'
import { useMarketStore } from '../stores/marketStore'
import {
  fetchInitialData,
  startStream,
  stopStream,
  generatePrediction,
} from '../api/market'
import { parseHistoricalBars } from '../utils/chartHelpers'

const PREDICTION_REFRESH_EVERY = 5

export function useChartData(symbol: string, timeframe: number) {
  const barCountRef = useRef(0)
  const prevLenRef = useRef(0)

  useEffect(() => {
    let cancelled = false
    barCountRef.current = 0

    async function load() {
      try {
        await stopStream().catch(() => {})
        const data = await fetchInitialData(symbol, timeframe)
        if (cancelled) return
        const candles = parseHistoricalBars(data.historical)
        useMarketStore.getState().setHistoricalData(candles)
        prevLenRef.current = candles.length
        if (data.prediction) {
          useMarketStore.getState().setPrediction(data.prediction)
        }
        await startStream(symbol, timeframe)
      } catch (err) {
        console.error('useChartData load error:', err)
      }
    }

    // Also set the symbol in the store
    useMarketStore.getState().setSymbol(symbol)
    useMarketStore.getState().setTimeframe(timeframe)

    load()

    return () => {
      cancelled = true
      stopStream().catch(() => {})
    }
  }, [symbol, timeframe])

  // Watch for bar_complete (length increase) → trigger prediction refresh every N bars
  useEffect(() => {
    const unsub = useMarketStore.subscribe((state) => {
      const len = state.historicalData.length
      if (len > prevLenRef.current) {
        barCountRef.current++
        prevLenRef.current = len
        if (barCountRef.current % PREDICTION_REFRESH_EVERY === 0) {
          generatePrediction()
            .then((data) => {
              if (data.prediction) {
                useMarketStore.getState().setPrediction(data.prediction)
              }
            })
            .catch(console.error)
        }
      } else {
        prevLenRef.current = len
      }
    })
    return unsub
  }, [])
}
