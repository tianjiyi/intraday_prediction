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

        // Load historical bars first — chart renders immediately
        const data = await fetchInitialData(symbol, timeframe)
        if (cancelled) return
        const candles = parseHistoricalBars(data.historical)
        useMarketStore.getState().setHistoricalData(candles)
        prevLenRef.current = candles.length

        // Start streaming
        await startStream(symbol, timeframe)

        // Fetch prediction in background — doesn't block chart display
        if (!cancelled) {
          generatePrediction()
            .then((pred) => {
              if (!cancelled && pred.prediction) {
                useMarketStore.getState().setPrediction(pred.prediction)
              }
            })
            .catch(console.error)
        }
      } catch (err) {
        console.error('useChartData load error:', err)
      }
    }

    // Clear stale data and block incoming bar updates until new data loads
    const store = useMarketStore.getState()
    store.setSymbol(symbol)
    store.setTimeframe(timeframe)
    store.setPrediction(null)
    store.setHistoricalData([])
    store.setLoading(true)  // blocks updateBar/addBar until setHistoricalData clears it

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
