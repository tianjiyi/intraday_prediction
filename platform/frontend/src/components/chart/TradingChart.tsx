import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react'
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  LineStyle,
} from 'lightweight-charts'
import type { IChartApi, ISeriesApi, SeriesType } from 'lightweight-charts'
import type { Candle, Prediction } from '../../types/market'
import { useMarketStore } from '../../stores/marketStore'
import {
  buildChartOptions,
  aggregateCandles,
  splitCandlesVolume,
  mergeIncomingBar,
  buildPredictionPoints,
  buildHorizontalLine,
  buildSmaPoints,
  type OhlcvBar,
} from '../../utils/chartHelpers'
import styles from './TradingChart.module.css'

export interface TradingChartHandle {
  getChart: () => IChartApi | null
  getSeries: () => ISeriesApi<'Candlestick'> | null
}

interface Props {
  symbol: string
  timeframe: number
  historicalData: Candle[]
  prediction: Prediction | null
  showPredictions: boolean
  showConfidence: boolean
  showIndicators: boolean
  showSMAs: boolean
}

export const TradingChart = forwardRef<TradingChartHandle, Props>(
  function TradingChart(
    {
      symbol,
      timeframe,
      historicalData,
      prediction,
      showPredictions,
      showConfidence,
      showIndicators,
      showSMAs,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null)
    const chartRef = useRef<IChartApi | null>(null)

    // Series refs
    const csRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
    const predRef = useRef<ISeriesApi<'Line'> | null>(null)
    const confRefs = useRef<Record<string, ISeriesApi<'Line'>>>({})
    const vwapRef = useRef<ISeriesApi<'Line'> | null>(null)
    const bbRefs = useRef<Record<string, ISeriesApi<'Line'>>>({})
    const smaRefs = useRef<Record<string, ISeriesApi<'Line'>>>({})

    // Real-time aggregation state
    const aggBarRef = useRef<OhlcvBar | null>(null)
    const prevDataLenRef = useRef(0)

    // OHLC overlay state
    const [overlayBar, setOverlayBar] = useState<{
      open: number
      high: number
      low: number
      close: number
    } | null>(null)

    // Expose chart to parent for VolumeChart time-scale sync
    useImperativeHandle(ref, () => ({
      getChart: () => chartRef.current,
      getSeries: () => csRef.current,
    }))

    // EFFECT 1: Create chart + all series on mount
    useEffect(() => {
      if (!containerRef.current) return

      const chart = createChart(containerRef.current, buildChartOptions())
      chartRef.current = chart

      // Candlestick
      csRef.current = chart.addSeries(CandlestickSeries, {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      })

      // Prediction mean line
      predRef.current = chart.addSeries(LineSeries, {
        color: '#2962FF',
        lineWidth: 3,
        title: 'Pred',
      })

      // Confidence bands
      const bandDefs = [
        { key: 'p90', color: 'rgba(41,98,255,0.10)' },
        { key: 'p75', color: 'rgba(41,98,255,0.20)' },
        { key: 'p25', color: 'rgba(41,98,255,0.20)' },
        { key: 'p10', color: 'rgba(41,98,255,0.10)' },
      ]
      for (const { key, color } of bandDefs) {
        confRefs.current[key] = chart.addSeries(LineSeries, {
          color,
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          title: key.toUpperCase(),
        })
      }

      // VWAP
      vwapRef.current = chart.addSeries(LineSeries, {
        color: '#FF6B6B',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        title: 'VWAP',
      })

      // Bollinger Bands
      const bbDefs = [
        { key: 'upper', color: 'rgba(255,193,7,0.5)', width: 1 },
        { key: 'middle', color: 'rgba(255,193,7,0.7)', width: 1 },
        { key: 'lower', color: 'rgba(255,193,7,0.5)', width: 1 },
      ] as const
      for (const { key, color, width } of bbDefs) {
        bbRefs.current[key] = chart.addSeries(LineSeries, {
          color,
          lineWidth: width,
          title: `BB ${key}`,
        })
      }

      // SMA series
      const smaDefs = [
        { key: 'sma5', color: '#FF8C00', width: 2, title: 'SMA 5' },
        { key: 'sma21', color: '#FF0000', width: 2, title: 'SMA 21' },
        { key: 'sma233', color: '#808080', width: 3, title: 'SMA 233' },
      ] as const
      for (const { key, color, width, title } of smaDefs) {
        smaRefs.current[key] = chart.addSeries(LineSeries, {
          color,
          lineWidth: width,
          title,
        })
      }

      // Crosshair → OHLC overlay
      chart.subscribeCrosshairMove((param) => {
        const cs = csRef.current
        if (!param.time || !cs) {
          setOverlayBar(null)
          return
        }
        const d = param.seriesData?.get(cs as ISeriesApi<SeriesType>) as
          | { open: number; high: number; low: number; close: number }
          | undefined
        if (d) {
          setOverlayBar({ open: d.open, high: d.high, low: d.low, close: d.close })
        } else {
          setOverlayBar(null)
        }
      })

      // Real-time tick subscription (imperative, outside React render cycle)
      const unsub = useMarketStore.subscribe((state, prevState) => {
        const cs = csRef.current
        if (!cs || !chartRef.current) return

        const curr = state.historicalData
        const prev = prevState.historicalData
        if (curr === prev || curr.length === 0) return

        const lastBar = curr[curr.length - 1]
        const tf = state.timeframe
        const isAppend = curr.length > prevDataLenRef.current

        aggBarRef.current = mergeIncomingBar(
          isAppend ? null : aggBarRef.current,
          lastBar,
          tf
        )
        prevDataLenRef.current = curr.length

        cs.update(aggBarRef.current as never)
      })

      chart.timeScale().fitContent()

      return () => {
        unsub()
        chart.remove()
        chartRef.current = null
      }
    }, [])

    // EFFECT 2a: Set historical candle data only when data is bulk-replaced
    // (initial load or timeframe change), not on every real-time tick.
    const prevHistLenRef = useRef(0)
    const prevTimeframeRef = useRef(timeframe)

    useEffect(() => {
      const cs = csRef.current
      if (!cs || historicalData.length === 0) return

      // Detect bulk data replacement: timeframe changed, length changed significantly, or first load
      const tfChanged = timeframe !== prevTimeframeRef.current
      const lenDiff = historicalData.length - prevHistLenRef.current
      const isBulkReplace = prevHistLenRef.current === 0 || tfChanged || lenDiff < -5 || lenDiff > 50
      prevHistLenRef.current = historicalData.length
      prevTimeframeRef.current = timeframe

      if (!isBulkReplace) return

      const aggregated = aggregateCandles(historicalData, timeframe)
      const { candles } = splitCandlesVolume(aggregated)
      cs.setData(candles as never)

      aggBarRef.current = null
      prevDataLenRef.current = historicalData.length

      chartRef.current?.timeScale().fitContent()
    }, [historicalData, timeframe])

    // EFFECT 2b: Update prediction overlays when prediction changes
    useEffect(() => {
      const chart = chartRef.current
      if (!chart || historicalData.length === 0) return

      if (!prediction) {
        predRef.current?.setData([])
        Object.values(confRefs.current).forEach((s) => s.setData([]))
        vwapRef.current?.setData([])
        Object.values(bbRefs.current).forEach((s) => s.setData([]))
        Object.values(smaRefs.current).forEach((s) => s.setData([]))
        return
      }

      const aggregated = aggregateCandles(historicalData, timeframe)
      const lastCandle = aggregated[aggregated.length - 1]
      if (!lastCandle) return
      const lastTime = lastCandle.time

      // Prediction mean line
      if (predRef.current && prediction.mean_path) {
        predRef.current.setData(
          buildPredictionPoints(prediction.mean_path, lastTime, timeframe) as never
        )
      }

      // Confidence bands
      const pcts = prediction.percentiles
      if (pcts) {
        for (const key of ['p90', 'p75', 'p25', 'p10'] as const) {
          confRefs.current[key]?.setData(
            buildPredictionPoints(pcts[key], lastTime, timeframe) as never
          )
        }
      }

      // VWAP (intraday only)
      const isDailyOrWeekly = timeframe >= 1440
      if (vwapRef.current) {
        if (!isDailyOrWeekly && prediction.current_vwap) {
          vwapRef.current.setData(
            buildHorizontalLine(prediction.current_vwap, lastTime, timeframe) as never
          )
        } else {
          vwapRef.current.setData([])
        }
      }

      // Bollinger Bands
      const bb = prediction.bollinger_bands
      if (bb) {
        bbRefs.current['upper']?.setData(
          buildHorizontalLine(bb.upper, lastTime, timeframe) as never
        )
        bbRefs.current['middle']?.setData(
          buildHorizontalLine(bb.middle, lastTime, timeframe) as never
        )
        bbRefs.current['lower']?.setData(
          buildHorizontalLine(bb.lower, lastTime, timeframe) as never
        )
      }

      // SMAs (flat arrays, one value per historical candle)
      if (prediction.sma_5_series) {
        smaRefs.current['sma5']?.setData(
          buildSmaPoints(prediction.sma_5_series, historicalData, timeframe) as never
        )
      }
      if (prediction.sma_21_series) {
        smaRefs.current['sma21']?.setData(
          buildSmaPoints(prediction.sma_21_series, historicalData, timeframe) as never
        )
      }
      if (prediction.sma_233_series) {
        smaRefs.current['sma233']?.setData(
          buildSmaPoints(prediction.sma_233_series, historicalData, timeframe) as never
        )
      }
    }, [prediction, timeframe]) // eslint-disable-line react-hooks/exhaustive-deps

    // EFFECT 3-6: Visibility toggles
    useEffect(() => {
      predRef.current?.applyOptions({ visible: showPredictions })
    }, [showPredictions])

    useEffect(() => {
      Object.values(confRefs.current).forEach((s) =>
        s.applyOptions({ visible: showConfidence })
      )
    }, [showConfidence])

    useEffect(() => {
      vwapRef.current?.applyOptions({ visible: showIndicators })
      Object.values(bbRefs.current).forEach((s) =>
        s.applyOptions({ visible: showIndicators })
      )
    }, [showIndicators])

    useEffect(() => {
      Object.values(smaRefs.current).forEach((s) =>
        s.applyOptions({ visible: showSMAs })
      )
    }, [showSMAs])

    return (
      <div className={styles.wrapper}>
        <div ref={containerRef} className={styles.chart} />
        {overlayBar && (
          <div className={styles.overlay}>
            <span className={styles.overlaySymbol}>{symbol}</span>
            <span className={styles.ohlcItem}>
              <span className={styles.ohlcLabel}>O</span>
              {overlayBar.open.toFixed(2)}
            </span>
            <span className={styles.ohlcItem}>
              <span className={styles.ohlcLabel}>H</span>
              {overlayBar.high.toFixed(2)}
            </span>
            <span className={styles.ohlcItem}>
              <span className={styles.ohlcLabel}>L</span>
              {overlayBar.low.toFixed(2)}
            </span>
            <span
              className={`${styles.ohlcItem} ${overlayBar.close >= overlayBar.open ? styles.positive : styles.negative}`}
            >
              <span className={styles.ohlcLabel}>C</span>
              {overlayBar.close.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    )
  }
)
