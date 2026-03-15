import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react'
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  LineStyle,
  createSeriesMarkers,
} from 'lightweight-charts'
import type { IChartApi, ISeriesApi, ISeriesMarkersPluginApi, SeriesType, Time } from 'lightweight-charts'
import type { Candle, Prediction } from '../../types/market'
import { useMarketStore } from '../../stores/marketStore'
import { useSignalStore } from '../../stores/signalStore'
import { useUiStore } from '../../stores/uiStore'
import {
  buildChartOptions,
  aggregateCandles,
  splitCandlesVolume,
  mergeIncomingBar,
  buildPredictionPoints,
  buildSmaPoints,
  computeSessionVwap,
  type OhlcvBar,
} from '../../utils/chartHelpers'
import { registerChart, unregisterChart } from '../../utils/chartRegistry'
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
  showSMAs: boolean
  dayTradingMode: boolean
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
      showSMAs,
      dayTradingMode,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null)
    const chartRef = useRef<IChartApi | null>(null)

    // Series refs
    const csRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
    const predRef = useRef<ISeriesApi<'Line'> | null>(null)
    const confRefs = useRef<Record<string, ISeriesApi<'Line'>>>({})
    const smaRefs = useRef<Record<string, ISeriesApi<'Line'>>>({})

    // Signal markers plugin
    const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)

    // Day Trading VWAP + band series
    const dtVwapRef = useRef<ISeriesApi<'Line'> | null>(null)
    const dtBandRefs = useRef<Record<string, ISeriesApi<'Line'>>>({})

    // Real-time aggregation state
    const aggBarRef = useRef<OhlcvBar | null>(null)
    const prevDataLenRef = useRef(0)

    // OHLC + indicator overlay state
    const [overlayBar, setOverlayBar] = useState<{
      open: number
      high: number
      low: number
      close: number
      sma5?: number
      sma21?: number
      sma233?: number
      vwap?: number
      bbUpper?: number
      bbLower?: number
      signal?: { type: string; name: string; direction: string; stop?: number; target1?: number; target2?: number; pnl?: number }
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
      registerChart(chart)

      // Candlestick
      csRef.current = chart.addSeries(CandlestickSeries, {
        upColor: '#22d1a0',
        downColor: '#f7525f',
        borderVisible: false,
        wickUpColor: '#22d1a0',
        wickDownColor: '#f7525f',
      })

      // Signal markers plugin (attached to candlestick series)
      markersRef.current = createSeriesMarkers(csRef.current, [])

      // Prediction mean line
      predRef.current = chart.addSeries(LineSeries, {
        color: '#2962FF',
        lineWidth: 3,
        title: 'Pred',
        priceLineVisible: false,
      })

      // Confidence bands
      const bandDefs = [
        { key: 'p90', color: '#5C6BC0', width: 1 as const, style: LineStyle.Dotted },
        { key: 'p75', color: '#7986CB', width: 1 as const, style: LineStyle.Dashed },
        { key: 'p25', color: '#7986CB', width: 1 as const, style: LineStyle.Dashed },
        { key: 'p10', color: '#5C6BC0', width: 1 as const, style: LineStyle.Dotted },
      ]
      for (const { key, color, width, style } of bandDefs) {
        confRefs.current[key] = chart.addSeries(LineSeries, {
          color,
          lineWidth: width,
          lineStyle: style,
          title: key.toUpperCase(),
          priceLineVisible: false,
          lastValueVisible: false,
        })
      }

      // SMA series
      const smaDefs = [
        { key: 'sma5', color: '#FF8C00', width: 2 },
        { key: 'sma21', color: '#FF0000', width: 2 },
        { key: 'sma233', color: '#808080', width: 3 },
      ] as const
      for (const { key, color, width } of smaDefs) {
        smaRefs.current[key] = chart.addSeries(LineSeries, {
          color,
          lineWidth: width,
          lastValueVisible: false,
          priceLineVisible: false,
        })
      }

      // Day Trading session VWAP + sigma bands
      dtVwapRef.current = chart.addSeries(LineSeries, {
        color: '#FFB300',
        lineWidth: 2,
        title: 'VWAP',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      const dtBandDefs = [
        { key: 'upper1', color: '#E08A3C', style: LineStyle.Dashed },
        { key: 'lower1', color: '#E08A3C', style: LineStyle.Dashed },
        { key: 'upper2', color: '#C0653A', style: LineStyle.Dotted },
        { key: 'lower2', color: '#C0653A', style: LineStyle.Dotted },
      ] as const
      for (const { key, color, style } of dtBandDefs) {
        dtBandRefs.current[key] = chart.addSeries(LineSeries, {
          color,
          lineWidth: 1,
          lineStyle: style,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        })
      }

      // Crosshair → OHLC + indicator overlay
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
          const getVal = (s: ISeriesApi<'Line'> | null) => {
            if (!s) return undefined
            const v = param.seriesData?.get(s as ISeriesApi<SeriesType>) as { value?: number } | undefined
            return v?.value
          }
          // Look up signal for this bar's time (prefer entry, fall back to exit)
          const barTime = param.time as number
          const sigs = useSignalStore.getState().signals
          const sig = sigs.find((s) => s.time === barTime && s.type === 'entry')
            || sigs.find((s) => s.time === barTime && s.type === 'exit')
          setOverlayBar({
            open: d.open, high: d.high, low: d.low, close: d.close,
            sma5: getVal(smaRefs.current['sma5']),
            sma21: getVal(smaRefs.current['sma21']),
            sma233: getVal(smaRefs.current['sma233']),
            vwap: getVal(dtVwapRef.current),
            bbUpper: getVal(dtBandRefs.current['upper1']),
            bbLower: getVal(dtBandRefs.current['lower1']),
            signal: sig ? {
              type: sig.type,
              name: sig.signal.replace(/_/g, ' '),
              direction: sig.direction,
              stop: sig.stop,
              target1: sig.target1,
              target2: sig.target2,
              pnl: sig.pnl,
            } : undefined,
          })
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
        unregisterChart()
        chart.remove()
        chartRef.current = null
        csRef.current = null
        predRef.current = null
        confRefs.current = {}
        smaRefs.current = {}
        markersRef.current?.detach()
        markersRef.current = null
        dtVwapRef.current = null
        dtBandRefs.current = {}
        aggBarRef.current = null
        prevDataLenRef.current = 0
      }
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

    // EFFECT 2a: Set historical candle data only when data is bulk-replaced
    // (initial load or timeframe change), not on every real-time tick.
    const prevHistLenRef = useRef(0)
    const prevTimeframeRef = useRef(timeframe)

    useEffect(() => {
      const cs = csRef.current
      if (!cs || historicalData.length === 0) return

      // Detect bulk data replacement
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
      Object.values(smaRefs.current).forEach((s) =>
        s.applyOptions({ visible: showSMAs })
      )
    }, [showSMAs])

    // EFFECT: Day Trading VWAP + bands
    const IS_MINUTE_TF = [1, 5, 15]
    useEffect(() => {
      const enabled = dayTradingMode && IS_MINUTE_TF.includes(timeframe)
      if (!enabled || historicalData.length === 0) {
        dtVwapRef.current?.setData([])
        Object.values(dtBandRefs.current).forEach((s) => s.setData([]))
        useMarketStore.getState().setDayTradingVwap(null)
        return
      }

      const aggregated = aggregateCandles(historicalData, timeframe)
      const vwapPoints = computeSessionVwap(aggregated)
      if (vwapPoints.length === 0) return

      dtVwapRef.current?.setData(
        vwapPoints.map((p) => ({ time: p.time, value: p.vwap })) as never
      )
      dtBandRefs.current['upper1']?.setData(
        vwapPoints.map((p) => ({ time: p.time, value: p.upper1 })) as never
      )
      dtBandRefs.current['lower1']?.setData(
        vwapPoints.map((p) => ({ time: p.time, value: p.lower1 })) as never
      )
      dtBandRefs.current['upper2']?.setData(
        vwapPoints.map((p) => ({ time: p.time, value: p.upper2 })) as never
      )
      dtBandRefs.current['lower2']?.setData(
        vwapPoints.map((p) => ({ time: p.time, value: p.lower2 })) as never
      )

      // Store latest VWAP for chat context
      const last = vwapPoints[vwapPoints.length - 1]
      useMarketStore.getState().setDayTradingVwap({
        value: last.vwap,
        std: last.std,
        upper1: last.upper1,
        lower1: last.lower1,
        upper2: last.upper2,
        lower2: last.lower2,
      })
    }, [historicalData, timeframe, dayTradingMode]) // eslint-disable-line react-hooks/exhaustive-deps

    // Visibility toggle for DT VWAP
    useEffect(() => {
      const visible = dayTradingMode && IS_MINUTE_TF.includes(timeframe)
      dtVwapRef.current?.applyOptions({ visible })
      Object.values(dtBandRefs.current).forEach((s) =>
        s.applyOptions({ visible })
      )
    }, [dayTradingMode, timeframe])

    // EFFECT: Signal markers (entry/exit arrows on candles)
    const signals = useSignalStore((s) => s.signals)
    const showSignals = useUiStore((s) => s.showSignals)

    useEffect(() => {
      const mp = markersRef.current
      if (!mp) return

      if (!showSignals || signals.length === 0) {
        mp.setMarkers([])
        return
      }

      const markers = signals.map((sig) => {
        if (sig.type === 'exit') {
          // Exit markers: purple arrow down above bar
          return {
            time: sig.time as unknown as Time,
            position: 'aboveBar' as const,
            color: '#AB47BC',
            shape: 'arrowDown' as const,
            size: 1,
          }
        }
        // Entry markers: blue arrow up below bar
        return {
          time: sig.time as unknown as Time,
          position: 'belowBar' as const,
          color: '#2962FF',
          shape: 'arrowUp' as const,
          size: 1,
        }
      })

      // LWC requires markers sorted by time ascending
      markers.sort((a, b) => (a.time as number) - (b.time as number))
      mp.setMarkers(markers)
    }, [signals, showSignals])

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
            {overlayBar.sma5 != null && (
              <span className={styles.indicatorItem} style={{ color: '#FF8C00' }}>
                <span className={styles.ohlcLabel}>SMA5</span>
                {overlayBar.sma5.toFixed(2)}
              </span>
            )}
            {overlayBar.sma21 != null && (
              <span className={styles.indicatorItem} style={{ color: '#FF0000' }}>
                <span className={styles.ohlcLabel}>SMA21</span>
                {overlayBar.sma21.toFixed(2)}
              </span>
            )}
            {overlayBar.sma233 != null && (
              <span className={styles.indicatorItem} style={{ color: '#808080' }}>
                <span className={styles.ohlcLabel}>SMA233</span>
                {overlayBar.sma233.toFixed(2)}
              </span>
            )}
            {overlayBar.vwap != null && (
              <span className={styles.indicatorItem} style={{ color: '#FF9800' }}>
                <span className={styles.ohlcLabel}>VWAP</span>
                {overlayBar.vwap.toFixed(2)}
              </span>
            )}
            {overlayBar.bbUpper != null && overlayBar.bbLower != null && (
              <span className={styles.indicatorItem} style={{ color: '#FFC107' }}>
                <span className={styles.ohlcLabel}>BB</span>
                {overlayBar.bbLower.toFixed(2)}–{overlayBar.bbUpper.toFixed(2)}
              </span>
            )}
            {overlayBar.signal && overlayBar.signal.type === 'entry' && (
              <span
                className={styles.indicatorItem}
                style={{ color: '#2962FF' }}
              >
                <span className={styles.ohlcLabel}>
                  {overlayBar.signal.direction === 'long' ? 'BUY' : 'SELL'}
                </span>
                {overlayBar.signal.name}
                {overlayBar.signal.stop != null && ` S:${overlayBar.signal.stop.toFixed(2)}`}
                {overlayBar.signal.target1 != null && ` T1:${overlayBar.signal.target1.toFixed(2)}`}
                {overlayBar.signal.target2 != null && ` T2:${overlayBar.signal.target2.toFixed(2)}`}
              </span>
            )}
            {overlayBar.signal && overlayBar.signal.type === 'exit' && (
              <span
                className={styles.indicatorItem}
                style={{ color: (overlayBar.signal.pnl ?? 0) >= 0 ? '#22d1a0' : '#f7525f' }}
              >
                <span className={styles.ohlcLabel}>EXIT</span>
                {overlayBar.signal.name}
                {overlayBar.signal.pnl != null && (
                  <>{' '}P&L: ${overlayBar.signal.pnl.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</>
                )}
              </span>
            )}
          </div>
        )}
      </div>
    )
  }
)
