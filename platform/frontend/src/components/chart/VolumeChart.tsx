import { useEffect, useRef } from 'react'
import {
  createChart,
  HistogramSeries,
  ColorType,
  CrosshairMode,
} from 'lightweight-charts'
import type { IChartApi, ISeriesApi } from 'lightweight-charts'
import type { Candle } from '../../types/market'
import { useMarketStore } from '../../stores/marketStore'
import { aggregateCandles, splitCandlesVolume } from '../../utils/chartHelpers'
import type { TradingChartHandle } from './TradingChart'
import styles from './VolumeChart.module.css'

interface Props {
  timeframe: number
  historicalData: Candle[]
  mainChartRef: React.RefObject<TradingChartHandle | null>
  height: number
}

export function VolumeChart({
  timeframe,
  historicalData,
  mainChartRef,
  height,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const syncingRef = useRef(false)

  // Create volume chart on mount
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f1218' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#1c2030' },
        horzLines: { color: '#1c2030' },
      },
      rightPriceScale: {
        borderColor: '#1c2030',
        scaleMargins: { top: 0.1, bottom: 0 },
        minimumWidth: 80,
      },
      timeScale: {
        borderColor: '#1c2030',
        visible: false,
      },
      crosshair: { mode: CrosshairMode.Normal },
      autoSize: true,
    })
    chartRef.current = chart

    seriesRef.current = chart.addSeries(HistogramSeries, {
      color: '#22d1a0',
      priceFormat: { type: 'volume' },
      priceScaleId: 'right',
    })

    // Bidirectional time-scale sync (logical range)
    const mainChart = mainChartRef.current?.getChart()
    const mainSeries = mainChartRef.current?.getSeries()
    if (mainChart) {
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (syncingRef.current || !range) return
        syncingRef.current = true
        try { mainChart.timeScale().setVisibleLogicalRange(range) } catch { /* ignore */ }
        syncingRef.current = false
      })
      const onMainRangeChange = (range: { from: number; to: number } | null) => {
        if (syncingRef.current || !range) return
        syncingRef.current = true
        try { chart.timeScale().setVisibleLogicalRange(range) } catch { /* ignore */ }
        syncingRef.current = false
      }
      mainChart.timeScale().subscribeVisibleLogicalRangeChange(onMainRangeChange)

      // Crosshair sync: main → volume
      const onMainCrosshairMove = (param: { time?: unknown }) => {
        if (syncingRef.current) return
        syncingRef.current = true
        if (param.time && seriesRef.current) {
          chart.setCrosshairPosition(0, param.time as never, seriesRef.current)
        } else {
          chart.clearCrosshairPosition()
        }
        syncingRef.current = false
      }
      mainChart.subscribeCrosshairMove(onMainCrosshairMove as never)
      // Crosshair sync: volume → main
      chart.subscribeCrosshairMove((param) => {
        if (syncingRef.current) return
        syncingRef.current = true
        if (param.time && mainSeries) {
          mainChart.setCrosshairPosition(0, param.time, mainSeries)
        } else {
          mainChart.clearCrosshairPosition()
        }
        syncingRef.current = false
      })

      return () => {
        mainChart.timeScale().unsubscribeVisibleLogicalRangeChange(onMainRangeChange)
        mainChart.unsubscribeCrosshairMove(onMainCrosshairMove as never)
        chart.remove()
        chartRef.current = null
      }
    }

    return () => {
      chart.remove()
      chartRef.current = null
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Update volume data — bulk replace only (initial load / timeframe change),
  // plus real-time updates via store subscription
  const prevVolLenRef = useRef(0)
  const prevVolTfRef = useRef(timeframe)

  useEffect(() => {
    if (!seriesRef.current || historicalData.length === 0) return
    const tfChanged = timeframe !== prevVolTfRef.current
    const lenDiff = historicalData.length - prevVolLenRef.current
    const isBulkReplace = prevVolLenRef.current === 0 || tfChanged || lenDiff < -5 || lenDiff > 50
    prevVolLenRef.current = historicalData.length
    prevVolTfRef.current = timeframe
    if (!isBulkReplace) return

    const aggregated = aggregateCandles(historicalData, timeframe)
    const { volumes } = splitCandlesVolume(aggregated)
    seriesRef.current.setData(volumes as never)
  }, [historicalData, timeframe])

  // Real-time volume updates via store subscription
  useEffect(() => {
    const unsub = useMarketStore.subscribe((state, prevState) => {
      const s = seriesRef.current
      if (!s) return
      const curr = state.historicalData
      if (curr === prevState.historicalData || curr.length === 0) return
      const last = curr[curr.length - 1]
      s.update({
        time: last.time,
        value: last.volume ?? 0,
        color: last.close >= last.open ? '#22d1a0' : '#f7525f',
      } as never)
    })
    return unsub
  }, [])

  return (
    <div
      ref={containerRef}
      className={styles.chart}
      style={{ height }}
    />
  )
}
