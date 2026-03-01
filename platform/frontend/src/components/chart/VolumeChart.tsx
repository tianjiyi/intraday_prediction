import { useEffect, useRef } from 'react'
import {
  createChart,
  HistogramSeries,
  ColorType,
  CrosshairMode,
} from 'lightweight-charts'
import type { IChartApi, ISeriesApi } from 'lightweight-charts'
import type { Candle } from '../../types/market'
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
        background: { type: ColorType.Solid, color: '#1e222d' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      rightPriceScale: {
        borderColor: '#2B2B43',
        scaleMargins: { top: 0.1, bottom: 0 },
      },
      timeScale: {
        borderColor: '#2B2B43',
        visible: false,
      },
      crosshair: { mode: CrosshairMode.Normal },
      autoSize: true,
    })
    chartRef.current = chart

    seriesRef.current = chart.addSeries(HistogramSeries, {
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: 'right',
    })

    // Bidirectional time-scale sync
    const mainChart = mainChartRef.current?.getChart()
    if (mainChart) {
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (syncingRef.current || !range) return
        syncingRef.current = true
        try {
          mainChart.timeScale().setVisibleLogicalRange(range)
        } catch { /* ignore */ }
        syncingRef.current = false
      })
      mainChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (syncingRef.current || !range) return
        syncingRef.current = true
        try {
          chart.timeScale().setVisibleLogicalRange(range)
        } catch { /* ignore */ }
        syncingRef.current = false
      })
    }

    return () => {
      chart.remove()
      chartRef.current = null
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Update volume data
  useEffect(() => {
    if (!seriesRef.current || historicalData.length === 0) return
    const aggregated = aggregateCandles(historicalData, timeframe)
    const { volumes } = splitCandlesVolume(aggregated)
    seriesRef.current.setData(volumes as never)
  }, [historicalData, timeframe])

  return (
    <div
      ref={containerRef}
      className={styles.chart}
      style={{ height }}
    />
  )
}
