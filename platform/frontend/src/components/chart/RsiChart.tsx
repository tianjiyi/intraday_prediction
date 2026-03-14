import { useEffect, useRef, useState } from 'react'
import {
  createChart,
  LineSeries,
  BaselineSeries,
  ColorType,
  CrosshairMode,
  LineStyle,
} from 'lightweight-charts'
import type { IChartApi, ISeriesApi } from 'lightweight-charts'
import type { Candle } from '../../types/market'
import { useMarketStore } from '../../stores/marketStore'
import { aggregateCandles, computeRsi } from '../../utils/chartHelpers'
import type { TradingChartHandle } from './TradingChart'
import styles from './RsiChart.module.css'

const RSI_PERIOD = 14
const SMA_PERIOD = 14

function smaOfRsi(
  rsiPoints: { time: number; value: number }[],
  period: number
): { time: number; value: number }[] {
  if (rsiPoints.length < period) return []
  const result: { time: number; value: number }[] = []
  let sum = 0
  for (let i = 0; i < rsiPoints.length; i++) {
    sum += rsiPoints[i].value
    if (i >= period) sum -= rsiPoints[i - period].value
    if (i >= period - 1) {
      result.push({ time: rsiPoints[i].time, value: sum / period })
    }
  }
  return result
}

interface Props {
  timeframe: number
  historicalData: Candle[]
  mainChartRef: React.RefObject<TradingChartHandle | null>
  height: number
}

export function RsiChart({
  timeframe,
  historicalData,
  mainChartRef,
  height,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const rsiRef = useRef<ISeriesApi<'Line'> | null>(null)
  const smaRef = useRef<ISeriesApi<'Line'> | null>(null)
  const obZoneRef = useRef<ISeriesApi<'Baseline'> | null>(null)
  const osZoneRef = useRef<ISeriesApi<'Baseline'> | null>(null)
  const obRef = useRef<ISeriesApi<'Line'> | null>(null)
  const osRef = useRef<ISeriesApi<'Line'> | null>(null)
  const midRef = useRef<ISeriesApi<'Line'> | null>(null)
  const midBandRef = useRef<ISeriesApi<'Baseline'> | null>(null)
  const syncingRef = useRef(false)
  const [currentRsi, setCurrentRsi] = useState<number | null>(null)
  const [currentSma, setCurrentSma] = useState<number | null>(null)

  // Create chart on mount
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
        scaleMargins: { top: 0.05, bottom: 0.05 },
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

    // Overbought zone fill (above 70) — drawn first so it's behind the line
    obZoneRef.current = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 70 },
      topLineColor: 'transparent',
      topFillColor1: 'rgba(239, 83, 80, 0.25)',
      topFillColor2: 'rgba(239, 83, 80, 0.05)',
      bottomLineColor: 'transparent',
      bottomFillColor1: 'transparent',
      bottomFillColor2: 'transparent',
      lineWidth: 1,
      priceScaleId: 'right',
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    })

    // Oversold zone fill (below 30)
    osZoneRef.current = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 30 },
      topLineColor: 'transparent',
      topFillColor1: 'transparent',
      topFillColor2: 'transparent',
      bottomLineColor: 'transparent',
      bottomFillColor1: 'rgba(38, 166, 154, 0.05)',
      bottomFillColor2: 'rgba(38, 166, 154, 0.25)',
      lineWidth: 1,
      priceScaleId: 'right',
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    })

    // Neutral mid-band fill (30–70)
    midBandRef.current = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 30 },
      topLineColor: 'transparent',
      topFillColor1: 'rgba(255, 255, 255, 0.04)',
      topFillColor2: 'rgba(255, 255, 255, 0.04)',
      bottomLineColor: 'transparent',
      bottomFillColor1: 'transparent',
      bottomFillColor2: 'transparent',
      lineWidth: 1,
      priceScaleId: 'right',
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    })

    // Overbought line (70)
    obRef.current = chart.addSeries(LineSeries, {
      color: 'rgba(239, 83, 80, 0.4)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceScaleId: 'right',
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    })

    // Middle line (50)
    midRef.current = chart.addSeries(LineSeries, {
      color: 'rgba(255, 255, 255, 0.15)',
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceScaleId: 'right',
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    })

    // Oversold line (30)
    osRef.current = chart.addSeries(LineSeries, {
      color: 'rgba(38, 166, 154, 0.4)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceScaleId: 'right',
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    })

    // RSI line
    rsiRef.current = chart.addSeries(LineSeries, {
      color: '#7E57C2',
      lineWidth: 2,
      priceScaleId: 'right',
      priceFormat: { type: 'price', precision: 1, minMove: 0.1 },
    })

    // RSI SMA line
    smaRef.current = chart.addSeries(LineSeries, {
      color: '#FF9800',
      lineWidth: 1,
      priceScaleId: 'right',
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    })

    // Bidirectional time-scale sync (logical range — RSI data is padded with whitespace to align)
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

      // Crosshair sync: main → RSI
      const onMainCrosshairMove = (param: { time?: unknown }) => {
        if (syncingRef.current) return
        syncingRef.current = true
        if (param.time && rsiRef.current) {
          chart.setCrosshairPosition(50, param.time as never, rsiRef.current)
        } else {
          chart.clearCrosshairPosition()
        }
        syncingRef.current = false
      }
      mainChart.subscribeCrosshairMove(onMainCrosshairMove as never)
      // Crosshair sync: RSI → main
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

  // Bulk data update (initial load / timeframe change)
  const prevLenRef = useRef(0)
  const prevTfRef = useRef(timeframe)

  useEffect(() => {
    if (!rsiRef.current || historicalData.length === 0) return
    const tfChanged = timeframe !== prevTfRef.current
    const lenDiff = historicalData.length - prevLenRef.current
    const isBulkReplace = prevLenRef.current === 0 || tfChanged || lenDiff < -5 || lenDiff > 50
    prevLenRef.current = historicalData.length
    prevTfRef.current = timeframe
    if (!isBulkReplace) return

    const aggregated = aggregateCandles(historicalData, timeframe)
    const rsiPoints = computeRsi(aggregated, RSI_PERIOD)

    // Pad with whitespace entries so bar indices align with main chart
    const rsiPad = aggregated.length - rsiPoints.length
    const rsiWhitespace = aggregated.slice(0, rsiPad).map(b => ({ time: b.time }))
    const paddedRsi = [...rsiWhitespace, ...rsiPoints]
    rsiRef.current.setData(paddedRsi as never)

    // SMA of RSI (also padded)
    const smaPoints = smaOfRsi(rsiPoints, SMA_PERIOD)
    const smaPad = aggregated.length - smaPoints.length
    const smaWhitespace = aggregated.slice(0, smaPad).map(b => ({ time: b.time }))
    smaRef.current?.setData([...smaWhitespace, ...smaPoints] as never)

    // Zone fills use the padded RSI data
    obZoneRef.current?.setData(paddedRsi as never)
    osZoneRef.current?.setData(paddedRsi as never)

    // Reference lines spanning the full time range (use aggregated range, not RSI range)
    if (aggregated.length > 0) {
      const firstTime = aggregated[0].time
      const lastTime = aggregated[aggregated.length - 1].time
      const refLine = (val: number) => [
        { time: firstTime, value: val },
        { time: lastTime, value: val },
      ]
      obRef.current?.setData(refLine(70) as never)
      midRef.current?.setData(refLine(50) as never)
      osRef.current?.setData(refLine(30) as never)
      midBandRef.current?.setData(refLine(70) as never)
    }

    if (rsiPoints.length > 0) {
      setCurrentRsi(rsiPoints[rsiPoints.length - 1].value)
    }
    if (smaPoints.length > 0) {
      setCurrentSma(smaPoints[smaPoints.length - 1].value)
    }
  }, [historicalData, timeframe])

  // Real-time RSI updates via store subscription
  useEffect(() => {
    const unsub = useMarketStore.subscribe((state, prevState) => {
      const s = rsiRef.current
      if (!s) return
      const curr = state.historicalData
      if (curr === prevState.historicalData || curr.length < RSI_PERIOD + 1) return

      const tf = state.timeframe
      const aggregated = aggregateCandles(curr, tf)
      if (aggregated.length < RSI_PERIOD + 2) return

      const tail = aggregated.slice(-(RSI_PERIOD + SMA_PERIOD + 30))
      const rsiPoints = computeRsi(tail, RSI_PERIOD)
      if (rsiPoints.length > 0) {
        const last = rsiPoints[rsiPoints.length - 1]
        s.update(last as never)
        setCurrentRsi(last.value)

        // Update zone fills
        obZoneRef.current?.update(last as never)
        osZoneRef.current?.update(last as never)

        // Extend reference lines
        obRef.current?.update({ time: last.time, value: 70 } as never)
        midRef.current?.update({ time: last.time, value: 50 } as never)
        osRef.current?.update({ time: last.time, value: 30 } as never)
        midBandRef.current?.update({ time: last.time, value: 70 } as never)

        // Update SMA
        const smaPoints = smaOfRsi(rsiPoints, SMA_PERIOD)
        if (smaPoints.length > 0) {
          const lastSma = smaPoints[smaPoints.length - 1]
          smaRef.current?.update(lastSma as never)
          setCurrentSma(lastSma.value)
        }
      }
    })
    return unsub
  }, [])

  return (
    <div className={styles.wrapper} style={{ height }}>
      <div className={styles.header}>
        <span className={styles.label}>RSI {RSI_PERIOD}</span>
        {currentRsi !== null && (
          <span className={styles.value}>{currentRsi.toFixed(1)}</span>
        )}
        {currentSma !== null && (
          <span className={styles.smaValue}>SMA {currentSma.toFixed(1)}</span>
        )}
      </div>
      <div ref={containerRef} className={styles.chart} />
    </div>
  )
}
