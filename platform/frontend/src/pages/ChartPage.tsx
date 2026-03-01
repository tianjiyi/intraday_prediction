import { useRef } from 'react'
import { useParams } from 'react-router-dom'
import { useMarketStore } from '../stores/marketStore'
import { useChartData } from '../hooks/useChartData'
import { useResizableDivider } from '../hooks/useResizableDivider'
import { useUiStore } from '../stores/uiStore'
import { ChartToolbar } from '../components/chart/ChartToolbar'
import { TradingChart, type TradingChartHandle } from '../components/chart/TradingChart'
import { VolumeChart } from '../components/chart/VolumeChart'
import { StatsPanel } from '../components/chart/StatsPanel'
import styles from './ChartPage.module.css'

export function ChartPage() {
  const { symbol = 'QQQ' } = useParams<{ symbol: string }>()
  const upperSymbol = symbol.toUpperCase()

  const timeframe = useMarketStore((s) => s.timeframe)
  const setTimeframe = useMarketStore((s) => s.setTimeframe)
  const historicalData = useMarketStore((s) => s.historicalData)
  const prediction = useMarketStore((s) => s.prediction)
  const isStreaming = useMarketStore((s) => s.isStreaming)

  const {
    showPredictions,
    showConfidence,
    showIndicators,
    showSMAs,
  } = useUiStore()

  // Data lifecycle
  useChartData(upperSymbol, timeframe)

  // Chart ref for volume sync
  const chartHandleRef = useRef<TradingChartHandle>(null)

  // Resizable divider
  const wrapperRef = useRef<HTMLDivElement>(null)
  const { volumeHeight, onDividerMouseDown } = useResizableDivider(wrapperRef)

  return (
    <div className={styles.page}>
      <ChartToolbar
        symbol={upperSymbol}
        timeframe={timeframe}
        isStreaming={isStreaming}
        onTimeframeChange={setTimeframe}
      />

      <div className={styles.chartsArea}>
        <div className={styles.chartsWrapper} ref={wrapperRef}>
          <TradingChart
            ref={chartHandleRef}
            symbol={upperSymbol}
            timeframe={timeframe}
            historicalData={historicalData}
            prediction={prediction}
            showPredictions={showPredictions}
            showConfidence={showConfidence}
            showIndicators={showIndicators}
            showSMAs={showSMAs}
          />
          <div
            className={styles.divider}
            onMouseDown={onDividerMouseDown}
          />
          <VolumeChart
            timeframe={timeframe}
            historicalData={historicalData}
            mainChartRef={chartHandleRef}
            height={volumeHeight}
          />
        </div>

        <StatsPanel
          symbol={upperSymbol}
          prediction={prediction}
          isStreaming={isStreaming}
        />
      </div>
    </div>
  )
}
