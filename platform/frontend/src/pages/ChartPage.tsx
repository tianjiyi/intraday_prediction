import { useRef } from 'react'
import { useParams } from 'react-router-dom'
import { useMarketStore } from '../stores/marketStore'
import { useChartData } from '../hooks/useChartData'
import { usePanelResize } from '../hooks/usePanelResize'
import { useUiStore } from '../stores/uiStore'
import { ChartToolbar } from '../components/chart/ChartToolbar'
import { TradingChart, type TradingChartHandle } from '../components/chart/TradingChart'
import { VolumeChart } from '../components/chart/VolumeChart'
import { RsiChart } from '../components/chart/RsiChart'
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
    showRSI,
  } = useUiStore()

  // Data lifecycle
  useChartData(upperSymbol, timeframe)

  // Chart ref for volume sync
  const chartHandleRef = useRef<TradingChartHandle>(null)

  // Resizable panels
  const rsiResize = usePanelResize({
    storageKey: 'rsiChartHeight',
    defaultSize: 120,
    minSize: 60,
    maxSize: 300,
    direction: 'row',
    invert: true,
  })
  const volResize = usePanelResize({
    storageKey: 'volumeChartHeight',
    defaultSize: 120,
    minSize: 60,
    maxSize: 300,
    direction: 'row',
    invert: true,
  })
  const statsResize = usePanelResize({
    storageKey: 'statsPanelWidth',
    defaultSize: 200,
    minSize: 140,
    maxSize: 400,
    direction: 'col',
    invert: true,
  })

  return (
    <div className={styles.page}>
      <ChartToolbar
        symbol={upperSymbol}
        timeframe={timeframe}
        isStreaming={isStreaming}
        onTimeframeChange={setTimeframe}
      />

      <div className={styles.chartsArea}>
        <div className={styles.chartsWrapper}>
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
          {showRSI && (
            <>
              <div className={styles.divider} onMouseDown={rsiResize.onMouseDown} />
              <RsiChart
                timeframe={timeframe}
                historicalData={historicalData}
                mainChartRef={chartHandleRef}
                height={rsiResize.size}
              />
            </>
          )}
          <div
            className={styles.divider}
            onMouseDown={volResize.onMouseDown}
          />
          <VolumeChart
            timeframe={timeframe}
            historicalData={historicalData}
            mainChartRef={chartHandleRef}
            height={volResize.size}
          />
        </div>

        <div
          className={styles.colDivider}
          onMouseDown={statsResize.onMouseDown}
        />
        <StatsPanel
          symbol={upperSymbol}
          prediction={prediction}
          isStreaming={isStreaming}
          width={statsResize.size}
        />
      </div>
    </div>
  )
}
