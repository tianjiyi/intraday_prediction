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

  // Clear stale data synchronously during render when symbol changes,
  // BEFORE the new TradingChart (keyed by symbol) mounts with old data.
  const prevSymbolRef = useRef(upperSymbol)
  if (prevSymbolRef.current !== upperSymbol) {
    prevSymbolRef.current = upperSymbol
    const store = useMarketStore.getState()
    store.setHistoricalData([])
    store.setPrediction(null)
    store.setLoading(true)
  }

  const timeframe = useMarketStore((s) => s.timeframe)
  const setTimeframe = useMarketStore((s) => s.setTimeframe)
  const historicalData = useMarketStore((s) => s.historicalData)
  const prediction = useMarketStore((s) => s.prediction)
  const isStreaming = useMarketStore((s) => s.isStreaming)

  const {
    showPredictions,
    showConfidence,
    showSMAs,
    showRSI,
    dayTradingMode,
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
            key={upperSymbol}
            ref={chartHandleRef}
            symbol={upperSymbol}
            timeframe={timeframe}
            historicalData={historicalData}
            prediction={prediction}
            showPredictions={showPredictions}
            showConfidence={showConfidence}
            showSMAs={showSMAs}
            dayTradingMode={dayTradingMode}
          />
          {showRSI && (
            <>
              <div className={styles.divider} onMouseDown={rsiResize.onMouseDown} />
              <RsiChart
                key={`rsi-${upperSymbol}`}
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
            key={`vol-${upperSymbol}`}
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
