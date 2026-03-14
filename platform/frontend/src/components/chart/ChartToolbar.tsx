import { useUiStore } from '../../stores/uiStore'
import { useDrawingStore } from '../../stores/drawingStore'
import { useT } from '../../i18n'
import type { DrawingTool } from '../../types/drawing'
import styles from './ChartToolbar.module.css'

const TIMEFRAMES = [
  { label: '1m', value: 1 },
  { label: '5m', value: 5 },
  { label: '15m', value: 15 },
  { label: '30m', value: 30 },
  { label: 'D', value: 1440 },
  { label: 'W', value: 10080 },
]

const DRAWING_TOOLS: { tool: DrawingTool; label: string; titleKey: string }[] = [
  { tool: 'cursor', label: '\u2197', titleKey: 'toolbar.select' },
  { tool: 'hline', label: '\u2500', titleKey: 'toolbar.hline' },
  { tool: 'trendline', label: '\u2571', titleKey: 'toolbar.trendLine' },
  { tool: 'zone', label: '\u25AD', titleKey: 'toolbar.zone' },
]

interface Props {
  symbol: string
  timeframe: number
  isStreaming: boolean
  onTimeframeChange: (tf: number) => void
}

export function ChartToolbar({
  symbol,
  timeframe,
  isStreaming,
  onTimeframeChange,
}: Props) {
  const {
    showPredictions,
    showConfidence,
    showSMAs,
    showRSI,
    dayTradingMode,
    togglePredictions,
    toggleConfidence,
    toggleSMAs,
    toggleRSI,
    toggleDayTradingMode,
  } = useUiStore()

  const t = useT()
  const activeTool = useDrawingStore((s) => s.activeTool)
  const setActiveTool = useDrawingStore((s) => s.setActiveTool)
  const clearAll = useDrawingStore((s) => s.clearAll)
  const drawingCount = useDrawingStore((s) => s.drawings.length)

  return (
    <div className={styles.toolbar}>
      <div className={styles.left}>
        <span className={styles.symbol}>{symbol}</span>
        {isStreaming && <span className={styles.live}>{t('toolbar.live')}</span>}
      </div>

      <div className={styles.timeframes}>
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.value}
            className={`${styles.tfBtn} ${timeframe === tf.value ? styles.active : ''}`}
            onClick={() => onTimeframeChange(tf.value)}
          >
            {tf.label}
          </button>
        ))}
      </div>

      <div className={styles.drawingTools}>
        {DRAWING_TOOLS.map(({ tool, label, titleKey }) => (
          <button
            key={tool}
            className={`${styles.drawBtn} ${activeTool === tool ? styles.active : ''}`}
            onClick={() => setActiveTool(tool)}
            title={t(titleKey)}
          >
            {label}
          </button>
        ))}
        {drawingCount > 0 && (
          <button
            className={styles.drawBtn}
            onClick={clearAll}
            title={t('toolbar.clearAll')}
          >
            &#x2715;
          </button>
        )}
      </div>

      <div className={styles.toggles}>
        <button
          className={`${styles.toggle} ${showPredictions ? styles.on : ''}`}
          onClick={togglePredictions}
        >
          {t('toolbar.pred')}
        </button>
        <button
          className={`${styles.toggle} ${showConfidence ? styles.on : ''}`}
          onClick={toggleConfidence}
        >
          {t('toolbar.conf')}
        </button>
        <button
          className={`${styles.toggle} ${showSMAs ? styles.on : ''}`}
          onClick={toggleSMAs}
        >
          SMA
        </button>
        <button
          className={`${styles.toggle} ${showRSI ? styles.on : ''}`}
          onClick={toggleRSI}
        >
          RSI
        </button>
        <button
          className={`${styles.toggle} ${dayTradingMode ? styles.on : ''}`}
          onClick={toggleDayTradingMode}
        >
          {t('toolbar.dt')}
        </button>
      </div>
    </div>
  )
}
