import { useUiStore } from '../../stores/uiStore'
import styles from './ChartToolbar.module.css'

const TIMEFRAMES = [
  { label: '1m', value: 1 },
  { label: '5m', value: 5 },
  { label: '15m', value: 15 },
  { label: '30m', value: 30 },
  { label: 'D', value: 1440 },
  { label: 'W', value: 10080 },
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
    showIndicators,
    showSMAs,
    showRSI,
    togglePredictions,
    toggleConfidence,
    toggleIndicators,
    toggleSMAs,
    toggleRSI,
  } = useUiStore()

  return (
    <div className={styles.toolbar}>
      <div className={styles.left}>
        <span className={styles.symbol}>{symbol}</span>
        {isStreaming && <span className={styles.live}>LIVE</span>}
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

      <div className={styles.toggles}>
        <button
          className={`${styles.toggle} ${showPredictions ? styles.on : ''}`}
          onClick={togglePredictions}
        >
          Pred
        </button>
        <button
          className={`${styles.toggle} ${showConfidence ? styles.on : ''}`}
          onClick={toggleConfidence}
        >
          Conf
        </button>
        <button
          className={`${styles.toggle} ${showIndicators ? styles.on : ''}`}
          onClick={toggleIndicators}
        >
          Ind
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
      </div>
    </div>
  )
}
