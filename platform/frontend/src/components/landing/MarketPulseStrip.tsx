import { useLandingStore } from '../../stores/landingStore'
import styles from './MarketPulseStrip.module.css'

const RISK_LABELS: Record<string, string> = {
  risk_on: 'Risk On',
  risk_off: 'Risk Off',
  mixed: 'Mixed',
}

const RISK_CLASSES: Record<string, string> = {
  risk_on: styles.riskOn,
  risk_off: styles.riskOff,
  mixed: styles.mixed,
}

function fngColor(score: number): string {
  if (score <= 25) return 'var(--red)'
  if (score <= 45) return '#ef5350'
  if (score <= 55) return 'var(--text-secondary)'
  if (score <= 75) return '#66bb6a'
  return 'var(--green)'
}

function pcrColor(ratio: number): string {
  // > 1.0 = bearish (more puts), < 0.7 = bullish (more calls)
  if (ratio >= 1.0) return 'var(--red)'
  if (ratio >= 0.85) return '#ef5350'
  if (ratio <= 0.7) return 'var(--green)'
  return 'var(--text-secondary)'
}

export function MarketPulseStrip() {
  const pulse = useLandingStore((s) => s.pulse)
  const loading = useLandingStore((s) => s.loading)

  if (loading && !pulse) return <div className={styles.skeleton} />

  if (!pulse) return null

  // API returned but no real market data yet — show skeleton instead of misleading defaults
  const hasData = pulse.change_summary !== 'No data' && pulse.volatility_level > 0
  if (!hasData) return <div className={styles.skeleton} />

  const volColor = pulse.volatility_state === 'high' ? 'var(--red)'
    : pulse.volatility_state === 'elevated' ? 'var(--orange)' : undefined

  const fng = pulse.fear_greed
  const pcr = pulse.put_call_ratio

  return (
    <div className={styles.strip}>
      <div className={`${styles.riskBadge} ${RISK_CLASSES[pulse.risk_mode] || styles.mixed}`}>
        <span style={{ fontSize: 14 }}>
          {pulse.risk_mode === 'risk_on' ? '▲' : pulse.risk_mode === 'risk_off' ? '▼' : '◆'}
        </span>
        {RISK_LABELS[pulse.risk_mode] || pulse.risk_mode}
        <span className={styles.riskScore}>{pulse.risk_score > 0 ? '+' : ''}{pulse.risk_score}</span>
      </div>

      <div className={styles.divider} />

      <div className={styles.metric}>
        <span className={styles.metricLabel}>Sentiment</span>
        <span
          className={styles.metricValue}
          style={{ color: pulse.sentiment_score > 0 ? 'var(--green)' : pulse.sentiment_score < 0 ? 'var(--red)' : undefined }}
        >
          {pulse.sentiment_score > 0 ? '+' : ''}{pulse.sentiment_score}
        </span>
      </div>

      <div className={styles.divider} />

      <div className={styles.metric}>
        <span className={styles.metricLabel}>Volatility</span>
        <span className={styles.metricValue} style={{ color: volColor }}>
          {pulse.volatility_state.charAt(0).toUpperCase() + pulse.volatility_state.slice(1)}
          {pulse.volatility_level > 0 && (
            <span className={styles.volDetail}>
              {' '}{pulse.volatility_source} {pulse.volatility_level}
            </span>
          )}
        </span>
      </div>

      {fng && fng.score != null && (
        <>
          <div className={styles.divider} />
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Fear & Greed</span>
            <span className={styles.metricValue} style={{ color: fngColor(fng.score) }}>
              {fng.score}
              <span className={styles.volDetail}> {fng.label}</span>
            </span>
          </div>
        </>
      )}

      {pcr && pcr.ratio != null && (
        <>
          <div className={styles.divider} />
          <div className={styles.metric}>
            <span className={styles.metricLabel}>P/C Ratio</span>
            <span className={styles.metricValue} style={{ color: pcrColor(pcr.ratio) }}>
              {pcr.ratio.toFixed(2)}
              {pcr.change != null && (
                <span className={styles.volDetail}>
                  {' '}{pcr.change >= 0 ? '+' : ''}{pcr.change.toFixed(2)}
                </span>
              )}
            </span>
          </div>
        </>
      )}

      <div className={styles.changeSummary}>
        {typeof pulse.change_summary === 'string'
          ? pulse.change_summary
          : ''}
      </div>
    </div>
  )
}
