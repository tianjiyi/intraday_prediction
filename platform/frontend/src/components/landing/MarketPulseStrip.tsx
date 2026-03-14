import { useLandingStore } from '../../stores/landingStore'
import { useT } from '../../i18n'
import styles from './MarketPulseStrip.module.css'

const RISK_LABEL_KEYS: Record<string, string> = {
  risk_on: 'pulse.riskOn',
  risk_off: 'pulse.riskOff',
  mixed: 'pulse.mixed',
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
  const t = useT()
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
        {RISK_LABEL_KEYS[pulse.risk_mode] ? t(RISK_LABEL_KEYS[pulse.risk_mode]) : pulse.risk_mode}
        <span className={styles.riskScore}>{pulse.risk_score > 0 ? '+' : ''}{pulse.risk_score}</span>
      </div>

      <div className={styles.divider} />

      <div className={styles.metric}>
        <span className={styles.metricLabel}>{t('pulse.sentiment')}</span>
        <span
          className={styles.metricValue}
          style={{ color: pulse.sentiment_score > 0 ? 'var(--green)' : pulse.sentiment_score < 0 ? 'var(--red)' : undefined }}
        >
          {pulse.sentiment_score > 0 ? '+' : ''}{pulse.sentiment_score}
        </span>
      </div>

      <div className={styles.divider} />

      <div className={styles.metric}>
        <span className={styles.metricLabel}>{t('pulse.volatility')}</span>
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
            <span className={styles.metricLabel}>{t('pulse.fearGreed')}</span>
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
            <span className={styles.metricLabel}>{t('pulse.pcRatio')}</span>
            <span className={styles.metricValue} style={{ color: pcrColor(pcr.ratio) }}>
              {pcr.ratio.toFixed(2)}
              <span className={styles.volDetail}> SPY</span>
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
