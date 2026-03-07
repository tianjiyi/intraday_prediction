import { useLandingStore } from '../../stores/landingStore'
import type { TradeContext, SRZone } from '../../types/landing'
import styles from './TradeContextSnapshot.module.css'

function RegimeBadge({ regime, confidence }: { regime: string; confidence: number }) {
  const cls =
    regime === 'trend' ? styles.regimeTrend :
    regime === 'volatile' ? styles.regimeVolatile :
    regime === 'range' ? styles.regimeRange :
    styles.regimeTransition

  return (
    <div className={styles.metricCard}>
      <span className={styles.metricLabel}>Regime</span>
      <span className={`${styles.regimeBadge} ${cls}`}>{regime.toUpperCase()}</span>
      <span className={styles.metricSub}>{Math.round(confidence * 100)}% conf</span>
    </div>
  )
}

function VwapCard({ vwap }: { vwap: NonNullable<TradeContext['vwap_state']> }) {
  const cls =
    vwap.relation === 'above' ? styles.vwapAbove :
    vwap.relation === 'below' ? styles.vwapBelow :
    styles.vwapCrossing

  const sigmaLabel =
    vwap.sigma_position === 'inside_1sigma' ? 'Inside 1σ' :
    vwap.sigma_position === 'between_1_2sigma' ? '1-2σ' :
    'Outside 2σ'

  return (
    <div className={styles.metricCard}>
      <span className={styles.metricLabel}>VWAP</span>
      <span className={`${styles.vwapRelation} ${cls}`}>
        {vwap.relation.toUpperCase()}
      </span>
      <span className={styles.metricSub}>
        {vwap.distance_to_vwap_pct >= 0 ? '+' : ''}{vwap.distance_to_vwap_pct.toFixed(2)}% | {sigmaLabel}
      </span>
    </div>
  )
}

function ZoneRow({ zone, price }: { zone: SRZone; price?: number }) {
  const isSupport = zone.kind === 'support'
  const distPct = price ? ((price - zone.mid) / price * 100) : 0

  return (
    <div className={styles.zoneRow}>
      <span className={`${styles.zoneDot} ${isSupport ? styles.zoneDotSupport : styles.zoneDotResistance}`} />
      <span className={styles.zoneRange}>{zone.low.toFixed(1)} - {zone.high.toFixed(1)}</span>
      <span className={`${styles.zoneStrength} ${styles[`zone${zone.strength.charAt(0).toUpperCase() + zone.strength.slice(1)}`]}`}>
        {zone.strength}
      </span>
      <span className={styles.zoneDist}>{distPct >= 0 ? '+' : ''}{distPct.toFixed(2)}%</span>
      <span className={styles.zoneTouches}>{zone.touch_count}x</span>
    </div>
  )
}

function EventRiskCard({ event }: { event: NonNullable<TradeContext['event_risk']> }) {
  if (event.status === 'none') {
    return (
      <div className={styles.metricCard}>
        <span className={styles.metricLabel}>Event Risk</span>
        <span className={styles.metricValue}>None</span>
      </div>
    )
  }

  const countdown = event.countdown_seconds ?? 0
  let timeStr: string
  if (countdown < 3600) timeStr = `${Math.floor(countdown / 60)}m`
  else if (countdown < 86400) timeStr = `${Math.floor(countdown / 3600)}h ${Math.floor((countdown % 3600) / 60)}m`
  else timeStr = `${Math.floor(countdown / 86400)}d`

  const isImminent = event.status === 'imminent'

  return (
    <div className={styles.metricCard}>
      <span className={styles.metricLabel}>Event Risk</span>
      <span className={`${styles.eventStatus} ${isImminent ? styles.eventImminent : styles.eventUpcoming}`}>
        {isImminent ? 'IMMINENT' : 'UPCOMING'}
      </span>
      <span className={styles.eventName}>{event.next_event}</span>
      <span className={styles.eventCountdown}>in {timeStr}</span>
    </div>
  )
}

export function TradeContextSnapshot() {
  const tc = useLandingStore((s) => s.tradeContext)

  if (!tc || tc.state === 'not_applicable') {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>
          <span className={styles.title}>Trade Context</span>
          <span className={styles.symbol}>{tc?.symbol ?? 'QQQ'} / {tc?.timeframe ?? '1m'}</span>
        </div>
        <div className={styles.placeholder}>
          {tc?.state === 'not_applicable'
            ? 'Trade context applies to minute timeframes (1m/5m/15m) only'
            : 'Loading trade context...'}
        </div>
      </div>
    )
  }

  if (tc.state === 'unavailable') {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>
          <span className={styles.title}>Trade Context</span>
          <span className={styles.symbol}>{tc.symbol} / {tc.timeframe}</span>
        </div>
        <div className={styles.placeholder}>
          {tc.reason ?? 'Trade context unavailable'}
        </div>
      </div>
    )
  }

  const supports = tc.sr_zones?.nearest_support ?? []
  const resistances = tc.sr_zones?.nearest_resistance ?? []

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>Trade Context</span>
        <span className={styles.symbol}>{tc.symbol} / {tc.timeframe}</span>
      </div>

      {tc.summary && (
        <div className={styles.summaryBar}>{tc.summary}</div>
      )}

      <div className={styles.metricsRow}>
        {tc.intraday_regime && (
          <RegimeBadge regime={tc.intraday_regime} confidence={tc.regime_confidence ?? 0} />
        )}
        {tc.vwap_state && <VwapCard vwap={tc.vwap_state} />}
        {tc.event_risk && <EventRiskCard event={tc.event_risk} />}
      </div>

      {(supports.length > 0 || resistances.length > 0) && (
        <div className={styles.srSection}>
          <span className={styles.srLabel}>S/R Zones</span>
          <div className={styles.zoneList}>
            {resistances.map((z, i) => <ZoneRow key={`r${i}`} zone={z} />)}
            {supports.map((z, i) => <ZoneRow key={`s${i}`} zone={z} />)}
          </div>
        </div>
      )}
    </div>
  )
}
