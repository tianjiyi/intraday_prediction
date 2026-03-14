import { useEffect, useState } from 'react'
import { useLandingStore } from '../../stores/landingStore'
import { useT } from '../../i18n'
import type { CatalystEvent } from '../../types/landing'
import styles from './CatalystClock.module.css'

function formatCountdown(seconds: number, t: (key: string) => string): string {
  if (seconds <= 0) return t('catalyst.now')
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d > 0) return `${d}d ${h}h`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function formatTimeAgo(iso: string, t: (key: string) => string): string {
  try {
    const diff = Date.now() - new Date(iso).getTime()
    const h = Math.floor(diff / 3600000)
    const d = Math.floor(h / 24)
    if (d > 0) return `${d}${t('catalyst.dAgo')}`
    if (h > 0) return `${h}${t('catalyst.hAgo')}`
    return t('catalyst.justNow')
  } catch {
    return ''
  }
}

function formatEventTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }) +
      ' ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
  } catch {
    return ''
  }
}

function impactClass(impact: string): string {
  if (impact === 'high') return styles.impactHigh
  if (impact === 'medium') return styles.impactMedium
  return styles.impactLow
}

function EventRow({ event, t }: { event: CatalystEvent; t: (key: string) => string }) {
  const isPast = event.status === 'past'
  const [countdown, setCountdown] = useState(event.countdown_seconds ?? 0)

  useEffect(() => {
    if (isPast || countdown <= 0) return
    const timer = setInterval(() => {
      setCountdown((c) => Math.max(0, c - 1))
    }, 60_000)
    return () => clearInterval(timer)
  }, [isPast, countdown > 0]) // eslint-disable-line react-hooks/exhaustive-deps

  const hasActual = !!event.actual

  return (
    <div className={`${styles.eventRow} ${isPast ? styles.pastRow : ''}`}>
      <div className={styles.eventLeft}>
        <span className={`${styles.impactDot} ${impactClass(event.impact)}`} />
        <div className={styles.eventInfo}>
          <span className={`${styles.eventTitle} ${isPast ? styles.pastTitle : ''}`}>
            {event.title}
          </span>
          <span className={styles.eventTime}>{formatEventTime(event.time)}</span>
          {isPast && hasActual ? (
            <span className={styles.actualValue}>
              {t('catalyst.actual')}{event.actual}{event.prior ? ` ${t('catalyst.prior')}${event.prior})` : ''}
            </span>
          ) : (
            event.detail && <span className={styles.eventDetail}>{event.detail}</span>
          )}
        </div>
      </div>
      <div className={styles.eventRight}>
        {isPast ? (
          <span className={styles.timeAgo}>{formatTimeAgo(event.time, t)}</span>
        ) : (
          <span className={`${styles.countdown} ${countdown === 0 ? styles.countdownNow : ''}`}>
            {formatCountdown(countdown, t)}
          </span>
        )}
        {event.source === 'inferred_news' && (
          <span className={styles.sourceBadge}>{t('catalyst.inferred')}</span>
        )}
      </div>
    </div>
  )
}

export function CatalystClock() {
  const t = useT()
  const catalysts = useLandingStore((s) => s.catalysts)
  const pastEvents = catalysts.filter((e) => e.status === 'past')
  const upcomingEvents = catalysts.filter((e) => e.status !== 'past')

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>{t('catalyst.title')}</span>
        {catalysts.length > 0 && (
          <span className={styles.count}>{catalysts.length}</span>
        )}
      </div>
      {catalysts.length === 0 ? (
        <div className={styles.placeholder}>{t('catalyst.empty')}</div>
      ) : (
        <div className={styles.eventList}>
          {upcomingEvents.length > 0 && (
            <>
              <div className={styles.sectionLabel}>{t('catalyst.upcoming')}</div>
              {upcomingEvents.map((evt) => (
                <EventRow key={evt.id} event={evt} t={t} />
              ))}
            </>
          )}
          {pastEvents.length > 0 && (
            <>
              <div className={styles.sectionLabel}>{t('catalyst.recent')}</div>
              {pastEvents.map((evt) => (
                <EventRow key={evt.id} event={evt} t={t} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}
