import { useState } from 'react'
import { useLandingStore } from '../../stores/landingStore'
import { useUiStore } from '../../stores/uiStore'
import { useT } from '../../i18n'
import type { Theme } from '../../types/landing'
import styles from './HotThemes.module.css'

const LIFECYCLE_COLORS: Record<string, string> = {
  emerging: '#42A5F5',
  hot: '#f7525f',
  cooling: '#FF9800',
  faded: 'var(--text-muted)',
}

const LIFECYCLE_LABELS: Record<string, string> = {
  emerging: 'Emerging',
  hot: 'Hot',
  cooling: 'Cooling',
  faded: 'Faded',
}

// Legacy fallback
const MOMENTUM_ARROWS: Record<string, string> = {
  rising: '▲',
  falling: '▼',
  stable: '—',
}

const SENTIMENT_COLORS: Record<string, string> = {
  bullish: 'var(--green)',
  bearish: 'var(--red)',
  neutral: 'var(--text-muted)',
}

function timeAgo(isoStr: string | undefined): string {
  if (!isoStr) return ''
  const diff = Date.now() - new Date(isoStr).getTime()
  const days = Math.floor(diff / 86400000)
  if (days < 1) return 'today'
  if (days < 30) return `${days}d ago`
  const months = Math.floor(days / 30)
  return `${months}mo ago`
}

function isPersistentTheme(th: Theme): boolean {
  return !!th.lifecycle_stage
}

export function HotThemes() {
  const t = useT()
  const themes = useLandingStore((s) => s.themes)
  const loading = useLandingStore((s) => s.loading)
  const locale = useUiStore((s) => s.locale)
  const [refreshing, setRefreshing] = useState(false)

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await fetch('/api/landing/themes/refresh', { method: 'POST' })
      // Reload themes with locale
      const res = await fetch(`/api/landing/themes?limit=10&locale=${locale}`)
      const data = await res.json()
      if (data.themes) {
        useLandingStore.getState().setThemes(data.themes)
      }
    } catch { /* ignore */ }
    setRefreshing(false)
  }

  if (loading && themes.length === 0) return <div className={styles.skeleton} />

  const hasPersistent = themes.some(isPersistentTheme)

  return (
    <div className={styles.panel}>
      <div className={styles.titleRow}>
        <div className={styles.title}>{t('themes.title')}</div>
        {hasPersistent && (
          <button
            className={styles.refreshBtn}
            onClick={handleRefresh}
            disabled={refreshing}
            title="Refresh theme analysis"
          >
            {refreshing ? '...' : '↻'}
          </button>
        )}
      </div>
      {themes.length === 0 ? (
        <div className={styles.empty}>{t('themes.empty')}</div>
      ) : (
        <div className={styles.grid}>
          {themes.map((th) =>
            isPersistentTheme(th) ? (
              <PersistentCard key={th.name} theme={th} />
            ) : (
              <LegacyCard key={th.name} theme={th} t={t} />
            )
          )}
        </div>
      )}
    </div>
  )
}

function PersistentCard({ theme: th }: { theme: Theme }) {
  const stage = th.lifecycle_stage || 'emerging'
  const color = LIFECYCLE_COLORS[stage] || LIFECYCLE_COLORS.emerging

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span
          className={styles.lifecycleBadge}
          style={{ background: color }}
        >
          {LIFECYCLE_LABELS[stage] || stage}
        </span>
        <span className={styles.themeName}>{th.name}</span>
      </div>
      {th.summary && (
        <div className={styles.summary} title={th.summary}>
          {th.summary}
        </div>
      )}
      {th.related_tickers && th.related_tickers.length > 0 && (
        <div className={styles.tickers}>
          {th.related_tickers.slice(0, 6).map((ticker) => (
            <span key={ticker} className={styles.tickerChip}>{ticker}</span>
          ))}
        </div>
      )}
      <div className={styles.meta}>
        {th.first_seen && (
          <span>Since {timeAgo(th.first_seen)}</span>
        )}
        {th.confidence != null && (
          <span className={styles.confidence}>
            {Math.round(th.confidence * 100)}%
          </span>
        )}
      </div>
    </div>
  )
}

function LegacyCard({ theme: th, t }: { theme: Theme; t: (key: string) => string }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div
          className={styles.sentimentDot}
          style={{ background: SENTIMENT_COLORS[th.sentiment || 'neutral'] }}
        />
        <span className={styles.themeName}>{th.name}</span>
        <span className={`${styles.momentum} ${styles[th.momentum || 'stable']}`}>
          {MOMENTUM_ARROWS[th.momentum || 'stable'] || '—'}
        </span>
      </div>
      {th.top_headline && (
        <div className={styles.headline} title={th.top_headline}>
          {th.top_headline}
        </div>
      )}
      <div className={styles.meta}>
        {th.item_count} {t('themes.items')} · {t('themes.score')} {th.impact_score}
      </div>
    </div>
  )
}
