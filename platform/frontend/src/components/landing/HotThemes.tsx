import { useLandingStore } from '../../stores/landingStore'
import { useT } from '../../i18n'
import styles from './HotThemes.module.css'

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

export function HotThemes() {
  const t = useT()
  const themes = useLandingStore((s) => s.themes)
  const loading = useLandingStore((s) => s.loading)

  if (loading && themes.length === 0) return <div className={styles.skeleton} />

  return (
    <div className={styles.panel}>
      <div className={styles.title}>{t('themes.title')}</div>
      {themes.length === 0 ? (
        <div className={styles.empty}>{t('themes.empty')}</div>
      ) : (
        <div className={styles.grid}>
          {themes.map((th) => (
            <div key={th.name} className={styles.card}>
              <div className={styles.cardHeader}>
                <div
                  className={styles.sentimentDot}
                  style={{ background: SENTIMENT_COLORS[th.sentiment] || SENTIMENT_COLORS.neutral }}
                />
                <span className={styles.themeName}>{th.name}</span>
                <span className={`${styles.momentum} ${styles[th.momentum]}`}>
                  {MOMENTUM_ARROWS[th.momentum] || '—'}
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
          ))}
        </div>
      )}
    </div>
  )
}
