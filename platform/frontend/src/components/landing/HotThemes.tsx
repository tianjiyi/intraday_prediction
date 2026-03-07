import { useLandingStore } from '../../stores/landingStore'
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
  const themes = useLandingStore((s) => s.themes)
  const loading = useLandingStore((s) => s.loading)

  if (loading && themes.length === 0) return <div className={styles.skeleton} />

  return (
    <div className={styles.panel}>
      <div className={styles.title}>Hot Themes</div>
      {themes.length === 0 ? (
        <div className={styles.empty}>No active themes</div>
      ) : (
        <div className={styles.grid}>
          {themes.map((t) => (
            <div key={t.name} className={styles.card}>
              <div className={styles.cardHeader}>
                <div
                  className={styles.sentimentDot}
                  style={{ background: SENTIMENT_COLORS[t.sentiment] || SENTIMENT_COLORS.neutral }}
                />
                <span className={styles.themeName}>{t.name}</span>
                <span className={`${styles.momentum} ${styles[t.momentum]}`}>
                  {MOMENTUM_ARROWS[t.momentum] || '—'}
                </span>
              </div>
              {t.top_headline && (
                <div className={styles.headline} title={t.top_headline}>
                  {t.top_headline}
                </div>
              )}
              <div className={styles.meta}>
                {t.item_count} items · Score {t.impact_score}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
