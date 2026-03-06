import { useEffect } from 'react'
import { useNewsStore } from '../stores/newsStore'
import { fetchTrendingSectors } from '../api/news'
import styles from './NewsPanels.module.css'

const SENTIMENT_COLORS: Record<string, string> = {
  bullish: 'var(--green)',
  bearish: 'var(--red)',
  neutral: 'var(--text-muted)',
}

export function TrendingSectorsPanel() {
  const trends = useNewsStore((s) => s.sectorTrends)
  const selected = useNewsStore((s) => s.selectedSector)
  const window = useNewsStore((s) => s.window)
  const criticalOnly = useNewsStore((s) => s.criticalOnly)
  const setSectorTrends = useNewsStore((s) => s.setSectorTrends)
  const selectSector = useNewsStore((s) => s.selectSector)

  useEffect(() => {
    fetchTrendingSectors(window, 8, criticalOnly)
      .then((data) => setSectorTrends(data.sectors))
      .catch(console.error)
  }, [window, criticalOnly, setSectorTrends])

  if (trends.length === 0) return null

  const maxRank = Math.max(...trends.map((t) => t.rank_score), 1)

  return (
    <div className={styles.sectorsWrapper}>
      <div className={styles.sectorsTitle}>
        Trending Sectors ({window})
      </div>
      <div className={styles.sectorsList}>
        {trends.map((t) => (
          <div
            key={t.sector}
            className={`${styles.sectorRow} ${selected === t.sector ? styles.selected : ''}`}
            onClick={() => selectSector(t.sector)}
          >
            <span
              className={styles.sentimentDot}
              style={{ background: SENTIMENT_COLORS[t.dominant_sentiment] || 'var(--text-muted)' }}
            />
            <span className={styles.sectorName}>{t.sector}</span>
            <div className={styles.impactBar}>
              <div
                className={styles.impactFill}
                style={{
                  width: `${(t.rank_score / maxRank) * 100}%`,
                  background: t.dominant_sentiment === 'bearish'
                    ? 'var(--red)' : t.dominant_sentiment === 'bullish'
                    ? 'var(--green)' : 'var(--accent)',
                }}
              />
            </div>
            {t.critical_count > 0 && (
              <span className={styles.badge} style={{ background: 'var(--red)' }}>
                {t.critical_count}
              </span>
            )}
            {t.high_count > 0 && (
              <span className={styles.badge} style={{ background: '#FF9800' }}>
                {t.high_count}
              </span>
            )}
            <span className={styles.itemCount}>{t.item_count}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
