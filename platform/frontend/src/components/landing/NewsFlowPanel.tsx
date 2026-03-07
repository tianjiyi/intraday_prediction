import { useNavigate } from 'react-router-dom'
import { useNewsStore } from '../../stores/newsStore'
import { formatTimeAgo } from '../../utils/formatters'
import styles from './NewsFlowPanel.module.css'

const FILTERS = ['all', 'macro', 'earnings', 'geopolitics', 'policy'] as const
const FILTER_LABELS: Record<string, string> = {
  all: 'All',
  macro: 'Macro',
  earnings: 'Earn',
  geopolitics: 'Geo',
  policy: 'Policy',
}

const TIER_COLORS: Record<string, string> = {
  critical: '#ef5350',
  high: '#FF9800',
}

export function NewsFlowPanel() {
  const navigate = useNavigate()
  const items = useNewsStore((s) => s.items)
  const activeCategory = useNewsStore((s) => s.activeCategory)
  const criticalOnly = useNewsStore((s) => s.criticalOnly)
  const setCategory = useNewsStore((s) => s.setCategory)

  let filtered = items
  if (activeCategory !== 'all') {
    filtered = filtered.filter((i) => i.category === activeCategory)
  }
  if (criticalOnly) {
    filtered = filtered.filter((i) => i.impact_tier === 'critical')
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.title}>News Flow</div>
        <div className={styles.filterBar}>
          {FILTERS.map((f) => (
            <button
              key={f}
              className={`${styles.filterBtn} ${activeCategory === f ? styles.active : ''}`}
              onClick={() => setCategory(f)}
            >
              {FILTER_LABELS[f]}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.feed}>
        {filtered.length === 0 && (
          <div className={styles.empty}>Monitoring news sources...</div>
        )}
        {filtered.slice(0, 30).map((item) => (
          <div
            key={item.id}
            className={styles.item}
            onClick={() => {
              const ticker = item.symbols?.[0]
              if (ticker) navigate(`/chart/${ticker}`)
            }}
          >
            <span className={styles.itemTime}>
              {formatTimeAgo(item.created_at)}
            </span>
            <div className={styles.itemContent}>
              <div className={styles.itemHeadline}>{item.headline}</div>
              <div className={styles.itemMeta}>
                <span className={styles.itemSource}>{item.source}</span>
                {item.impact_tier && TIER_COLORS[item.impact_tier] && (
                  <span
                    className={styles.tierBadge}
                    style={{ background: TIER_COLORS[item.impact_tier] }}
                  >
                    {item.impact_tier}
                  </span>
                )}
                {item.symbols?.slice(0, 3).map((s) => (
                  <span key={s} className={styles.symbolTag}>{s}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
