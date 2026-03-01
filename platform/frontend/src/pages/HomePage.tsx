import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useNewsStore } from '../stores/newsStore'
import { fetchNewsFeed } from '../api/news'
import { formatTimeAgo } from '../utils/formatters'
import styles from './HomePage.module.css'

const CATEGORY_COLORS: Record<string, string> = {
  tech: '#2962FF',
  financial: '#FF9800',
  geopolitical: '#ef5350',
  prediction_market: '#ab47bc',
}

const SOURCE_LABELS: Record<string, string> = {
  'X.com': 'X',
  Alpaca: 'A',
  Benzinga: 'B',
  benzinga: 'B',
  Polymarket: 'P',
}

const CATEGORIES = ['all', 'tech', 'financial', 'geopolitical', 'prediction_market']

export function HomePage() {
  const navigate = useNavigate()
  const items = useNewsStore((s) => s.items)
  const activeCategory = useNewsStore((s) => s.activeCategory)
  const setCategory = useNewsStore((s) => s.setCategory)
  const setItems = useNewsStore((s) => s.setItems)

  useEffect(() => {
    fetchNewsFeed('all', 50)
      .then((data) => setItems(data.items))
      .catch(console.error)
  }, [setItems])

  const filtered =
    activeCategory === 'all'
      ? items
      : items.filter((i) => i.category === activeCategory)

  return (
    <div className={styles.page}>
      <div className={styles.categoryBar}>
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            className={`${styles.catBtn} ${activeCategory === cat ? styles.active : ''}`}
            onClick={() => setCategory(cat)}
          >
            {cat === 'all' ? 'All' : cat === 'prediction_market' ? 'Markets' : cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      <div className={styles.feed}>
        {filtered.length === 0 && (
          <p className={styles.empty}>Monitoring news sources...</p>
        )}
        {filtered.map((item) => (
          <div
            key={item.id}
            className={styles.card}
            onClick={() => {
              const ticker = item.symbols?.[0]
              if (ticker) navigate(`/chart/${ticker}`)
            }}
          >
            <div className={styles.cardHeader}>
              <span
                className={styles.sourceIcon}
                style={{ background: CATEGORY_COLORS[item.category] || '#9598a1' }}
              >
                {SOURCE_LABELS[item.source] || item.source?.[0] || '?'}
              </span>
              <span className={styles.time}>{formatTimeAgo(item.created_at)}</span>
              <span
                className={styles.catTag}
                style={{ color: CATEGORY_COLORS[item.category] }}
              >
                {(item.category || '').replace('_', ' ').toUpperCase().slice(0, 3)}
              </span>
              {item.sentiment === 'bullish' && <span className={styles.bull}>+</span>}
              {item.sentiment === 'bearish' && <span className={styles.bear}>-</span>}
            </div>

            <p className={styles.headline}>{item.headline}</p>

            {item.symbols && item.symbols.length > 0 && (
              <div className={styles.symbols}>{item.symbols.join(', ')}</div>
            )}

            {item.probability != null && (
              <div className={styles.probBar}>
                <div className={styles.probBg}>
                  <div
                    className={styles.probFill}
                    style={{ width: `${Math.round(item.probability * 100)}%` }}
                  />
                </div>
                <span className={styles.probPct}>
                  {Math.round(item.probability * 100)}% Yes
                </span>
              </div>
            )}

            {item.source === 'X.com' && (item.views || item.likes) ? (
              <div className={styles.engagement}>
                {item.views ? `${item.views} views` : ''}
                {item.likes ? ` · ${item.likes} likes` : ''}
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  )
}
