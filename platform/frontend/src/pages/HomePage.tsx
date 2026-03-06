import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useNewsStore } from '../stores/newsStore'
import { fetchNewsFeed } from '../api/news'
import { formatTimeAgo } from '../utils/formatters'
import { TrendingSectorsPanel } from '../components/TrendingSectorsPanel'
import { BreakImpactPanel } from '../components/BreakImpactPanel'
import { NewsFilters } from '../components/NewsFilters'
import styles from './HomePage.module.css'

const CATEGORY_COLORS: Record<string, string> = {
  macro: '#FF9800',
  earnings: '#2962FF',
  policy: '#26A69A',
  geopolitics: '#ef5350',
  company_specific: '#42A5F5',
  market_structure: '#ab47bc',
}

const CATEGORY_LABELS: Record<string, string> = {
  all: 'All',
  macro: 'Macro',
  earnings: 'Earnings',
  policy: 'Policy',
  geopolitics: 'Geopolitics',
  company_specific: 'Company',
  market_structure: 'Structure',
}

const TIER_COLORS: Record<string, string> = {
  critical: '#ef5350',
  high: '#FF9800',
}

const SOURCE_LABELS: Record<string, string> = {
  'X.com': 'X',
  Alpaca: 'A',
  Benzinga: 'B',
  benzinga: 'B',
  Polymarket: 'P',
}

const CATEGORIES = ['all', 'macro', 'earnings', 'policy', 'geopolitics', 'company_specific', 'market_structure']

export function HomePage() {
  const navigate = useNavigate()
  const items = useNewsStore((s) => s.items)
  const activeCategory = useNewsStore((s) => s.activeCategory)
  const criticalOnly = useNewsStore((s) => s.criticalOnly)
  const selectedSector = useNewsStore((s) => s.selectedSector)
  const setCategory = useNewsStore((s) => s.setCategory)
  const setItems = useNewsStore((s) => s.setItems)

  useEffect(() => {
    fetchNewsFeed('all', 50)
      .then((data) => setItems(data.items))
      .catch(console.error)
  }, [setItems])

  // Apply filters
  let filtered = items
  if (activeCategory !== 'all') {
    filtered = filtered.filter((i) => i.category === activeCategory)
  }
  if (criticalOnly) {
    filtered = filtered.filter((i) => i.impact_tier === 'critical')
  }
  if (selectedSector) {
    filtered = filtered.filter((i) => i.sector_tags?.includes(selectedSector))
  }

  return (
    <div className={styles.page}>
      <TrendingSectorsPanel />
      <BreakImpactPanel />

      <div className={styles.categoryBar}>
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            className={`${styles.catBtn} ${activeCategory === cat ? styles.active : ''}`}
            onClick={() => setCategory(cat)}
          >
            {CATEGORY_LABELS[cat] || cat}
          </button>
        ))}
      </div>

      <NewsFilters />

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
                {(CATEGORY_LABELS[item.category] || item.category || '').toUpperCase().slice(0, 4)}
              </span>

              {/* Impact tier badge */}
              {item.impact_tier && TIER_COLORS[item.impact_tier] && (
                <span style={{
                  fontSize: 9, fontWeight: 700, color: '#fff',
                  background: TIER_COLORS[item.impact_tier],
                  borderRadius: 3, padding: '1px 5px', textTransform: 'uppercase',
                }}>
                  {item.impact_tier}
                </span>
              )}

              {item.sentiment === 'bullish' && <span className={styles.bull}>+</span>}
              {item.sentiment === 'bearish' && <span className={styles.bear}>-</span>}
            </div>

            <p className={styles.headline}>{item.headline}</p>

            {item.symbols && item.symbols.length > 0 && (
              <div className={styles.symbols}>{item.symbols.join(', ')}</div>
            )}

            {/* Sector tags */}
            {item.sector_tags && item.sector_tags.length > 0 && (
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
                {item.sector_tags.map((tag) => (
                  <span key={tag} style={{
                    fontSize: 9, color: 'var(--text-muted)',
                    background: 'var(--bg-surface)', borderRadius: 3, padding: '1px 5px',
                  }}>
                    {tag}
                  </span>
                ))}
              </div>
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
