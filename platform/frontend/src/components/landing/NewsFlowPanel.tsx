import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useNewsStore } from '../../stores/newsStore'
import { useLandingStore } from '../../stores/landingStore'
import { useT } from '../../i18n'
import { formatTimeAgo } from '../../utils/formatters'
import type { NewsItem } from '../../types/news'
import styles from './NewsFlowPanel.module.css'

const FILTERS = ['all', 'macro', 'earnings', 'geopolitics', 'policy'] as const
const FILTER_KEYS: Record<string, string> = {
  all: 'news.filter.all',
  macro: 'news.filter.macro',
  earnings: 'news.filter.earnings',
  geopolitics: 'news.filter.geopolitics',
  policy: 'news.filter.policy',
}

const TIER_COLORS: Record<string, string> = {
  critical: '#ef5350',
  high: '#FF9800',
  medium: '#66BB6A',
}

const SENTIMENT_COLORS: Record<string, string> = {
  bullish: '#00c853',
  bearish: '#ef5350',
  neutral: '#78909c',
}

function ImpactBar({ score, tier }: { score: number; tier?: string }) {
  const color = tier ? (TIER_COLORS[tier] || '#555') : '#555'
  return (
    <div className={styles.impactBar}>
      <div className={styles.impactFill} style={{ width: `${score}%`, background: color }} />
      <span className={styles.impactLabel}>{Math.round(score)}</span>
    </div>
  )
}

function DetailView({ item, onNavigate, t }: { item: NewsItem; onNavigate: (ticker: string) => void; t: (key: string) => string }) {
  return (
    <div className={styles.detail}>
      {item.images?.[0] && (
        <img src={item.images[0]} alt="" className={styles.detailImage} />
      )}

      {item.summary && (
        <p className={styles.detailSummary}>{item.summary}</p>
      )}

      {item.impact_score != null && (
        <div className={styles.detailSection}>
          <span className={styles.detailLabel}>Impact</span>
          <ImpactBar score={item.impact_score} tier={item.impact_tier} />
        </div>
      )}

      {item.impact_reasons && item.impact_reasons.length > 0 && (
        <ul className={styles.impactReasons}>
          {item.impact_reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}

      <div className={styles.metaRow}>
        {item.sentiment && (
          <span
            className={styles.pill}
            style={{ borderColor: SENTIMENT_COLORS[item.sentiment] || '#555',
                     color: SENTIMENT_COLORS[item.sentiment] || '#999' }}
          >
            {item.sentiment}
          </span>
        )}
        {item.horizon && <span className={styles.pill}>{item.horizon}</span>}
        {item.market_breadth && <span className={styles.pill}>{item.market_breadth.replace('_', ' ')}</span>}
        {item.author && <span className={styles.detailAuthor}>By {item.author}</span>}
      </div>

      {item.sector_tags && item.sector_tags.length > 0 && (
        <div className={styles.sectorTags}>
          {item.sector_tags.map((s) => (
            <span key={s} className={styles.sectorPill}>{s}</span>
          ))}
        </div>
      )}

      {item.symbols && item.symbols.length > 0 && (
        <div className={styles.detailSymbols}>
          {item.symbols.map((s) => (
            <span
              key={s}
              className={styles.symbolChip}
              onClick={(e) => { e.stopPropagation(); onNavigate(s) }}
            >
              {s}
            </span>
          ))}
        </div>
      )}

      <div className={styles.detailActions}>
        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.actionBtn}
            onClick={(e) => e.stopPropagation()}
          >
            {t('news.openArticle')} ↗
          </a>
        )}
        {item.symbols?.[0] && (
          <button
            className={styles.actionBtnSecondary}
            onClick={(e) => { e.stopPropagation(); onNavigate(item.symbols[0]) }}
          >
            {t('news.viewChart')}
          </button>
        )}
      </div>
    </div>
  )
}

export function NewsFlowPanel() {
  const t = useT()
  const navigate = useNavigate()
  const items = useNewsStore((s) => s.items)
  const translating = useLandingStore((s) => s.translating)
  const activeCategory = useNewsStore((s) => s.activeCategory)
  const criticalOnly = useNewsStore((s) => s.criticalOnly)
  const setCategory = useNewsStore((s) => s.setCategory)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  let filtered = items
  if (activeCategory !== 'all') {
    filtered = filtered.filter((i) => i.category === activeCategory)
  }
  if (criticalOnly) {
    filtered = filtered.filter((i) => i.impact_tier === 'critical')
  }

  const handleItemClick = (item: NewsItem) => {
    setSelectedId((prev) => (prev === item.id ? null : item.id))
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.title}>
          {t('news.title')}
          {translating && <span className={styles.translatingBadge}> translating...</span>}
        </div>
        <div className={styles.filterBar}>
          {FILTERS.map((f) => (
            <button
              key={f}
              className={`${styles.filterBtn} ${activeCategory === f ? styles.active : ''}`}
              onClick={() => setCategory(f)}
            >
              {t(FILTER_KEYS[f])}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.feed}>
        {filtered.length === 0 && (
          <div className={styles.empty}>{t('news.empty')}</div>
        )}
        {filtered.slice(0, 30).map((item) => (
          <div key={item.id} className={styles.itemWrapper}>
            <div
              className={`${styles.item} ${selectedId === item.id ? styles.itemExpanded : ''}`}
              onClick={() => handleItemClick(item)}
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
            {selectedId === item.id && (
              <DetailView item={item} onNavigate={(ticker) => navigate(`/chart/${ticker}`)} t={t} />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
