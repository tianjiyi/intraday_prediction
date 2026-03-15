import { useState, useEffect, useCallback } from 'react'
import { useLandingStore } from '../../stores/landingStore'
import { useUiStore } from '../../stores/uiStore'
import { useT } from '../../i18n'
import { fetchThemeAnalysis } from '../../api/landing'
import type { ThemeAnalysis } from '../../api/landing'
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

function formatDate(isoStr: string | undefined): string {
  if (!isoStr) return ''
  const d = new Date(isoStr)
  const now = new Date()
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  if (d.getFullYear() !== now.getFullYear()) {
    return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`
  }
  return `${months[d.getMonth()]} ${d.getDate()}`
}

function isPersistentTheme(th: Theme): boolean {
  return !!th.lifecycle_stage
}

// Simple markdown-like rendering: **bold**, bullet points, headings
function renderAnalysis(text: string) {
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (!line.trim()) {
      elements.push(<div key={i} style={{ height: 6 }} />)
      continue
    }

    // Headings
    if (line.startsWith('## ')) {
      elements.push(
        <div key={i} className={styles.analysisHeading}>
          {line.replace(/^##\s+/, '').replace(/\*\*/g, '')}
        </div>
      )
      continue
    }

    // Bold sections like **Title**: content
    const boldMatch = line.match(/^\*\*(.+?)\*\*(.*)/)
    if (boldMatch) {
      elements.push(
        <div key={i} className={styles.analysisLine}>
          <strong>{boldMatch[1]}</strong>{boldMatch[2]}
        </div>
      )
      continue
    }

    // Numbered list or bullet
    if (/^[\d]+[.)]\s/.test(line) || line.startsWith('- ')) {
      elements.push(
        <div key={i} className={styles.analysisBullet}>
          {line.replace(/\*\*(.+?)\*\*/g, '$1')}
        </div>
      )
      continue
    }

    elements.push(
      <div key={i} className={styles.analysisLine}>
        {line.replace(/\*\*(.+?)\*\*/g, '$1')}
      </div>
    )
  }

  return elements
}

export function HotThemes() {
  const t = useT()
  const themes = useLandingStore((s) => s.themes)
  const loading = useLandingStore((s) => s.loading)
  const locale = useUiStore((s) => s.locale)
  const [refreshing, setRefreshing] = useState(false)
  const [modalTheme, setModalTheme] = useState<Theme | null>(null)
  const [analysisCache, setAnalysisCache] = useState<Record<string, ThemeAnalysis>>({})
  const [analysisLoading, setAnalysisLoading] = useState<string | null>(null)

  const closeModal = useCallback(() => {
    setModalTheme(null)
  }, [])

  // Close on Escape
  useEffect(() => {
    if (!modalTheme) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeModal()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [modalTheme, closeModal])

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await fetch('/api/landing/themes/refresh', { method: 'POST' })
      const res = await fetch(`/api/landing/themes?limit=10&locale=${locale}`)
      const data = await res.json()
      if (data.themes) {
        useLandingStore.getState().setThemes(data.themes)
      }
    } catch { /* ignore */ }
    setRefreshing(false)
  }

  const handleThemeClick = async (theme: Theme) => {
    const themeId = theme.id || theme.name
    setModalTheme(theme)

    // Check cache
    const cached = analysisCache[themeId]
    if (cached && cached.cached_until > Date.now() / 1000) {
      return
    }

    // Fetch analysis
    setAnalysisLoading(themeId)
    try {
      const data = await fetchThemeAnalysis(themeId, locale)
      setAnalysisCache((prev) => ({ ...prev, [themeId]: data }))
    } catch (e) {
      console.error('Failed to fetch theme analysis:', e)
    }
    setAnalysisLoading(null)
  }

  if (loading && themes.length === 0) return (
    <div className={styles.panel}>
      <div className={styles.titleRow}>
        <div className={styles.title}>{t('themes.title')}</div>
      </div>
      <div className={styles.panelLoading}>
        <span className={styles.spinner} />
        <span>Analyzing market themes...</span>
      </div>
    </div>
  )

  const hasPersistent = themes.some(isPersistentTheme)
  const modalKey = modalTheme ? (modalTheme.id || modalTheme.name) : ''

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
          {themes.map((th) => {
            const themeKey = th.id || th.name
            return isPersistentTheme(th) ? (
              <PersistentCard
                key={themeKey}
                theme={th}
                expanded={false}
                onClick={() => handleThemeClick(th)}
              />
            ) : (
              <LegacyCard key={th.name} theme={th} t={t} />
            )
          })}
        </div>
      )}

      {/* Modal overlay */}
      {modalTheme && (
        <div className={styles.modalOverlay} onClick={closeModal}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <div className={styles.modalHeaderLeft}>
                <span
                  className={styles.lifecycleBadge}
                  style={{ background: LIFECYCLE_COLORS[modalTheme.lifecycle_stage || 'emerging'] }}
                >
                  {LIFECYCLE_LABELS[modalTheme.lifecycle_stage || 'emerging']}
                </span>
                <span className={styles.modalTitle}>{modalTheme.name}</span>
                {modalTheme.confidence != null && (
                  <span className={styles.modalConfidence}>
                    {Math.round(modalTheme.confidence * 100)}%
                  </span>
                )}
              </div>
              <button className={styles.modalClose} onClick={closeModal}>✕</button>
            </div>

            {modalTheme.summary && (
              <div className={styles.modalSummary}>{modalTheme.summary}</div>
            )}

            {modalTheme.related_tickers && modalTheme.related_tickers.length > 0 && (
              <div className={styles.modalTickers}>
                {modalTheme.related_tickers.map((ticker) => (
                  <span key={ticker} className={styles.tickerChip}>{ticker}</span>
                ))}
              </div>
            )}

            <div className={styles.modalDivider} />

            <div className={styles.modalBody}>
              {analysisLoading === modalKey ? (
                <div className={styles.analysisLoading}>
                  <span className={styles.spinner} />
                  Generating deep analysis...
                </div>
              ) : analysisCache[modalKey] ? (
                <div className={styles.analysisContent}>
                  {renderAnalysis(analysisCache[modalKey].analysis)}
                  <div className={styles.analysisMeta}>
                    {analysisCache[modalKey].related_news_count} related articles analyzed
                  </div>
                </div>
              ) : (
                <div className={styles.analysisLoading}>Loading...</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PersistentCard({ theme: th, expanded, onClick }: { theme: Theme; expanded: boolean; onClick: () => void }) {
  const stage = th.lifecycle_stage || 'emerging'
  const color = LIFECYCLE_COLORS[stage] || LIFECYCLE_COLORS.emerging

  return (
    <div className={`${styles.card} ${expanded ? styles.cardExpanded : ''}`} onClick={onClick}>
      <div className={styles.cardHeader}>
        <span
          className={styles.lifecycleBadge}
          style={{ background: color }}
        >
          {LIFECYCLE_LABELS[stage] || stage}
        </span>
        <span className={styles.themeName}>{th.name}</span>
        <span className={styles.expandIcon}>{expanded ? '▾' : '▸'}</span>
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
          <span>Since {formatDate(th.first_seen)}</span>
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
