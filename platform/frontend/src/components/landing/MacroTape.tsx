import { useNavigate } from 'react-router-dom'
import { useLandingStore } from '../../stores/landingStore'
import styles from './MacroTape.module.css'

const CLICKABLE_SYMBOLS = new Set(['SPY', 'QQQ', 'IWM'])

export function MacroTape() {
  const navigate = useNavigate()
  const items = useLandingStore((s) => s.macroTape)
  const loading = useLandingStore((s) => s.loading)

  if (loading && items.length === 0) return <div className={styles.skeleton} />

  if (items.length === 0) return null

  return (
    <div className={styles.strip}>
      {items.map((item, i) => {
        const isClickable = CLICKABLE_SYMBOLS.has(item.symbol)
        const isPositive = item.pct_1d >= 0

        return (
          <div key={item.label}>
            {i > 0 && <div className={styles.divider} style={{ display: 'inline-block' }} />}
            <div
              className={`${styles.item} ${isClickable ? styles.clickable : ''}`}
              onClick={isClickable ? () => navigate(`/chart/${item.symbol}`) : undefined}
            >
              <span className={styles.label}>{item.label}</span>
              <span className={styles.price}>
                {item.price > 0 ? item.price.toFixed(2) : '—'}
              </span>
              <span className={`${styles.pct} ${isPositive ? styles.positive : styles.negative}`}>
                {isPositive ? '+' : ''}{item.pct_1d.toFixed(2)}%
              </span>
              {item.is_fallback && <span className={styles.fallback}>*</span>}
            </div>
          </div>
        )
      })}
    </div>
  )
}
