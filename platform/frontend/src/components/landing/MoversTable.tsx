import { useNavigate } from 'react-router-dom'
import { useLandingStore } from '../../stores/landingStore'
import { useT } from '../../i18n'
import type { Mover } from '../../types/landing'
import styles from './MoversTable.module.css'

function MoverRow({ mover }: { mover: Mover }) {
  const navigate = useNavigate()
  const isPositive = mover.pct_change >= 0

  return (
    <div
      className={styles.row}
      onClick={() => navigate(`/chart/${mover.symbol}`)}
    >
      <span className={styles.symbol}>{mover.symbol}</span>
      <span className={styles.price}>{mover.price.toFixed(2)}</span>
      {mover.sector && <span className={styles.sector}>{mover.sector}</span>}
      <span className={`${styles.pct} ${isPositive ? styles.positive : styles.negative}`}>
        {isPositive ? '+' : ''}{mover.pct_change.toFixed(2)}%
      </span>
    </div>
  )
}

export function MoversTable() {
  const t = useT()
  const gainers = useLandingStore((s) => s.gainers)
  const losers = useLandingStore((s) => s.losers)
  const loading = useLandingStore((s) => s.loading)

  if (loading && gainers.length === 0) return <div className={styles.skeleton} />

  return (
    <div className={styles.panel}>
      <div className={styles.title}>{t('movers.title')}</div>
      <div className={styles.tables}>
        <div className={styles.tableSection}>
          <div className={`${styles.sectionLabel} ${styles.gainersLabel}`}>{t('movers.gainers')}</div>
          {gainers.length === 0 && <div className={styles.empty}>{t('movers.noData')}</div>}
          {gainers.map((m) => <MoverRow key={m.symbol} mover={m} />)}
        </div>
        <div className={styles.tableSection}>
          <div className={`${styles.sectionLabel} ${styles.losersLabel}`}>{t('movers.losers')}</div>
          {losers.length === 0 && <div className={styles.empty}>{t('movers.noData')}</div>}
          {losers.map((m) => <MoverRow key={m.symbol} mover={m} />)}
        </div>
      </div>
    </div>
  )
}
