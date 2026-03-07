import styles from './TradeContextSnapshot.module.css'

export function TradeContextSnapshot() {
  return (
    <div className={styles.panel}>
      <div className={styles.title}>Trade Context</div>
      <div className={styles.placeholder}>
        Intraday regime, VWAP state, and S/R zones coming soon
      </div>
    </div>
  )
}
