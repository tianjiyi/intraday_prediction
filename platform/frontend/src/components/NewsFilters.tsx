import { useNewsStore } from '../stores/newsStore'
import styles from './NewsPanels.module.css'

const WINDOWS = ['1h', '6h', '24h'] as const

export function NewsFilters() {
  const window = useNewsStore((s) => s.window)
  const criticalOnly = useNewsStore((s) => s.criticalOnly)
  const selectedSector = useNewsStore((s) => s.selectedSector)
  const setWindow = useNewsStore((s) => s.setWindow)
  const setCriticalOnly = useNewsStore((s) => s.setCriticalOnly)
  const selectSector = useNewsStore((s) => s.selectSector)

  return (
    <div className={styles.filtersBar}>
      {WINDOWS.map((w) => (
        <button
          key={w}
          className={`${styles.filterBtn} ${window === w ? styles.active : ''}`}
          onClick={() => setWindow(w)}
        >
          {w}
        </button>
      ))}

      <span className={styles.divider} />

      <button
        className={`${styles.filterBtn} ${criticalOnly ? styles.critical : ''}`}
        onClick={() => setCriticalOnly(!criticalOnly)}
      >
        Critical Only
      </button>

      {selectedSector && (
        <span className={styles.sectorPill}>
          {selectedSector}
          <button className={styles.pillClose} onClick={() => selectSector(null)}>
            x
          </button>
        </span>
      )}
    </div>
  )
}
