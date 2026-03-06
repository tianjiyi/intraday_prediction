import { useEffect, useState } from 'react'
import { fetchBreakImpact } from '../api/news'
import type { BreakImpactResponse } from '../types/news'
import styles from './NewsPanels.module.css'

export function BreakImpactPanel() {
  const [data, setData] = useState<BreakImpactResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    let mounted = true
    const load = () => {
      setLoading(true)
      fetchBreakImpact()
        .then((res) => { if (mounted) setData(res) })
        .catch(console.error)
        .finally(() => { if (mounted) setLoading(false) })
    }
    load()
    const interval = setInterval(load, 5 * 60 * 1000)
    return () => { mounted = false; clearInterval(interval) }
  }, [])

  if (!data && !loading) return null

  return (
    <div className={styles.breakWrapper}>
      <div className={styles.breakHeader} onClick={() => setCollapsed(!collapsed)}>
        <span className={`${styles.breakArrow} ${collapsed ? styles.collapsed : ''}`}>
          ▼
        </span>
        Break Impact
        {data && data.critical_count > 0 && (
          <span className={styles.badge} style={{ background: 'var(--red)', marginLeft: 4 }}>
            {data.critical_count}
          </span>
        )}
        {loading && <span className={styles.updating}>updating...</span>}
      </div>

      {!collapsed && data && (
        <div className={styles.breakContent}>
          {data.analysis.split('\n').map((line, i) => {
            if (line.startsWith('##') || line.startsWith('**')) {
              return <div key={i} className={styles.breakHeading}>{line.replace(/^#+\s*/, '').replace(/\*\*/g, '')}</div>
            }
            if (line.startsWith('- ')) {
              return <div key={i} className={styles.breakBullet}>{line}</div>
            }
            return <div key={i}>{line}</div>
          })}
        </div>
      )}
    </div>
  )
}
