import { useEffect, useRef } from 'react'
import { useLandingStore } from '../stores/landingStore'
import { useNewsStore } from '../stores/newsStore'
import { fetchMarketPulse, fetchMacroTape, fetchMovers, fetchThemes, fetchCatalystClock } from '../api/landing'
import { fetchNewsFeed } from '../api/news'
import { MarketPulseStrip } from '../components/landing/MarketPulseStrip'
import { MacroTape } from '../components/landing/MacroTape'
import { CatalystClock } from '../components/landing/CatalystClock'
import { MoversTable } from '../components/landing/MoversTable'
import { HotThemes } from '../components/landing/HotThemes'
import { NewsFlowPanel } from '../components/landing/NewsFlowPanel'
import { TradeContextSnapshot } from '../components/landing/TradeContextSnapshot'
import styles from './HomePage.module.css'

const REFRESH_INTERVAL = 60_000

function loadLandingData() {
  const { setPulse, setMacroTape, setMovers, setThemes, setCatalysts, setLoading, setError } = useLandingStore.getState()
  setLoading(true)
  setError(null)

  Promise.all([
    fetchMarketPulse().then(setPulse).catch(() => {}),
    fetchMacroTape().then((d) => setMacroTape(d.items)).catch(() => {}),
    fetchMovers(10).then((d) => setMovers(d.gainers, d.losers)).catch(() => {}),
    fetchThemes(6).then((d) => setThemes(d.themes)).catch(() => {}),
    fetchCatalystClock(72).then((d) => setCatalysts(d.events)).catch(() => {}),
  ])
    .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'))
    .finally(() => setLoading(false))
}

export function HomePage() {
  const error = useLandingStore((s) => s.error)
  const setNewsItems = useNewsStore((s) => s.setItems)
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined)

  useEffect(() => {
    loadLandingData()

    fetchNewsFeed('all', 50)
      .then((data) => setNewsItems(data.items))
      .catch(console.error)

    intervalRef.current = setInterval(loadLandingData, REFRESH_INTERVAL)
    return () => clearInterval(intervalRef.current)
  }, [setNewsItems])

  return (
    <div className={styles.page}>
      {error && <div className={styles.error}>{error}</div>}

      <MarketPulseStrip />
      <MacroTape />

      <div className={styles.grid}>
        <CatalystClock />
        <MoversTable />
      </div>

      <div className={styles.grid}>
        <HotThemes />
        <NewsFlowPanel />
      </div>

      <TradeContextSnapshot />
    </div>
  )
}
