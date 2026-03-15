import { useEffect, useRef } from 'react'
import { useLandingStore } from '../stores/landingStore'
import { useUiStore } from '../stores/uiStore'
import { useNewsStore } from '../stores/newsStore'
import { fetchMarketPulse, fetchMacroTape, fetchMovers, fetchThemes, fetchCatalystClock, fetchTradeContext } from '../api/landing'
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

function loadLandingData(locale = 'en', setNewsItems?: (items: any[]) => void) {
  const store = useLandingStore.getState()
  store.setLoading(true)
  store.setError(null)

  // Phase 1: Load all data in English (fast, no translation delay)
  const fetches: Promise<void>[] = [
    fetchMarketPulse().then(store.setPulse).catch(() => {}),
    fetchMacroTape().then((d) => store.setMacroTape(d.items)).catch(() => {}),
    fetchMovers(10).then((d) => store.setMovers(d.gainers, d.losers)).catch(() => {}),
    fetchThemes(10, 'en').then((d) => store.setThemes(d.themes)).catch(() => {}),
    fetchCatalystClock(72, 'en').then((d) => store.setCatalysts(d.events)).catch(() => {}),
    fetchTradeContext('QQQ', '1m').then(store.setTradeContext).catch(() => {}),
  ]
  if (setNewsItems) {
    fetches.push(fetchNewsFeed('all', 50, 'en').then((d) => setNewsItems(d.items)).catch(() => {}))
  }

  Promise.all(fetches)
    .catch((e) => store.setError(e instanceof Error ? e.message : 'Failed to load'))
    .finally(() => {
      store.setLoading(false)
      // Phase 2: Translate in background if non-English
      if (locale !== 'en') {
        store.setTranslating(true)
        const translateFetches: Promise<void>[] = [
          fetchThemes(10, locale).then((d) => store.setThemes(d.themes)).catch(() => {}),
          fetchCatalystClock(72, locale).then((d) => store.setCatalysts(d.events)).catch(() => {}),
        ]
        if (setNewsItems) {
          translateFetches.push(fetchNewsFeed('all', 50, locale).then((d) => setNewsItems(d.items)).catch(() => {}))
        }
        Promise.all(translateFetches)
          .finally(() => store.setTranslating(false))
      }
    })
}

export function HomePage() {
  const error = useLandingStore((s) => s.error)
  const locale = useUiStore((s) => s.locale)
  const setNewsItems = useNewsStore((s) => s.setItems)
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined)

  useEffect(() => {
    loadLandingData(locale, setNewsItems)

    intervalRef.current = setInterval(() => loadLandingData(locale, setNewsItems), REFRESH_INTERVAL)
    return () => clearInterval(intervalRef.current)
  }, [setNewsItems, locale])

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
