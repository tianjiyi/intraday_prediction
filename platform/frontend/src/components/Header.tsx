import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useMarketStore } from '../stores/marketStore'
import { useChatStore } from '../stores/chatStore'
import { useUiStore } from '../stores/uiStore'
import { useT } from '../i18n'
import styles from './Header.module.css'

export function Header() {
  const navigate = useNavigate()
  const location = useLocation()
  const wsConnected = useMarketStore((s) => s.wsConnected)
  const toggleChat = useChatStore((s) => s.toggleChat)
  const isOpen = useChatStore((s) => s.isOpen)
  const locale = useUiStore((s) => s.locale)
  const setLocale = useUiStore((s) => s.setLocale)
  const t = useT()
  const [query, setQuery] = useState('')

  const isHome = location.pathname === '/'

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const symbol = query.trim().toUpperCase()
    if (symbol) {
      navigate(`/chart/${symbol}`)
      setQuery('')
    }
  }

  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <h1 className={styles.logo} onClick={() => navigate('/')}>
          {t('header.title')}
        </h1>
        {!isHome && (
          <button className={styles.backBtn} onClick={() => navigate('/')}>
            {t('header.home')}
          </button>
        )}
      </div>

      <form className={styles.search} onSubmit={handleSearch}>
        <input
          className={styles.searchInput}
          type="text"
          placeholder={t('header.searchPlaceholder')}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </form>

      <div className={styles.right}>
        <span className={`${styles.dot} ${wsConnected ? styles.connected : ''}`} />
        <button
          className={styles.langToggle}
          onClick={() => navigate('/options')}
          title="Options Dashboard"
        >
          Options
        </button>
        <button
          className={styles.langToggle}
          onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}
          title={locale === 'en' ? '切换到中文' : 'Switch to English'}
        >
          {locale === 'en' ? '中文' : 'EN'}
        </button>
        <button
          className={`${styles.chatToggle} ${isOpen ? styles.active : ''}`}
          onClick={toggleChat}
          title={t('header.aiTooltip')}
        >
          AI
        </button>
      </div>
    </header>
  )
}
