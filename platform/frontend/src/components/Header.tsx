import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useMarketStore } from '../stores/marketStore'
import { useChatStore } from '../stores/chatStore'
import styles from './Header.module.css'

export function Header() {
  const navigate = useNavigate()
  const location = useLocation()
  const wsConnected = useMarketStore((s) => s.wsConnected)
  const toggleChat = useChatStore((s) => s.toggleChat)
  const isOpen = useChatStore((s) => s.isOpen)
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
          Kronos
        </h1>
        {!isHome && (
          <button className={styles.backBtn} onClick={() => navigate('/')}>
            Home
          </button>
        )}
      </div>

      <form className={styles.search} onSubmit={handleSearch}>
        <input
          className={styles.searchInput}
          type="text"
          placeholder="Symbol (e.g. QQQ)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </form>

      <div className={styles.right}>
        <span className={`${styles.dot} ${wsConnected ? styles.connected : ''}`} />
        <button
          className={`${styles.chatToggle} ${isOpen ? styles.active : ''}`}
          onClick={toggleChat}
          title="Toggle AI Assistant (Ctrl+K)"
        >
          AI
        </button>
      </div>
    </header>
  )
}
