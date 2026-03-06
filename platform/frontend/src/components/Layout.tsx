import { useEffect } from 'react'
import { Header } from './Header'
import { AiChat } from './AiChat'
import { useWebSocket } from '../hooks/useWebSocket'
import { useChatStore } from '../stores/chatStore'
import styles from './Layout.module.css'

interface Props {
  children: React.ReactNode
}

export function Layout({ children }: Props) {
  useWebSocket()

  const isOpen = useChatStore((s) => s.isOpen)

  // Ctrl+K to toggle chat
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        useChatStore.getState().toggleChat()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div className={styles.layout}>
      <Header />
      <div className={styles.body}>
        <main className={`${styles.main} ${isOpen ? styles.withChat : ''}`}>
          {children}
        </main>
        {isOpen && (
          <>
            <div
              className={styles.chatBackdrop}
              onClick={() => useChatStore.getState().setOpen(false)}
            />
            <aside className={styles.chatPanel}>
              <AiChat />
            </aside>
          </>
        )}
      </div>
    </div>
  )
}
