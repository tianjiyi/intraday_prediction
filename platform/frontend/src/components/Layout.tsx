import { useEffect } from 'react'
import { Header } from './Header'
import { AiChat } from './AiChat'
import { useWebSocket } from '../hooks/useWebSocket'
import { useChatStore } from '../stores/chatStore'
import { usePanelResize } from '../hooks/usePanelResize'
import styles from './Layout.module.css'

interface Props {
  children: React.ReactNode
}

export function Layout({ children }: Props) {
  useWebSocket()

  const isOpen = useChatStore((s) => s.isOpen)

  const chatResize = usePanelResize({
    storageKey: 'chatPanelWidth',
    defaultSize: 380,
    minSize: 280,
    maxSize: 600,
    direction: 'col',
    invert: true,
  })

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
            <div
              className={styles.chatDivider}
              onMouseDown={chatResize.onMouseDown}
            />
            <aside className={styles.chatPanel} style={{ width: chatResize.size }}>
              <AiChat />
            </aside>
          </>
        )}
      </div>
    </div>
  )
}
