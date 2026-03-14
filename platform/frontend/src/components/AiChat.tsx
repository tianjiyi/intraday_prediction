import { useState, useRef, useEffect } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useMarketStore } from '../stores/marketStore'
import { useT } from '../i18n'
import { AiChatMessage } from './AiChatMessage'
import styles from './AiChat.module.css'

export function AiChat() {
  const t = useT()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const messages = useChatStore((s) => s.messages)
  const isLoading = useChatStore((s) => s.isLoading)
  const isOpen = useChatStore((s) => s.isOpen)
  const setOpen = useChatStore((s) => s.setOpen)
  const sendMessage = useChatStore((s) => s.sendMessage)
  const symbol = useMarketStore((s) => s.symbol)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (isOpen) inputRef.current?.focus()
  }, [isOpen])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || isLoading) return
    setInput('')
    sendMessage(text, symbol)
  }

  if (!isOpen) return null

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>{t('chat.title')}</span>
        <button className={styles.closeBtn} onClick={() => setOpen(false)} aria-label="Close chat">
          ✕
        </button>
      </div>

      <div className={styles.messages}>
        {messages.map((msg) => (
          <AiChatMessage key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className={`${styles.message} ${styles.assistant}`}>
            <div className={styles.bubble}>
              <span className={styles.typing}>
                <span /><span /><span />
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className={styles.inputArea} onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          className={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={t('chat.placeholder')}
          disabled={isLoading}
        />
        <button className={styles.sendBtn} type="submit" disabled={isLoading || !input.trim()}>
          {t('chat.send')}
        </button>
      </form>
    </div>
  )
}
