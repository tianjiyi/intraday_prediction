import type { ChatMessage } from '../types/chat'
import styles from './AiChat.module.css'

interface Props {
  message: ChatMessage
}

export function AiChatMessage({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}>
      <div className={styles.bubble}>
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className={styles.toolCalls}>
            {message.toolCalls.map((tc, i) => (
              <span key={i} className={styles.toolBadge}>
                {toolIcon(tc.tool)} {toolLabel(tc.tool)}
              </span>
            ))}
          </div>
        )}
        <div
          className={styles.content}
          dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
        />
      </div>
    </div>
  )
}

function toolIcon(tool: string): string {
  const icons: Record<string, string> = {
    search_news: '🔍',
    search_twitter: '🐦',
    get_polymarket_predictions: '📊',
    run_kronos_prediction: '🔮',
    get_technical_analysis: '📈',
    search_memory: '🧠',
    store_memory: '💾',
    get_trending_sectors: '🔥',
    draw_on_chart: '✏️',
  }
  return icons[tool] || '⚡'
}

function toolLabel(tool: string): string {
  return tool.replace(/_/g, ' ').replace(/^get /, '')
}

function renderMarkdown(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
}
