import Markdown from 'react-markdown'
import type { ChatMessage } from '../types/chat'
import styles from './AiChat.module.css'

interface Props {
  message: ChatMessage
}

/** Strip DRAW_COMMAND blocks from display — they're executed, not shown */
function stripDrawCommands(text: string): string {
  return text.replace(/```DRAW_COMMAND[\s\S]*?```/g, '').trim()
}

export function AiChatMessage({ message }: Props) {
  const isUser = message.role === 'user'
  const content = isUser ? message.content : stripDrawCommands(message.content)

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
        <div className={`${styles.content} ${isUser ? '' : styles.markdown}`}>
          {isUser ? (
            content
          ) : (
            <Markdown>{content}</Markdown>
          )}
        </div>
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
