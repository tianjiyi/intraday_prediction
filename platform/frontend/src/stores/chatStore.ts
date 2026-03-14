import { create } from 'zustand'
import type { ChatMessage } from '../types/chat'
import { sendChatMessage, formatHistory } from '../api/chat'
import { useNewsStore } from './newsStore'
import { useMarketStore } from './marketStore'
import { useUiStore } from './uiStore'
import { useDrawingStore } from './drawingStore'
import { captureChartScreenshot, getVisibleTimeRange } from '../utils/chartRegistry'

interface ChatState {
  messages: ChatMessage[]
  sessionId: string | null
  isLoading: boolean
  isOpen: boolean

  toggleChat: () => void
  setOpen: (open: boolean) => void
  sendMessage: (text: string, symbol?: string) => Promise<void>
  addMessage: (msg: ChatMessage) => void
  clearMessages: () => void
}

let msgId = 0
function nextId() {
  return `msg-${Date.now()}-${++msgId}`
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [
    {
      id: 'welcome',
      role: 'assistant',
      content:
        "Hi! I'm your trading assistant. Ask me anything about markets, news, or technical analysis.",
      timestamp: new Date().toISOString(),
    },
  ],
  sessionId: null,
  isLoading: false,
  isOpen: window.innerWidth > 900,

  toggleChat: () => set((s) => ({ isOpen: !s.isOpen })),
  setOpen: (open) => set({ isOpen: open }),

  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),

  clearMessages: () =>
    set({
      messages: [
        {
          id: 'welcome',
          role: 'assistant',
          content:
            "Hi! I'm your trading assistant. Ask me anything about markets, news, or technical analysis.",
          timestamp: new Date().toISOString(),
        },
      ],
    }),

  sendMessage: async (text, symbol) => {
    const userMsg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }

    set((s) => ({
      messages: [...s.messages, userMsg],
      isLoading: true,
    }))

    try {
      const history = formatHistory(get().messages)
      const selectedSector = useNewsStore.getState().selectedSector
      const { timeframe, dayTradingVwap } = useMarketStore.getState()
      const { dayTradingMode } = useUiStore.getState()
      const dtEnabled = dayTradingMode && [1, 5, 15].includes(timeframe)
      const drawings = useDrawingStore.getState().drawings
      const chartScreenshot = captureChartScreenshot()
      const data = await sendChatMessage({
        message: text,
        symbol: symbol || 'QQQ',
        chat_history: history,
        session_id: get().sessionId || undefined,
        selected_sector: selectedSector || undefined,
        chart_screenshot: chartScreenshot,
        chart_state: {
          visible_time_range: getVisibleTimeRange() || null,
          day_trading: {
            enabled: dtEnabled,
            timeframe_minutes: timeframe,
            vwap: dtEnabled ? dayTradingVwap : null,
          },
          drawings: drawings.map((d) => ({
            type: d.type,
            label: d.label,
            price: d.price,
            priceHigh: d.priceHigh,
            priceLow: d.priceLow,
            startPrice: d.startPrice,
            endPrice: d.endPrice,
            color: d.color,
            source: d.source,
          })),
        },
      })

      const assistantMsg: ChatMessage = {
        id: nextId(),
        role: 'assistant',
        content: data.response,
        toolCalls: data.tool_calls?.map((tc) => ({
          ...tc,
          status: 'done' as const,
        })),
        timestamp: data.timestamp || new Date().toISOString(),
      }

      set((s) => ({
        messages: [...s.messages, assistantMsg],
        sessionId: data.session_id || s.sessionId,
        isLoading: false,
      }))
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: nextId(),
        role: 'assistant',
        content: `Error: ${err instanceof Error ? err.message : 'Something went wrong'}`,
        timestamp: new Date().toISOString(),
      }
      set((s) => ({
        messages: [...s.messages, errorMsg],
        isLoading: false,
      }))
    }
  },
}))
