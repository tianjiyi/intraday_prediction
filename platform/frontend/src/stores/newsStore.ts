import { create } from 'zustand'
import type { NewsItem, SectorSummary } from '../types/news'

interface NewsState {
  items: NewsItem[]
  sectors: SectorSummary[]
  unreadCount: number
  activeCategory: string

  setCategory: (cat: string) => void
  addItems: (items: NewsItem[]) => void
  setItems: (items: NewsItem[]) => void
  setSectors: (sectors: SectorSummary[]) => void
  resetUnread: () => void
}

export const useNewsStore = create<NewsState>((set, get) => ({
  items: [],
  sectors: [],
  unreadCount: 0,
  activeCategory: 'all',

  setCategory: (cat) => set({ activeCategory: cat }),

  addItems: (newItems) => {
    const existing = new Set(get().items.map((i) => i.id))
    const fresh = newItems.filter((i) => !existing.has(i.id))
    if (fresh.length === 0) return
    set((s) => ({
      items: [...fresh, ...s.items].slice(0, 200),
      unreadCount: s.unreadCount + fresh.length,
    }))
  },

  setItems: (items) => set({ items }),
  setSectors: (sectors) => set({ sectors }),
  resetUnread: () => set({ unreadCount: 0 }),
}))
