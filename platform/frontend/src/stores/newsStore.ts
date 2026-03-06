import { create } from 'zustand'
import type { NewsItem, SectorTrend } from '../types/news'

interface NewsState {
  items: NewsItem[]
  sectorTrends: SectorTrend[]
  unreadCount: number
  activeCategory: string
  selectedSector: string | null
  window: '1h' | '6h' | '24h'
  criticalOnly: boolean

  setCategory: (cat: string) => void
  addItems: (items: NewsItem[]) => void
  setItems: (items: NewsItem[]) => void
  setSectorTrends: (trends: SectorTrend[]) => void
  selectSector: (sector: string | null) => void
  setWindow: (w: '1h' | '6h' | '24h') => void
  setCriticalOnly: (v: boolean) => void
  resetUnread: () => void
}

export const useNewsStore = create<NewsState>((set, get) => ({
  items: [],
  sectorTrends: [],
  unreadCount: 0,
  activeCategory: 'all',
  selectedSector: null,
  window: '6h',
  criticalOnly: false,

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
  setSectorTrends: (trends) => set({ sectorTrends: trends }),

  selectSector: (sector) =>
    set((s) => ({
      selectedSector: s.selectedSector === sector ? null : sector,
    })),

  setWindow: (w) => set({ window: w }),
  setCriticalOnly: (v) => set({ criticalOnly: v }),
  resetUnread: () => set({ unreadCount: 0 }),
}))
