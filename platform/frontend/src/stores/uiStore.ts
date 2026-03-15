import { create } from 'zustand'

export type Locale = 'en' | 'zh'

export interface UiState {
  showPredictions: boolean
  showConfidence: boolean
  showSMAs: boolean
  showRSI: boolean
  showSignals: boolean
  dayTradingMode: boolean
  locale: Locale

  togglePredictions: () => void
  toggleConfidence: () => void
  toggleSMAs: () => void
  toggleRSI: () => void
  toggleSignals: () => void
  toggleDayTradingMode: () => void
  setLocale: (locale: Locale) => void
}

function loadLocale(): Locale {
  try {
    const saved = localStorage.getItem('locale')
    if (saved === 'en' || saved === 'zh') return saved
  } catch { /* ignore */ }
  return 'en'
}

export const useUiStore = create<UiState>((set) => ({
  showPredictions: true,
  showConfidence: true,
  showSMAs: true,
  showRSI: true,
  showSignals: true,
  dayTradingMode: true,
  locale: loadLocale(),

  togglePredictions: () => set((s) => ({ showPredictions: !s.showPredictions })),
  toggleConfidence: () => set((s) => ({ showConfidence: !s.showConfidence })),
  toggleSMAs: () => set((s) => ({ showSMAs: !s.showSMAs })),
  toggleRSI: () => set((s) => ({ showRSI: !s.showRSI })),
  toggleSignals: () => set((s) => ({ showSignals: !s.showSignals })),
  toggleDayTradingMode: () => set((s) => ({ dayTradingMode: !s.dayTradingMode })),
  setLocale: (locale) => {
    try { localStorage.setItem('locale', locale) } catch { /* ignore */ }
    set({ locale })
  },
}))
