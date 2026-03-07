import { create } from 'zustand'

export interface UiState {
  showPredictions: boolean
  showConfidence: boolean
  showSMAs: boolean
  showRSI: boolean
  dayTradingMode: boolean

  togglePredictions: () => void
  toggleConfidence: () => void
  toggleSMAs: () => void
  toggleRSI: () => void
  toggleDayTradingMode: () => void
}

export const useUiStore = create<UiState>((set) => ({
  showPredictions: true,
  showConfidence: true,
  showSMAs: true,
  showRSI: true,
  dayTradingMode: true,

  togglePredictions: () => set((s) => ({ showPredictions: !s.showPredictions })),
  toggleConfidence: () => set((s) => ({ showConfidence: !s.showConfidence })),
  toggleSMAs: () => set((s) => ({ showSMAs: !s.showSMAs })),
  toggleRSI: () => set((s) => ({ showRSI: !s.showRSI })),
  toggleDayTradingMode: () => set((s) => ({ dayTradingMode: !s.dayTradingMode })),
}))
