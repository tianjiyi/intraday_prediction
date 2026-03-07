import { create } from 'zustand'

export interface UiState {
  showPredictions: boolean
  showConfidence: boolean
  showIndicators: boolean
  showSMAs: boolean
  showRSI: boolean

  togglePredictions: () => void
  toggleConfidence: () => void
  toggleIndicators: () => void
  toggleSMAs: () => void
  toggleRSI: () => void
}

export const useUiStore = create<UiState>((set) => ({
  showPredictions: true,
  showConfidence: true,
  showIndicators: true,
  showSMAs: true,
  showRSI: true,

  togglePredictions: () => set((s) => ({ showPredictions: !s.showPredictions })),
  toggleConfidence: () => set((s) => ({ showConfidence: !s.showConfidence })),
  toggleIndicators: () => set((s) => ({ showIndicators: !s.showIndicators })),
  toggleSMAs: () => set((s) => ({ showSMAs: !s.showSMAs })),
  toggleRSI: () => set((s) => ({ showRSI: !s.showRSI })),
}))
