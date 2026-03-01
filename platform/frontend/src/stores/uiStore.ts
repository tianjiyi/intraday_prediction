import { create } from 'zustand'

export interface UiState {
  showPredictions: boolean
  showConfidence: boolean
  showIndicators: boolean
  showSMAs: boolean

  togglePredictions: () => void
  toggleConfidence: () => void
  toggleIndicators: () => void
  toggleSMAs: () => void
}

export const useUiStore = create<UiState>((set) => ({
  showPredictions: true,
  showConfidence: true,
  showIndicators: true,
  showSMAs: true,

  togglePredictions: () => set((s) => ({ showPredictions: !s.showPredictions })),
  toggleConfidence: () => set((s) => ({ showConfidence: !s.showConfidence })),
  toggleIndicators: () => set((s) => ({ showIndicators: !s.showIndicators })),
  toggleSMAs: () => set((s) => ({ showSMAs: !s.showSMAs })),
}))
