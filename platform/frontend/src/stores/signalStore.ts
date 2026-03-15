import { create } from 'zustand'
import type { Signal } from '../types/market'

interface SignalState {
  signals: Signal[]
  setSignals: (signals: Signal[]) => void
  addSignal: (signal: Signal) => void
  clearSignals: () => void
}

export const useSignalStore = create<SignalState>((set) => ({
  signals: [],
  setSignals: (signals) => set({ signals }),
  addSignal: (signal) =>
    set((s) => ({ signals: [...s.signals, signal] })),
  clearSignals: () => set({ signals: [] }),
}))
