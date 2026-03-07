import { create } from 'zustand'
import type { MarketPulse, MacroTapeItem, Mover, Theme, CatalystEvent } from '../types/landing'

interface LandingState {
  pulse: MarketPulse | null
  macroTape: MacroTapeItem[]
  gainers: Mover[]
  losers: Mover[]
  themes: Theme[]
  catalysts: CatalystEvent[]
  loading: boolean
  error: string | null

  setPulse: (p: MarketPulse) => void
  setMacroTape: (items: MacroTapeItem[]) => void
  setMovers: (gainers: Mover[], losers: Mover[]) => void
  setThemes: (t: Theme[]) => void
  setCatalysts: (c: CatalystEvent[]) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
}

export const useLandingStore = create<LandingState>((set) => ({
  pulse: null,
  macroTape: [],
  gainers: [],
  losers: [],
  themes: [],
  catalysts: [],
  loading: false,
  error: null,

  setPulse: (pulse) => set({ pulse }),
  setMacroTape: (macroTape) => set({ macroTape }),
  setMovers: (gainers, losers) => set({ gainers, losers }),
  setThemes: (themes) => set({ themes }),
  setCatalysts: (catalysts) => set({ catalysts }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}))
