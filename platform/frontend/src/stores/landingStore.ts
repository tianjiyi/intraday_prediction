import { create } from 'zustand'
import type { MarketPulse, MacroTapeItem, Mover, Theme, CatalystEvent, TradeContext } from '../types/landing'

interface LandingState {
  pulse: MarketPulse | null
  macroTape: MacroTapeItem[]
  gainers: Mover[]
  losers: Mover[]
  themes: Theme[]
  catalysts: CatalystEvent[]
  tradeContext: TradeContext | null
  loading: boolean
  translating: boolean
  error: string | null

  setPulse: (p: MarketPulse) => void
  setMacroTape: (items: MacroTapeItem[]) => void
  setMovers: (gainers: Mover[], losers: Mover[]) => void
  setThemes: (t: Theme[]) => void
  setCatalysts: (c: CatalystEvent[]) => void
  setTradeContext: (tc: TradeContext) => void
  setLoading: (v: boolean) => void
  setTranslating: (v: boolean) => void
  setError: (e: string | null) => void
}

export const useLandingStore = create<LandingState>((set) => ({
  pulse: null,
  macroTape: [],
  gainers: [],
  losers: [],
  themes: [],
  catalysts: [],
  tradeContext: null,
  loading: false,
  translating: false,
  error: null,

  setPulse: (pulse) => set({ pulse }),
  setMacroTape: (macroTape) => set({ macroTape }),
  setMovers: (gainers, losers) => set({ gainers, losers }),
  setThemes: (themes) => set({ themes }),
  setCatalysts: (catalysts) => set({ catalysts }),
  setTradeContext: (tradeContext) => set({ tradeContext }),
  setLoading: (loading) => set({ loading }),
  setTranslating: (translating) => set({ translating }),
  setError: (error) => set({ error }),
}))
