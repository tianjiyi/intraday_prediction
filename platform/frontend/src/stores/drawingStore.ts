import { create } from 'zustand'
import type { ChartDrawing, DrawingTool } from '../types/drawing'

interface DragState {
  drawingId: string
  grabType: string // 'body' | 'start' | 'end' | 'top' | 'bottom'
  startTime: number
  startPrice: number
  original: ChartDrawing
}

interface DrawingState {
  drawings: ChartDrawing[]
  activeTool: DrawingTool
  pendingPoint: { time: number; price: number } | null
  dragState: DragState | null

  setActiveTool: (tool: DrawingTool) => void
  addDrawing: (drawing: ChartDrawing) => void
  updateDrawing: (id: string, updates: Partial<ChartDrawing>) => void
  removeDrawing: (id: string) => void
  clearAll: () => void
  setPendingPoint: (point: { time: number; price: number } | null) => void
  setDragState: (state: DragState | null) => void
}

export const useDrawingStore = create<DrawingState>((set) => ({
  drawings: [],
  activeTool: 'cursor',
  pendingPoint: null,
  dragState: null,

  setActiveTool: (tool: DrawingTool) =>
    set({ activeTool: tool, pendingPoint: null }),

  addDrawing: (drawing: ChartDrawing) =>
    set((state) => ({ drawings: [...state.drawings, drawing] })),

  updateDrawing: (id: string, updates: Partial<ChartDrawing>) =>
    set((state) => ({
      drawings: state.drawings.map((d) =>
        d.id === id ? { ...d, ...updates } : d
      ),
    })),

  removeDrawing: (id: string) =>
    set((state) => ({ drawings: state.drawings.filter((d) => d.id !== id) })),

  clearAll: () => set({ drawings: [] }),

  setPendingPoint: (point: { time: number; price: number } | null) =>
    set({ pendingPoint: point }),

  setDragState: (dragState: DragState | null) => set({ dragState }),
}))
