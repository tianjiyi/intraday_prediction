import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import type { MouseEventParams, Time, IChartApi, ISeriesApi } from 'lightweight-charts'
import type { TradingChartHandle } from '../components/chart/TradingChart'
import type { ChartDrawing } from '../types/drawing'
import { useDrawingStore } from '../stores/drawingStore'
import { TrendLinePrimitive } from '../components/chart/primitives/TrendLinePrimitive'
import { ZonePrimitive } from '../components/chart/primitives/ZonePrimitive'

let idCounter = 0
function nextDrawingId() {
  return `user-${Date.now()}-${++idCounter}`
}

const TRENDLINE_COLOR = '#ef5350'
const DEFAULT_COLOR = '#2962FF'
const HIT_THRESHOLD = 10 // CSS pixels

// Timestamp of last drawing completion — prevents drag handler from
// immediately grabbing a just-created drawing on the same click event.
let lastDrawFinishedAt = 0

// ── Geometry helpers ──

function distToSegment(
  px: number, py: number,
  x1: number, y1: number,
  x2: number, y2: number
): number {
  const dx = x2 - x1
  const dy = y2 - y1
  const lenSq = dx * dx + dy * dy
  if (lenSq === 0) return Math.hypot(px - x1, py - y1)
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq))
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
}

interface HitResult {
  drawingId: string
  grabType: string // 'body' | 'start' | 'end' | 'top' | 'bottom'
}

/** Our own hit testing — doesn't rely on LWC's hoveredObjectId */
function hitTestDrawings(
  x: number,
  y: number,
  drawings: ChartDrawing[],
  chart: IChartApi,
  series: ISeriesApi<'Candlestick'>
): HitResult | null {
  const ts = chart.timeScale()

  for (const d of drawings) {
    if (d.type === 'trendline') {
      const x1 = ts.timeToCoordinate(d.startTime as unknown as Time)
      const y1 = series.priceToCoordinate(d.startPrice!)
      const x2 = ts.timeToCoordinate(d.endTime as unknown as Time)
      const y2 = series.priceToCoordinate(d.endPrice!)
      if (x1 === null || y1 === null || x2 === null || y2 === null) continue

      // Check endpoints first
      if (Math.hypot(x - x1, y - y1) <= HIT_THRESHOLD) {
        return { drawingId: d.id, grabType: 'start' }
      }
      if (Math.hypot(x - x2, y - y2) <= HIT_THRESHOLD) {
        return { drawingId: d.id, grabType: 'end' }
      }
      // Check line body
      if (distToSegment(x, y, x1, y1, x2, y2) <= HIT_THRESHOLD) {
        return { drawingId: d.id, grabType: 'body' }
      }
    } else if (d.type === 'zone') {
      const yTop = series.priceToCoordinate(d.priceHigh!)
      const yBot = series.priceToCoordinate(d.priceLow!)
      if (yTop === null || yBot === null) continue

      // If bounded box, check x is within time range
      if (d.startTime != null && d.endTime != null) {
        const xLeft = ts.timeToCoordinate(d.startTime as unknown as Time)
        const xRight = ts.timeToCoordinate(d.endTime as unknown as Time)
        if (xLeft !== null && xRight !== null) {
          const minX = Math.min(xLeft, xRight)
          const maxX = Math.max(xLeft, xRight)
          if (x < minX - HIT_THRESHOLD || x > maxX + HIT_THRESHOLD) continue
        }
      }

      if (Math.abs(y - yTop) <= HIT_THRESHOLD / 2) {
        return { drawingId: d.id, grabType: 'top' }
      }
      if (Math.abs(y - yBot) <= HIT_THRESHOLD / 2) {
        return { drawingId: d.id, grabType: 'bottom' }
      }
      if (y >= yTop && y <= yBot) {
        return { drawingId: d.id, grabType: 'body' }
      }
    } else if (d.type === 'hline') {
      const yLine = series.priceToCoordinate(d.price!)
      if (yLine === null) continue
      if (Math.abs(y - yLine) <= HIT_THRESHOLD) {
        return { drawingId: d.id, grabType: 'body' }
      }
    }
  }
  return null
}

export function useDrawingInteraction(
  chartRef: RefObject<TradingChartHandle | null>
) {
  const previewRef = useRef<TrendLinePrimitive | ZonePrimitive | null>(null)

  // ── Drawing mode: click-to-draw with rubber-band preview ──
  useEffect(() => {
    const chart = chartRef.current?.getChart()
    const series = chartRef.current?.getSeries()
    if (!chart || !series) return

    let crosshairHandler: ((p: MouseEventParams<Time>) => void) | null = null

    function cleanupPreview() {
      if (previewRef.current) {
        series!.detachPrimitive(previewRef.current)
        previewRef.current = null
      }
      if (crosshairHandler) {
        chart!.unsubscribeCrosshairMove(crosshairHandler)
        crosshairHandler = null
      }
    }

    function handleClick(param: MouseEventParams<Time>) {
      const {
        activeTool,
        pendingPoint,
        setActiveTool,
        addDrawing,
        setPendingPoint,
      } = useDrawingStore.getState()

      if (activeTool === 'cursor') return
      if (!param.time || !param.point) return

      const price = series!.coordinateToPrice(param.point.y)
      if (price === null) return
      const time = param.time as unknown as number

      switch (activeTool) {
        case 'hline': {
          addDrawing({
            id: nextDrawingId(),
            type: 'hline',
            price: price as number,
            color: DEFAULT_COLOR,
            source: 'user',
          })
          lastDrawFinishedAt = Date.now()
          setActiveTool('cursor')
          break
        }

        case 'trendline': {
          if (!pendingPoint) {
            setPendingPoint({ time, price: price as number })
            const preview = new TrendLinePrimitive({
              startTime: time,
              startPrice: price as number,
              endTime: time,
              endPrice: price as number,
              color: TRENDLINE_COLOR,
              dashed: true,
            })
            series!.attachPrimitive(preview)
            previewRef.current = preview

            crosshairHandler = (moveParam) => {
              if (!moveParam.time || !moveParam.point) return
              const movePrice = series!.coordinateToPrice(moveParam.point.y)
              if (movePrice === null) return
              preview.updateData({
                startTime: time,
                startPrice: price as number,
                endTime: moveParam.time as unknown as number,
                endPrice: movePrice as number,
                color: TRENDLINE_COLOR,
                dashed: true,
              })
            }
            chart!.subscribeCrosshairMove(crosshairHandler)
          } else {
            cleanupPreview()
            addDrawing({
              id: nextDrawingId(),
              type: 'trendline',
              startTime: pendingPoint.time,
              startPrice: pendingPoint.price,
              endTime: time,
              endPrice: price as number,
              color: TRENDLINE_COLOR,
              source: 'user',
            })
            setPendingPoint(null)
            lastDrawFinishedAt = Date.now()
            setActiveTool('cursor')
          }
          break
        }

        case 'zone': {
          if (!pendingPoint) {
            setPendingPoint({ time, price: price as number })
            const preview = new ZonePrimitive({
              priceHigh: price as number,
              priceLow: price as number,
              startTime: time,
              endTime: time,
              color: DEFAULT_COLOR,
              dashed: true,
            })
            series!.attachPrimitive(preview)
            previewRef.current = preview

            crosshairHandler = (moveParam) => {
              if (!moveParam.point || !moveParam.time) return
              const movePrice = series!.coordinateToPrice(moveParam.point.y)
              if (movePrice === null) return
              const high = Math.max(price as number, movePrice as number)
              const low = Math.min(price as number, movePrice as number)
              ;(preview as ZonePrimitive).updateData({
                priceHigh: high,
                priceLow: low,
                startTime: time,
                endTime: moveParam.time as unknown as number,
                color: DEFAULT_COLOR,
                dashed: true,
              })
            }
            chart!.subscribeCrosshairMove(crosshairHandler)
          } else {
            cleanupPreview()
            const high = Math.max(pendingPoint.price, price as number)
            const low = Math.min(pendingPoint.price, price as number)
            addDrawing({
              id: nextDrawingId(),
              type: 'zone',
              priceHigh: high,
              priceLow: low,
              startTime: pendingPoint.time,
              endTime: time,
              color: DEFAULT_COLOR,
              source: 'user',
            })
            setPendingPoint(null)
            lastDrawFinishedAt = Date.now()
            setActiveTool('cursor')
          }
          break
        }
      }
    }

    chart.subscribeClick(handleClick)

    return () => {
      chart.unsubscribeClick(handleClick)
      cleanupPreview()
    }
  }, [chartRef])

  // ── Drag mode: reposition existing drawings ──
  // Uses direct geometric hit testing (not LWC's hoveredObjectId)
  useEffect(() => {
    const chart = chartRef.current?.getChart()
    const series = chartRef.current?.getSeries()
    if (!chart || !series) return

    let dragCrosshairHandler: ((p: MouseEventParams<Time>) => void) | null =
      null

    function handleDragClick(param: MouseEventParams<Time>) {
      const { activeTool, dragState, setDragState, updateDrawing, drawings } =
        useDrawingStore.getState()
      if (activeTool !== 'cursor') return

      if (dragState) {
        // Release drag
        setDragState(null)
        chart!.applyOptions({ handleScroll: true, handleScale: true })
        if (dragCrosshairHandler) {
          chart!.unsubscribeCrosshairMove(dragCrosshairHandler)
          dragCrosshairHandler = null
        }
        return
      }

      if (!param.point || !param.time) return

      // Skip if a drawing just completed on this same click event
      if (Date.now() - lastDrawFinishedAt < 100) return

      // Direct geometric hit testing
      const hit = hitTestDrawings(
        param.point.x,
        param.point.y,
        drawings,
        chart!,
        series!
      )
      if (!hit) return

      const drawing = drawings.find((d) => d.id === hit.drawingId)
      if (!drawing) return

      const price = series!.coordinateToPrice(param.point.y)
      if (price === null) return
      const time = param.time as unknown as number

      // Start drag
      setDragState({
        drawingId: hit.drawingId,
        grabType: hit.grabType,
        startTime: time,
        startPrice: price as number,
        original: { ...drawing },
      })
      chart!.applyOptions({ handleScroll: false, handleScale: false })

      // Track movement
      dragCrosshairHandler = (moveParam) => {
        if (!moveParam.point || !moveParam.time) return
        const movePrice = series!.coordinateToPrice(moveParam.point.y)
        if (movePrice === null) return
        const moveTime = moveParam.time as unknown as number

        const ds = useDrawingStore.getState().dragState
        if (!ds) return

        const orig = ds.original
        const dPrice = (movePrice as number) - ds.startPrice
        const dTime = moveTime - ds.startTime

        const updates: Partial<ChartDrawing> = {}

        if (orig.type === 'trendline') {
          if (ds.grabType === 'body') {
            updates.startTime = orig.startTime! + dTime
            updates.startPrice = orig.startPrice! + dPrice
            updates.endTime = orig.endTime! + dTime
            updates.endPrice = orig.endPrice! + dPrice
          } else if (ds.grabType === 'start') {
            updates.startTime = orig.startTime! + dTime
            updates.startPrice = orig.startPrice! + dPrice
          } else if (ds.grabType === 'end') {
            updates.endTime = orig.endTime! + dTime
            updates.endPrice = orig.endPrice! + dPrice
          }
        } else if (orig.type === 'zone') {
          if (ds.grabType === 'body') {
            updates.priceHigh = orig.priceHigh! + dPrice
            updates.priceLow = orig.priceLow! + dPrice
            // Move time bounds if bounded box
            if (orig.startTime != null && orig.endTime != null) {
              updates.startTime = orig.startTime + dTime
              updates.endTime = orig.endTime + dTime
            }
          } else if (ds.grabType === 'top') {
            updates.priceHigh = orig.priceHigh! + dPrice
          } else if (ds.grabType === 'bottom') {
            updates.priceLow = orig.priceLow! + dPrice
          }
        } else if (orig.type === 'hline') {
          updates.price = orig.price! + dPrice
        }

        updateDrawing(ds.drawingId, updates)
      }
      chart!.subscribeCrosshairMove(dragCrosshairHandler)
    }

    chart.subscribeClick(handleDragClick)

    return () => {
      chart.unsubscribeClick(handleDragClick)
      if (dragCrosshairHandler) {
        chart.unsubscribeCrosshairMove(dragCrosshairHandler)
      }
    }
  }, [chartRef])

  // ── Toggle chart scroll/scale + cursor based on active tool ──
  useEffect(() => {
    const unsub = useDrawingStore.subscribe((state, prevState) => {
      const chart = chartRef.current?.getChart()
      const series = chartRef.current?.getSeries()
      if (!chart) return

      const isDrawing = state.activeTool !== 'cursor'
      const wasDrawing = prevState.activeTool !== 'cursor'

      if (isDrawing !== wasDrawing) {
        chart.applyOptions({
          handleScroll: !isDrawing,
          handleScale: !isDrawing,
        })
        const container = (chart as any).chartElement?.()?.parentElement
        if (container) {
          container.style.cursor = isDrawing ? 'crosshair' : ''
        }
      }

      // Clean up preview when tool switches back to cursor mid-draw
      if (
        state.activeTool !== prevState.activeTool &&
        previewRef.current &&
        series
      ) {
        series.detachPrimitive(previewRef.current)
        previewRef.current = null
      }
    })
    return unsub
  }, [chartRef])
}
