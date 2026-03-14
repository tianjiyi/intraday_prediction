import { useEffect, useRef, type RefObject } from 'react'
import type { ISeriesApi, IPriceLine } from 'lightweight-charts'
import type { TradingChartHandle } from '../components/chart/TradingChart'
import type { ChartDrawing } from '../types/drawing'
import { useDrawingStore } from '../stores/drawingStore'
import { TrendLinePrimitive } from '../components/chart/primitives/TrendLinePrimitive'
import { ZonePrimitive } from '../components/chart/primitives/ZonePrimitive'

type RenderHandle = IPriceLine | TrendLinePrimitive | ZonePrimitive

/** Simple hash of drawing data fields that affect rendering */
function drawingHash(d: ChartDrawing): string {
  return `${d.price}|${d.startTime}|${d.startPrice}|${d.endTime}|${d.endPrice}|${d.priceHigh}|${d.priceLow}|${d.color}|${d.label}`
}

export function useDrawings(chartRef: RefObject<TradingChartHandle | null>) {
  const renderedRef = useRef<Map<string, RenderHandle>>(new Map())
  const hashRef = useRef<Map<string, string>>(new Map())

  useEffect(() => {
    const unsub = useDrawingStore.subscribe((state) => {
      const series = chartRef.current?.getSeries()
      if (!series) return

      const currentIds = new Set(state.drawings.map((d) => d.id))
      const rendered = renderedRef.current
      const hashes = hashRef.current

      // Remove drawings that no longer exist in state
      for (const [id, handle] of rendered) {
        if (!currentIds.has(id)) {
          removeFromChart(series, handle)
          rendered.delete(id)
          hashes.delete(id)
        }
      }

      // Add or update drawings
      for (const drawing of state.drawings) {
        const hash = drawingHash(drawing)
        const existingHandle = rendered.get(drawing.id)

        if (!existingHandle) {
          // New drawing — add
          const handle = addToChart(series, drawing)
          if (handle) {
            rendered.set(drawing.id, handle)
            hashes.set(drawing.id, hash)
          }
        } else if (hashes.get(drawing.id) !== hash) {
          // Drawing data changed — update in place
          updateInChart(series, existingHandle, drawing, rendered)
          hashes.set(drawing.id, hash)
        }
      }
    })

    return () => {
      unsub()
      const series = chartRef.current?.getSeries()
      if (series) {
        for (const handle of renderedRef.current.values()) {
          removeFromChart(series, handle)
        }
      }
      renderedRef.current.clear()
      hashRef.current.clear()
    }
  }, [chartRef])
}

function addToChart(
  series: ISeriesApi<'Candlestick'>,
  drawing: ChartDrawing
): RenderHandle | null {
  switch (drawing.type) {
    case 'hline': {
      if (drawing.price == null) return null
      return series.createPriceLine({
        price: drawing.price,
        color: drawing.color,
        lineWidth: 2,
        lineStyle: 0,
        axisLabelVisible: true,
        title: drawing.label || '',
      })
    }
    case 'trendline': {
      if (
        drawing.startTime == null ||
        drawing.startPrice == null ||
        drawing.endTime == null ||
        drawing.endPrice == null
      )
        return null
      const primitive = new TrendLinePrimitive({
        id: drawing.id,
        startTime: drawing.startTime,
        startPrice: drawing.startPrice,
        endTime: drawing.endTime,
        endPrice: drawing.endPrice,
        color: drawing.color,
        label: drawing.label,
      })
      series.attachPrimitive(primitive)
      return primitive
    }
    case 'zone': {
      if (drawing.priceHigh == null || drawing.priceLow == null) return null
      const primitive = new ZonePrimitive({
        id: drawing.id,
        priceHigh: drawing.priceHigh,
        priceLow: drawing.priceLow,
        startTime: drawing.startTime,
        endTime: drawing.endTime,
        color: drawing.color,
        label: drawing.label,
      })
      series.attachPrimitive(primitive)
      return primitive
    }
    default:
      return null
  }
}

function updateInChart(
  series: ISeriesApi<'Candlestick'>,
  handle: RenderHandle,
  drawing: ChartDrawing,
  rendered: Map<string, RenderHandle>
) {
  if (handle instanceof TrendLinePrimitive && drawing.type === 'trendline') {
    handle.updateData({
      id: drawing.id,
      startTime: drawing.startTime!,
      startPrice: drawing.startPrice!,
      endTime: drawing.endTime!,
      endPrice: drawing.endPrice!,
      color: drawing.color,
      label: drawing.label,
    })
  } else if (handle instanceof ZonePrimitive && drawing.type === 'zone') {
    handle.updateData({
      id: drawing.id,
      priceHigh: drawing.priceHigh!,
      priceLow: drawing.priceLow!,
      startTime: drawing.startTime,
      endTime: drawing.endTime,
      color: drawing.color,
      label: drawing.label,
    })
  } else {
    // hline — no update API, recreate
    removeFromChart(series, handle)
    const newHandle = addToChart(series, drawing)
    if (newHandle) rendered.set(drawing.id, newHandle)
  }
}

function removeFromChart(
  series: ISeriesApi<'Candlestick'>,
  handle: RenderHandle
) {
  if (handle instanceof TrendLinePrimitive || handle instanceof ZonePrimitive) {
    series.detachPrimitive(handle)
  } else {
    series.removePriceLine(handle)
  }
}
