import type { IChartApi } from 'lightweight-charts'

let _chart: IChartApi | null = null

export function registerChart(chart: IChartApi) {
  _chart = chart
}

export function unregisterChart() {
  _chart = null
}

export function captureChartScreenshot(): string | undefined {
  if (!_chart) return undefined
  try {
    // true = include top layer (primitives/drawings)
    const canvas = _chart.takeScreenshot(true)
    return canvas.toDataURL('image/png').split(',')[1]  // base64 without data:image/png;base64, prefix
  } catch {
    return undefined
  }
}

/** Get the visible time range of the chart as Unix timestamps */
export function getVisibleTimeRange(): { from: number; to: number } | undefined {
  if (!_chart) return undefined
  try {
    const range = _chart.timeScale().getVisibleRange()
    if (!range) return undefined
    return {
      from: range.from as unknown as number,
      to: range.to as unknown as number,
    }
  } catch {
    return undefined
  }
}
