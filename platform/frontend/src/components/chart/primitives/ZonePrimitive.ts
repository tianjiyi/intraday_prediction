import type {
  ISeriesPrimitive,
  SeriesAttachedParameter,
  Time,
  IChartApi,
  ISeriesApi,
  SeriesType,
  PrimitiveHoveredItem,
} from 'lightweight-charts'

export interface ZoneData {
  id?: string
  priceHigh: number
  priceLow: number
  startTime?: number  // optional: if set, box is bounded left
  endTime?: number    // optional: if set, box is bounded right
  color: string
  label?: string
  dashed?: boolean
}

const EDGE_THRESHOLD = 6 // px

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export class ZonePrimitive implements ISeriesPrimitive<Time> {
  _data: ZoneData
  private _chart: IChartApi | null = null
  private _series: ISeriesApi<SeriesType> | null = null
  private _requestUpdate: (() => void) | null = null

  // Cache pixel coords for hit testing
  private _pyHigh = 0
  private _pyLow = 0
  private _pxLeft = 0
  private _pxRight = 0
  private _bounded = false
  private _coordsValid = false

  constructor(data: ZoneData) {
    this._data = data
  }

  attached(param: SeriesAttachedParameter<Time>) {
    this._chart = param.chart
    this._series = param.series as ISeriesApi<SeriesType>
    this._requestUpdate = param.requestUpdate
  }

  detached() {
    this._chart = null
    this._series = null
    this._requestUpdate = null
    this._coordsValid = false
  }

  updateAllViews() {}

  updateData(data: ZoneData) {
    this._data = data
    this._coordsValid = false
    this._requestUpdate?.()
  }

  private _computeCoords(): boolean {
    if (!this._series) return false
    const yHigh = this._series.priceToCoordinate(this._data.priceHigh)
    const yLow = this._series.priceToCoordinate(this._data.priceLow)
    if (yHigh === null || yLow === null) return false
    this._pyHigh = yHigh
    this._pyLow = yLow

    // Compute horizontal bounds if times are set
    if (this._data.startTime != null && this._data.endTime != null && this._chart) {
      const ts = this._chart.timeScale()
      const xLeft = ts.timeToCoordinate(this._data.startTime as unknown as Time)
      const xRight = ts.timeToCoordinate(this._data.endTime as unknown as Time)
      if (xLeft !== null && xRight !== null) {
        this._pxLeft = Math.min(xLeft, xRight)
        this._pxRight = Math.max(xLeft, xRight)
        this._bounded = true
      } else {
        this._bounded = false
      }
    } else {
      this._bounded = false
    }

    this._coordsValid = true
    return true
  }

  paneViews() {
    const self = this
    return [
      {
        renderer() {
          return {
            draw(target: any) {
              target.useBitmapCoordinateSpace(
                ({
                  context,
                  bitmapSize,
                  horizontalPixelRatio: hpr,
                  verticalPixelRatio: vpr,
                }: any) => {
                  if (!self._computeCoords()) return

                  const byHigh = self._pyHigh * vpr
                  const byLow = self._pyLow * vpr

                  let bxLeft: number
                  let bxRight: number
                  if (self._bounded) {
                    bxLeft = self._pxLeft * hpr
                    bxRight = self._pxRight * hpr
                  } else {
                    bxLeft = 0
                    bxRight = bitmapSize.width
                  }
                  const boxWidth = bxRight - bxLeft

                  // Semi-transparent fill
                  context.fillStyle = hexToRgba(self._data.color, 0.15)
                  context.fillRect(bxLeft, byHigh, boxWidth, byLow - byHigh)

                  // Border
                  const dashPattern = self._data.dashed
                    ? [6 * hpr, 4 * hpr]
                    : [4 * hpr, 4 * hpr]
                  context.setLineDash(dashPattern)
                  context.strokeStyle = hexToRgba(self._data.color, 0.6)
                  context.lineWidth = 1 * hpr

                  if (self._bounded) {
                    // Draw full rectangle
                    context.strokeRect(bxLeft, byHigh, boxWidth, byLow - byHigh)
                  } else {
                    // Full-width zone: just top and bottom lines
                    context.beginPath()
                    context.moveTo(bxLeft, byHigh)
                    context.lineTo(bxRight, byHigh)
                    context.stroke()

                    context.beginPath()
                    context.moveTo(bxLeft, byLow)
                    context.lineTo(bxRight, byLow)
                    context.stroke()
                  }

                  context.setLineDash([])

                  // Label
                  if (self._data.label) {
                    context.font = `${11 * vpr}px sans-serif`
                    context.fillStyle = self._data.color
                    context.textAlign = 'left'
                    context.fillText(
                      self._data.label,
                      bxLeft + 8 * hpr,
                      (byHigh + byLow) / 2 + 4 * vpr
                    )
                  }
                }
              )
            },
          }
        },

        hitTest(x: number, y: number): PrimitiveHoveredItem | null {
          if (!self._coordsValid) self._computeCoords()
          if (!self._coordsValid) return null

          const id = self._data.id || ''
          const { _pyHigh: yTop, _pyLow: yBot } = self

          // If bounded, check x is within box
          if (self._bounded) {
            if (x < self._pxLeft - EDGE_THRESHOLD || x > self._pxRight + EDGE_THRESHOLD) {
              return null
            }
          }

          // Check top edge
          if (Math.abs(y - yTop) <= EDGE_THRESHOLD) {
            return {
              cursorStyle: 'ns-resize',
              externalId: `${id}:top`,
              zOrder: 'normal',
            }
          }

          // Check bottom edge
          if (Math.abs(y - yBot) <= EDGE_THRESHOLD) {
            return {
              cursorStyle: 'ns-resize',
              externalId: `${id}:bottom`,
              zOrder: 'normal',
            }
          }

          // Check interior
          if (y >= yTop && y <= yBot) {
            return {
              cursorStyle: 'grab',
              externalId: `${id}:body`,
              zOrder: 'normal',
            }
          }

          return null
        },
      },
    ]
  }
}
