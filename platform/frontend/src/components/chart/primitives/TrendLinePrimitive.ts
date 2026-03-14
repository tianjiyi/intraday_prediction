import type {
  ISeriesPrimitive,
  SeriesAttachedParameter,
  Time,
  IChartApi,
  ISeriesApi,
  SeriesType,
  PrimitiveHoveredItem,
} from 'lightweight-charts'

export interface TrendLineData {
  id?: string
  startTime: number
  startPrice: number
  endTime: number
  endPrice: number
  color: string
  lineWidth?: number
  label?: string
  dashed?: boolean
}

const HIT_THRESHOLD = 8 // px
const ENDPOINT_RADIUS = 4 // px

function distToSegment(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  const dx = x2 - x1
  const dy = y2 - y1
  const lenSq = dx * dx + dy * dy
  if (lenSq === 0) return Math.hypot(px - x1, py - y1)
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq))
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
}

export class TrendLinePrimitive implements ISeriesPrimitive<Time> {
  _data: TrendLineData
  private _chart: IChartApi | null = null
  private _series: ISeriesApi<SeriesType> | null = null
  private _requestUpdate: (() => void) | null = null

  // Cache pixel coords for hit testing (computed during draw)
  private _px1 = 0
  private _py1 = 0
  private _px2 = 0
  private _py2 = 0
  private _coordsValid = false

  constructor(data: TrendLineData) {
    this._data = data
  }

  attached(param: SeriesAttachedParameter<Time>) {
    this._chart = param.chart as IChartApi
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

  updateData(data: TrendLineData) {
    this._data = data
    this._coordsValid = false
    this._requestUpdate?.()
  }

  private _computeCoords(): boolean {
    if (!this._chart || !this._series) return false
    const ts = this._chart.timeScale()
    const x1 = ts.timeToCoordinate(this._data.startTime as unknown as Time)
    const x2 = ts.timeToCoordinate(this._data.endTime as unknown as Time)
    const y1 = this._series.priceToCoordinate(this._data.startPrice)
    const y2 = this._series.priceToCoordinate(this._data.endPrice)
    if (x1 === null || x2 === null || y1 === null || y2 === null) return false
    this._px1 = x1
    this._py1 = y1
    this._px2 = x2
    this._py2 = y2
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
                  horizontalPixelRatio: hpr,
                  verticalPixelRatio: vpr,
                }: any) => {
                  if (!self._computeCoords()) return

                  const bx1 = self._px1 * hpr
                  const by1 = self._py1 * vpr
                  const bx2 = self._px2 * hpr
                  const by2 = self._py2 * vpr
                  const lw = (self._data.lineWidth || 2) * hpr

                  // Line
                  context.beginPath()
                  context.moveTo(bx1, by1)
                  context.lineTo(bx2, by2)
                  context.strokeStyle = self._data.color
                  context.lineWidth = lw
                  if (self._data.dashed) {
                    context.setLineDash([6 * hpr, 4 * hpr])
                  }
                  context.stroke()
                  context.setLineDash([])

                  // Endpoint circles
                  const r = ENDPOINT_RADIUS * hpr
                  for (const [bx, by] of [
                    [bx1, by1],
                    [bx2, by2],
                  ]) {
                    context.beginPath()
                    context.arc(bx, by, r, 0, Math.PI * 2)
                    context.fillStyle = self._data.color
                    context.fill()
                    context.beginPath()
                    context.arc(bx, by, r + 1 * hpr, 0, Math.PI * 2)
                    context.strokeStyle = self._data.color
                    context.lineWidth = 1 * hpr
                    context.stroke()
                  }

                  // Label at midpoint
                  if (self._data.label) {
                    const mx = (bx1 + bx2) / 2
                    const my = (by1 + by2) / 2
                    context.font = `${11 * vpr}px sans-serif`
                    context.fillStyle = self._data.color
                    context.textAlign = 'center'
                    context.fillText(
                      self._data.label,
                      mx,
                      my - 8 * vpr
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
          const { _px1: x1, _py1: y1, _px2: x2, _py2: y2 } = self

          // Check endpoints first (higher priority)
          const distStart = Math.hypot(x - x1, y - y1)
          if (distStart <= HIT_THRESHOLD) {
            return {
              cursorStyle: 'crosshair',
              externalId: `${id}:start`,
              zOrder: 'normal',
            }
          }

          const distEnd = Math.hypot(x - x2, y - y2)
          if (distEnd <= HIT_THRESHOLD) {
            return {
              cursorStyle: 'crosshair',
              externalId: `${id}:end`,
              zOrder: 'normal',
            }
          }

          // Check line body
          const distBody = distToSegment(x, y, x1, y1, x2, y2)
          if (distBody <= HIT_THRESHOLD) {
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
