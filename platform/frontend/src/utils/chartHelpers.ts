import { ColorType, CrosshairMode } from 'lightweight-charts'
import type { DeepPartial, ChartOptions } from 'lightweight-charts'
import type { Candle, HistoricalBar } from '../types/market'

// --- Chart theme ---------------------------------------------------------

export function buildChartOptions(
  overrides?: DeepPartial<ChartOptions>
): DeepPartial<ChartOptions> {
  return {
    layout: {
      background: { type: ColorType.Solid, color: '#1e222d' },
      textColor: '#d1d4dc',
    },
    grid: {
      vertLines: { color: '#2B2B43' },
      horzLines: { color: '#2B2B43' },
    },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2B2B43' },
    timeScale: {
      borderColor: '#2B2B43',
      timeVisible: true,
      secondsVisible: false,
      tickMarkFormatter: (t: number) => {
        const d = new Date(t * 1000)
        return d.toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        })
      },
    },
    localization: {
      timeFormatter: (t: number) => {
        const d = new Date(t * 1000)
        return d.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        })
      },
    },
    autoSize: true,
    ...overrides,
  }
}

// --- Historical data conversion ------------------------------------------

export function parseHistoricalBars(bars: HistoricalBar[]): Candle[] {
  return bars.map((b) => ({
    time: Math.floor(new Date(b.timestamp).getTime() / 1000),
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }))
}

// --- Timeframe bucket aggregation ----------------------------------------

export interface OhlcvBar {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export function bucketTime(unixSeconds: number, tfMinutes: number): number {
  const tf = tfMinutes * 60
  return Math.floor(unixSeconds / tf) * tf
}

export function aggregateCandles(
  candles: Candle[],
  tfMinutes: number
): OhlcvBar[] {
  if (tfMinutes <= 1) {
    return candles.map((c) => ({
      time: c.time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume ?? 0,
    }))
  }

  const buckets = new Map<number, OhlcvBar>()
  for (const c of candles) {
    const t = bucketTime(c.time, tfMinutes)
    const existing = buckets.get(t)
    if (!existing) {
      buckets.set(t, {
        time: t,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume ?? 0,
      })
    } else {
      existing.high = Math.max(existing.high, c.high)
      existing.low = Math.min(existing.low, c.low)
      existing.close = c.close
      existing.volume += c.volume ?? 0
    }
  }
  return Array.from(buckets.values()).sort((a, b) => a.time - b.time)
}

export function splitCandlesVolume(bars: OhlcvBar[]) {
  const candles = bars.map(({ time, open, high, low, close }) => ({
    time,
    open,
    high,
    low,
    close,
  }))
  const volumes = bars.map((b) => ({
    time: b.time,
    value: b.volume,
    color: b.close >= b.open ? '#26a69a80' : '#ef535080',
  }))
  return { candles, volumes }
}

// --- Real-time bar aggregation -------------------------------------------

export function mergeIncomingBar(
  existing: OhlcvBar | null,
  incoming: Candle,
  tfMinutes: number
): OhlcvBar {
  const t = tfMinutes <= 1 ? incoming.time : bucketTime(incoming.time, tfMinutes)
  if (existing && existing.time === t) {
    return {
      time: t,
      open: existing.open,
      high: Math.max(existing.high, incoming.high),
      low: Math.min(existing.low, incoming.low),
      close: incoming.close,
      volume: existing.volume + (incoming.volume ?? 0),
    }
  }
  return {
    time: t,
    open: incoming.open,
    high: incoming.high,
    low: incoming.low,
    close: incoming.close,
    volume: incoming.volume ?? 0,
  }
}

// --- Prediction timestamps -----------------------------------------------

export function buildPredictionPoints(
  values: number[],
  lastCandleTime: number,
  tfMinutes: number
): { time: number; value: number }[] {
  const tfSec = tfMinutes * 60
  return values.map((value, i) => ({
    time: lastCandleTime + (i + 1) * tfSec,
    value,
  }))
}

export function buildHorizontalLine(
  value: number,
  lastCandleTime: number,
  tfMinutes: number,
  extendBars = 30
): { time: number; value: number }[] {
  const tfSec = tfMinutes * 60
  return Array.from({ length: extendBars + 1 }, (_, i) => ({
    time: lastCandleTime + (i + 1) * tfSec,
    value,
  }))
}

// --- SMA series conversion -----------------------------------------------

/**
 * Build SMA line data from a flat array of values (one per historical candle).
 * Pairs each non-null value with the corresponding candle timestamp.
 */
export function buildSmaPoints(
  values: (number | null)[],
  candles: Candle[],
  tfMinutes: number
): { time: number; value: number }[] {
  const points: { time: number; value: number }[] = []
  const len = Math.min(values.length, candles.length)
  for (let i = 0; i < len; i++) {
    const v = values[i]
    if (v == null || isNaN(v)) continue
    points.push({ time: candles[i].time, value: v })
  }

  if (tfMinutes <= 1) {
    return points.sort((a, b) => a.time - b.time)
  }

  const buckets = new Map<number, number>()
  for (const p of points) {
    const t = bucketTime(p.time, tfMinutes)
    buckets.set(t, p.value) // last value in bucket wins
  }
  return Array.from(buckets.entries())
    .map(([time, value]) => ({ time, value }))
    .sort((a, b) => a.time - b.time)
}
