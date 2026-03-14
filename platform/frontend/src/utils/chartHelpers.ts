import { ColorType, CrosshairMode } from 'lightweight-charts'
import type { DeepPartial, ChartOptions } from 'lightweight-charts'
import type { Candle, HistoricalBar } from '../types/market'

// --- Chart theme ---------------------------------------------------------

export function buildChartOptions(
  overrides?: DeepPartial<ChartOptions>
): DeepPartial<ChartOptions> {
  return {
    layout: {
      background: { type: ColorType.Solid, color: '#0f1218' },
      textColor: '#d1d4dc',
    },
    grid: {
      vertLines: { color: '#1c2030' },
      horzLines: { color: '#1c2030' },
    },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: {
      borderColor: '#1c2030',
      scaleMargins: { top: 0.05, bottom: 0.05 },
      minimumWidth: 80,
    },
    timeScale: {
      borderColor: '#1c2030',
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
  return bars
    .map((b) => ({
      time: Math.floor(new Date(b.timestamp).getTime() / 1000),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      volume: b.volume,
    }))
    .filter((c) => !isNaN(c.time))
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
    value: b.volume ?? 0,
    color: b.close >= b.open ? '#22d1a0' : '#f7525f',
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

// --- Session VWAP + sigma bands (Pine-style) --------------------------------

export interface VwapPoint {
  time: number
  vwap: number
  std: number
  upper1: number; lower1: number
  upper2: number; lower2: number
}

function dayOfBar(unixSec: number): number {
  // Eastern time date as YYYYMMDD integer for session boundary detection
  // Using Intl to get Eastern time components (handles DST automatically)
  const d = new Date(unixSec * 1000)
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(d)
  const y = Number(parts.find(p => p.type === 'year')!.value)
  const m = Number(parts.find(p => p.type === 'month')!.value)
  const day = Number(parts.find(p => p.type === 'day')!.value)
  return y * 10000 + m * 100 + day
}

export function computeSessionVwap(bars: OhlcvBar[]): VwapPoint[] {
  const results: VwapPoint[] = []
  let cumV = 0
  let cumPV = 0
  let cumPV2 = 0
  let prevDay = -1

  for (const bar of bars) {
    const day = dayOfBar(bar.time)
    if (day !== prevDay) {
      // Session reset
      cumV = 0
      cumPV = 0
      cumPV2 = 0
      prevDay = day
    }

    const src = (bar.high + bar.low + bar.close) / 3 // HLC3
    const vol = bar.volume || 0
    cumV += vol
    cumPV += src * vol
    cumPV2 += src * src * vol

    if (cumV === 0) {
      results.push({
        time: bar.time,
        vwap: src, std: 0,
        upper1: src, lower1: src,
        upper2: src, lower2: src,
      })
      continue
    }

    const vwap = cumPV / cumV
    const variance = Math.max(0, cumPV2 / cumV - vwap * vwap)
    const std = Math.sqrt(variance)

    results.push({
      time: bar.time,
      vwap,
      std,
      upper1: vwap + std,
      lower1: vwap - std,
      upper2: vwap + 2 * std,
      lower2: vwap - 2 * std,
    })
  }

  return results
}

// --- RSI calculation (Wilder's smoothing) -----------------------------------

export function computeRsi(
  bars: OhlcvBar[],
  period = 14
): { time: number; value: number }[] {
  if (bars.length < period + 1) return []

  const results: { time: number; value: number }[] = []

  // Calculate initial average gain/loss from first `period` changes
  let avgGain = 0
  let avgLoss = 0
  for (let i = 1; i <= period; i++) {
    const change = bars[i].close - bars[i - 1].close
    if (change > 0) avgGain += change
    else avgLoss -= change
  }
  avgGain /= period
  avgLoss /= period

  const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss)
  results.push({ time: bars[period].time, value: rsi })

  // Wilder's smoothing for remaining bars
  for (let i = period + 1; i < bars.length; i++) {
    const change = bars[i].close - bars[i - 1].close
    const gain = change > 0 ? change : 0
    const loss = change < 0 ? -change : 0
    avgGain = (avgGain * (period - 1) + gain) / period
    avgLoss = (avgLoss * (period - 1) + loss) / period
    const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss)
    results.push({ time: bars[i].time, value: rsi })
  }

  return results
}
