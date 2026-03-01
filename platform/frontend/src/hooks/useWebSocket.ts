import { useEffect, useRef } from 'react'
import { useMarketStore } from '../stores/marketStore'
import { useNewsStore } from '../stores/newsStore'

const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws'
const WS_URL =
  import.meta.env.MODE === 'development'
    ? `ws://${window.location.hostname}:5000/ws`
    : `${wsProto}://${window.location.host}/ws`

const RECONNECT_DELAY = 3000

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    function connect() {
      if (wsRef.current?.readyState === WebSocket.OPEN) return

      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        useMarketStore.getState().setWsConnected(true)
      }

      ws.onclose = () => {
        useMarketStore.getState().setWsConnected(false)
        wsRef.current = null
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY)
      }

      ws.onerror = () => {
        ws.close()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleMessage(data)
        } catch {
          // ignore non-JSON messages
        }
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [])
}

function handleMessage(data: Record<string, unknown>) {
  const type = data.type as string

  switch (type) {
    case 'connected':
      break

    case 'bar_update': {
      const bar = parseBar(data)
      if (bar) useMarketStore.getState().updateBar(bar)
      break
    }

    case 'bar_complete': {
      const bar = parseBar(data)
      if (bar) useMarketStore.getState().addBar(bar)
      break
    }

    case 'prediction_update': {
      const pred = data.prediction as Record<string, unknown> | undefined
      if (pred) {
        useMarketStore.getState().setPrediction(pred as never)
      }
      const hist = data.historical as unknown[] | undefined
      if (hist) {
        useMarketStore.getState().setHistoricalData(hist as never)
      }
      break
    }

    case 'news_update': {
      const items = data.items as unknown[] | undefined
      if (items && items.length > 0) {
        useNewsStore.getState().addItems(items as never)
      }
      break
    }

    case 'stream_started':
      useMarketStore.getState().setStreaming(true)
      break

    case 'stream_stopped':
      useMarketStore.getState().setStreaming(false)
      break

    case 'stream_status': {
      const connected = data.connected as boolean
      useMarketStore.getState().setStreaming(connected)
      break
    }

    default:
      break
  }
}

function parseBar(data: Record<string, unknown>) {
  const o = Number(data.open)
  const h = Number(data.high)
  const l = Number(data.low)
  const c = Number(data.close)
  const v = Number(data.volume ?? 0)
  const ts = data.timestamp as string | undefined

  if (!ts || isNaN(o)) return null

  return {
    time: Math.floor(new Date(ts).getTime() / 1000) as never,
    open: o,
    high: h,
    low: l,
    close: c,
    volume: v,
  }
}
