import { useState } from 'react'

export function OptionsDashboardPage() {
  const [symbol, setSymbol] = useState('QQQ')
  const [expiration, setExpiration] = useState('')
  const [loading, setLoading] = useState(false)
  const [html, setHtml] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    setLoading(true)
    setError(null)
    setHtml(null)
    try {
      const res = await fetch('/api/options/dashboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: symbol.toUpperCase(), expiration }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }
      const content = await res.text()
      setHtml(content)
    } catch (e: any) {
      setError(e.message ?? 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '16px', height: '100%', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          placeholder="Ticker (e.g. QQQ)"
          style={{
            background: '#1a1a2e',
            border: '1px solid #333',
            color: '#e0e0e0',
            padding: '6px 10px',
            borderRadius: '4px',
            width: '120px',
            fontSize: '14px',
          }}
        />
        <input
          type="date"
          value={expiration}
          onChange={(e) => setExpiration(e.target.value)}
          style={{
            background: '#1a1a2e',
            border: '1px solid #333',
            color: '#e0e0e0',
            padding: '6px 10px',
            borderRadius: '4px',
            fontSize: '14px',
            colorScheme: 'dark',
          }}
        />
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            background: loading ? '#333' : '#2962ff',
            color: '#fff',
            border: 'none',
            padding: '6px 18px',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 600,
          }}
        >
          {loading ? 'Generating...' : 'Generate'}
        </button>
        {loading && (
          <span style={{ color: '#888', fontSize: '13px' }}>
            Loading options data...
          </span>
        )}
      </div>

      {error && (
        <div style={{ color: '#f44336', fontSize: '13px', padding: '8px', background: '#2a1010', borderRadius: '4px' }}>
          {error}
        </div>
      )}

      {html && (
        <iframe
          srcDoc={html}
          style={{ width: '100%', height: '90vh', border: 'none', borderRadius: '4px', flex: 1 }}
          title="Options Dashboard"
        />
      )}
    </div>
  )
}
