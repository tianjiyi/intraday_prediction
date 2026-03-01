import { useState } from 'react'
import type { Prediction } from '../../types/market'
import { formatPrice, formatPercent } from '../../utils/formatters'
import styles from './StatsPanel.module.css'

interface Props {
  symbol: string
  prediction: Prediction | null
  isStreaming: boolean
}

export function StatsPanel({ symbol, prediction: pred, isStreaming }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <button className={styles.expand} onClick={() => setCollapsed(false)}>
        Stats
      </button>
    )
  }

  const lastMean = pred?.mean_path?.at(-1)
  const pcts = pred?.percentiles
  const dc = pred?.daily_context
  const currentPrice = pred?.current_close

  function trendClass(trend?: string | null) {
    if (!trend) return ''
    if (trend.includes('Bullish')) return styles.positive
    if (trend.includes('Bearish')) return styles.negative
    return ''
  }

  return (
    <aside className={styles.panel}>
      <div className={styles.panelHeader}>
        <span className={styles.panelTitle}>{symbol}</span>
        <div className={styles.headerActions}>
          {isStreaming && <span className={styles.live}>LIVE</span>}
          <button className={styles.collapse} onClick={() => setCollapsed(true)}>
            —
          </button>
        </div>
      </div>

      <section className={styles.section}>
        <Row label="Current" value={formatPrice(pred?.current_close)} />
        <Row label="Predicted" value={formatPrice(lastMean)} />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>Probability</div>
        <Row
          label="P(Up 30m)"
          value={
            pred?.p_up_30m != null
              ? `${(pred.p_up_30m * 100).toFixed(1)}%`
              : '--'
          }
          className={
            (pred?.p_up_30m ?? 0.5) > 0.5 ? styles.positive : styles.negative
          }
        />
        <div className={styles.probBar}>
          <div
            className={styles.probFill}
            style={{
              width: `${((pred?.p_up_30m ?? 0) * 100).toFixed(1)}%`,
              background:
                (pred?.p_up_30m ?? 0.5) > 0.6
                  ? 'var(--green)'
                  : (pred?.p_up_30m ?? 0.5) < 0.4
                    ? 'var(--red)'
                    : '#FF9800',
            }}
          />
        </div>
        <Row
          label="Exp Return"
          value={
            pred?.exp_ret_30m != null ? formatPercent(pred.exp_ret_30m) : '--'
          }
          className={
            (pred?.exp_ret_30m ?? 0) >= 0 ? styles.positive : styles.negative
          }
        />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>Confidence</div>
        {(['p90', 'p75', 'p50', 'p25', 'p10'] as const).map((k) => (
          <Row
            key={k}
            label={k.toUpperCase()}
            value={formatPrice(pcts?.[k]?.at(-1))}
          />
        ))}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>Indicators</div>
        <Row label="VWAP" value={formatPrice(pred?.current_vwap)} />
        <Row label="BB Upper" value={formatPrice(pred?.bollinger_bands?.upper)} />
        <Row label="BB Mid" value={formatPrice(pred?.bollinger_bands?.middle)} />
        <Row label="BB Lower" value={formatPrice(pred?.bollinger_bands?.lower)} />
      </section>

      {dc && (
        <section className={styles.section}>
          <div className={styles.sectionTitle}>Daily</div>
          {dc.daily_sma_5 != null && (
            <Row
              label="SMA 5"
              value={`${formatPrice(dc.daily_sma_5)} ${currentPrice! > dc.daily_sma_5 ? '↑' : '↓'}`}
              className={
                currentPrice! > dc.daily_sma_5
                  ? styles.positive
                  : styles.negative
              }
            />
          )}
          {dc.daily_sma_21 != null && (
            <Row
              label="SMA 21"
              value={`${formatPrice(dc.daily_sma_21)} ${currentPrice! > dc.daily_sma_21 ? '↑' : '↓'}`}
              className={
                currentPrice! > dc.daily_sma_21
                  ? styles.positive
                  : styles.negative
              }
            />
          )}
          {dc.daily_sma_233 != null && (
            <Row
              label="SMA 233"
              value={`${formatPrice(dc.daily_sma_233)} ${currentPrice! > dc.daily_sma_233 ? '↑' : '↓'}`}
              className={
                currentPrice! > dc.daily_sma_233
                  ? styles.positive
                  : styles.negative
              }
            />
          )}
          {dc.daily_rsi != null && (
            <Row label="RSI" value={dc.daily_rsi.toFixed(1)} />
          )}
          {dc.daily_cci != null && (
            <Row label="CCI" value={dc.daily_cci.toFixed(1)} />
          )}
          <Row
            label="Trend"
            value={dc.daily_trend || '--'}
            className={trendClass(dc.daily_trend)}
          />
        </section>
      )}

      {dc && (
        <section className={styles.section}>
          <div className={styles.sectionTitle}>Key Levels</div>
          {dc.prev_day_high != null && (
            <Row
              label="Prev High"
              value={formatPrice(dc.prev_day_high)}
              className={
                currentPrice! > dc.prev_day_high
                  ? styles.positive
                  : styles.negative
              }
            />
          )}
          {dc.prev_day_low != null && (
            <Row
              label="Prev Low"
              value={formatPrice(dc.prev_day_low)}
              className={
                currentPrice! < dc.prev_day_low
                  ? styles.negative
                  : styles.positive
              }
            />
          )}
          {dc.prev_day_close != null && (
            <Row
              label="Prev Close"
              value={formatPrice(dc.prev_day_close)}
              className={
                currentPrice! > dc.prev_day_close
                  ? styles.positive
                  : styles.negative
              }
            />
          )}
          {dc.three_day_high != null && (
            <Row label="3D High" value={formatPrice(dc.three_day_high)} />
          )}
          {dc.three_day_low != null && (
            <Row label="3D Low" value={formatPrice(dc.three_day_low)} />
          )}
        </section>
      )}

      <section className={styles.section}>
        <div className={styles.sectionTitle}>Model</div>
        <Row
          label="Name"
          value={pred?.model_name ?? '--'}
          valueStyle={{ fontSize: 10 }}
        />
        <Row label="Bars" value={String(pred?.data_bars_count ?? '--')} />
        <Row label="Samples" value={String(pred?.n_samples ?? '--')} />
      </section>
    </aside>
  )
}

function Row({
  label,
  value,
  className,
  valueStyle,
}: {
  label: string
  value: string | undefined
  className?: string
  valueStyle?: React.CSSProperties
}) {
  return (
    <div className={styles.row}>
      <span className={styles.label}>{label}</span>
      <span
        className={`${styles.value} ${className || ''}`}
        style={valueStyle}
      >
        {value ?? '--'}
      </span>
    </div>
  )
}
