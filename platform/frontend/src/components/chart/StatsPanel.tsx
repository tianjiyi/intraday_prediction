import { useState } from 'react'
import type { Prediction } from '../../types/market'
import { formatPrice, formatPercent } from '../../utils/formatters'
import { useT } from '../../i18n'
import styles from './StatsPanel.module.css'

interface Props {
  symbol: string
  prediction: Prediction | null
  isStreaming: boolean
  width?: number
}

export function StatsPanel({ symbol, prediction: pred, isStreaming, width }: Props) {
  const t = useT()
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <button className={styles.expand} onClick={() => setCollapsed(false)}>
        {t('stats.title')}
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
    <aside className={styles.panel} style={width ? { width } : undefined}>
      <div className={styles.panelHeader}>
        <span className={styles.panelTitle}>{symbol}</span>
        <div className={styles.headerActions}>
          {isStreaming && <span className={styles.live}>{t('toolbar.live')}</span>}
          <button className={styles.collapse} onClick={() => setCollapsed(true)}>
            —
          </button>
        </div>
      </div>

      <section className={styles.section}>
        <Row label={t('stats.current')} value={formatPrice(pred?.current_close)} />
        <Row label={t('stats.predicted')} value={formatPrice(lastMean)} />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>{t('stats.probability')}</div>
        <Row
          label={t('stats.pUp30m')}
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
          label={t('stats.expReturn')}
          value={
            pred?.exp_ret_30m != null ? formatPercent(pred.exp_ret_30m) : '--'
          }
          className={
            (pred?.exp_ret_30m ?? 0) >= 0 ? styles.positive : styles.negative
          }
        />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>{t('stats.confidence')}</div>
        {(['p90', 'p75', 'p50', 'p25', 'p10'] as const).map((k) => (
          <Row
            key={k}
            label={k.toUpperCase()}
            value={formatPrice(pcts?.[k]?.at(-1))}
          />
        ))}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>{t('stats.indicators')}</div>
        <Row label="VWAP" value={formatPrice(pred?.current_vwap)} />
        <Row label="BB Upper" value={formatPrice(pred?.bollinger_bands?.upper)} />
        <Row label="BB Mid" value={formatPrice(pred?.bollinger_bands?.middle)} />
        <Row label="BB Lower" value={formatPrice(pred?.bollinger_bands?.lower)} />
      </section>

      {dc && (
        <section className={styles.section}>
          <div className={styles.sectionTitle}>{t('stats.daily')}</div>
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
            label={t('stats.trend')}
            value={dc.daily_trend || '--'}
            className={trendClass(dc.daily_trend)}
          />
        </section>
      )}

      {dc && (
        <section className={styles.section}>
          <div className={styles.sectionTitle}>{t('stats.keyLevels')}</div>
          {dc.prev_day_high != null && (
            <Row
              label={t('stats.prevHigh')}
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
              label={t('stats.prevLow')}
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
              label={t('stats.prevClose')}
              value={formatPrice(dc.prev_day_close)}
              className={
                currentPrice! > dc.prev_day_close
                  ? styles.positive
                  : styles.negative
              }
            />
          )}
          {dc.three_day_high != null && (
            <Row label={t('stats.3dHigh')} value={formatPrice(dc.three_day_high)} />
          )}
          {dc.three_day_low != null && (
            <Row label={t('stats.3dLow')} value={formatPrice(dc.three_day_low)} />
          )}
        </section>
      )}

      <section className={styles.section}>
        <div className={styles.sectionTitle}>{t('stats.model')}</div>
        <Row
          label={t('stats.name')}
          value={pred?.model_name ?? '--'}
          valueStyle={{ fontSize: 10 }}
        />
        <Row label={t('stats.bars')} value={String(pred?.data_bars_count ?? '--')} />
        <Row label={t('stats.samples')} value={String(pred?.n_samples ?? '--')} />
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
