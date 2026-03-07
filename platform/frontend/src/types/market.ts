export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface Prediction {
  mean_path: number[]
  percentiles: {
    p90: number[]
    p75: number[]
    p50: number[]
    p25: number[]
    p10: number[]
  }
  current_close: number
  p_up_30m: number
  exp_ret_30m: number
  current_vwap: number
  bollinger_bands: { upper: number; middle: number; lower: number }
  sma_5_series: (number | null)[]
  sma_21_series: (number | null)[]
  sma_233_series: (number | null)[]
  daily_context?: DailyContext
  data_bars_count: number
  n_samples: number
  model_name: string
  timestamp: string
  asset_type: string
  rth_only: boolean
}

export interface DailyContext {
  daily_sma_5: number | null
  daily_sma_21: number | null
  daily_sma_233: number | null
  daily_rsi: number | null
  daily_cci: number | null
  daily_trend: string
  prev_day_high: number | null
  prev_day_low: number | null
  prev_day_close: number | null
  three_day_high: number | null
  three_day_low: number | null
}

export interface DayTradingVwap {
  value: number
  std: number
  upper1: number
  lower1: number
  upper2: number
  lower2: number
}

export interface DayTradingChartState {
  enabled: boolean
  timeframe_minutes: number
  vwap: DayTradingVwap | null
}

export interface HistoricalBar {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface InitialDataResponse {
  historical: HistoricalBar[]
  prediction: Prediction
  timestamp: string
}
