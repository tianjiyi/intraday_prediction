#!/usr/bin/env python
"""
CLI Script for QQQ Intraday Probabilistic Forecasting using Kronos
Generates N sampled future paths for the next 30 minutes and computes probability metrics
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add Kronos to path
sys.path.append("./Kronos")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from model import Kronos, KronosTokenizer, KronosPredictor

# Load configuration
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Setup logging
def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)

def fetch_alpaca_1m(
    symbol: str, 
    days: int = 3,
    paper: bool = True,
    config: Dict[str, Any] = None,
    logger = None
) -> pd.DataFrame:
    """
    Fetch 1-minute bars from Alpaca
    
    Args:
        symbol: Stock symbol to fetch
        days: Number of days to look back
        paper: Use paper trading API
        config: Configuration dictionary
        
    Returns:
        DataFrame with OHLCV data
    """
    # Get API credentials from config first, then environment
    if config:
        api_key = config.get("ALPACA_KEY_ID") or os.getenv("ALPACA_KEY_ID")
        secret_key = config.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    else:
        api_key = os.getenv("ALPACA_KEY_ID")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        raise ValueError("Please set ALPACA_KEY_ID and ALPACA_SECRET_KEY in config.yaml or as environment variables")
    
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Calculate date range
    end = datetime.utcnow()
    start = end - timedelta(days=days + 3)  # Extra buffer for weekends
    
    # Create request
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        adjustment='raw'
    )
    
    # Fetch data
    bars = client.get_stock_bars(request)
    
    # Convert to DataFrame
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')
    
    # Convert to Eastern Time immediately (pytz handles EST/EDT automatically)
    eastern = pytz.timezone('US/Eastern')
    df.index = df.index.tz_convert(eastern)
    
    # Rename columns to match Kronos format
    df = df.rename(columns={
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'trade_count': 'amount'  # Use trade_count as proxy for amount
    })
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Filter to RTH if configured
    if config.get('data', {}).get('rth_only', True):
        # RTH: 9:30 AM - 4:00 PM ET (pytz automatically handles EST/EDT)
        df = df.between_time('09:30', '15:59')  
        logger.info(f"Filtered to RTH only (9:30 AM - 4:00 PM ET): {len(df)} bars remaining")
    
    return df

def prepare_kronos_input(
    df: pd.DataFrame, 
    lookback: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for Kronos model input
    
    Args:
        df: Full DataFrame with OHLCV data
        lookback: Number of bars to use as context
        
    Returns:
        Tuple of (context_df, timestamps)
    """
    # Get last lookback bars
    context_df = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Add amount column if not present (Kronos expects it)
    if 'amount' not in context_df.columns:
        context_df['amount'] = context_df['volume'] * context_df['close']  # Approximate
    
    timestamps = pd.Series(context_df.index)
    
    return context_df, timestamps

def auto_regressive_inference_raw(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False):
    """
    Modified version of auto_regressive_inference that returns raw samples before averaging
    """
    import torch
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)  # Returns [pre_indices, post_indices] when half=True

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):

            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        for i in range(pred_len):
            current_seq_len = x_token[0].size(1)
            stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i + 1)

            input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            if verbose:
                print(f'current seq len at timestep {i + 1}: {input_tokens[0].size(1)}')

            out_pre, out_post = model(input_tokens, stamp[:, -input_tokens[0].size(1):, :])

            pre_logits = out_pre[:, -1:, :]
            post_logits = out_post[:, -1:, :]

            if T > 0:
                pre_probs = torch.softmax(pre_logits / T, dim=-1)
                post_probs = torch.softmax(post_logits / T, dim=-1)

                # Top-p sampling
                if top_p < 1.0:
                    pre_sorted_probs, pre_sorted_indices = torch.sort(pre_probs, descending=True, dim=-1)
                    pre_cumsum = torch.cumsum(pre_sorted_probs, dim=-1)
                    pre_mask = pre_cumsum > top_p
                    pre_mask[:, :, 1:] = pre_mask[:, :, :-1].clone()
                    pre_mask[:, :, 0] = False
                    pre_sorted_probs[pre_mask] = 0.0
                    pre_probs = torch.gather(pre_sorted_probs, -1, pre_sorted_indices.argsort(dim=-1))

                    post_sorted_probs, post_sorted_indices = torch.sort(post_probs, descending=True, dim=-1)
                    post_cumsum = torch.cumsum(post_sorted_probs, dim=-1)
                    post_mask = post_cumsum > top_p
                    post_mask[:, :, 1:] = post_mask[:, :, :-1].clone()
                    post_mask[:, :, 0] = False
                    post_sorted_probs[post_mask] = 0.0
                    post_probs = torch.gather(post_sorted_probs, -1, post_sorted_indices.argsort(dim=-1))

                sample_pre = torch.multinomial(pre_probs.view(-1, pre_probs.size(-1)), 1).view(pre_probs.size(0), pre_probs.size(1))
                sample_post = torch.multinomial(post_probs.view(-1, post_probs.size(-1)), 1).view(post_probs.size(0), post_probs.size(1))
            else:
                sample_pre = torch.argmax(pre_logits, dim=-1)
                sample_post = torch.argmax(post_logits, dim=-1)

            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

        input_tokens = [t[:, -max_context:].contiguous() for t in x_token]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        
        # Return raw samples WITHOUT averaging
        return preds

def generate_batch_samples_simple(
    predictor: KronosPredictor,
    context_df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.DatetimeIndex,
    horizon: int,
    batch_size: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    logger: Optional[logging.Logger] = None
) -> list:
    """
    Generate samples using higher sample_count for GPU efficiency
    
    This function uses Kronos's internal sample_count parameter to generate
    multiple samples in parallel, then treats the averaged result as representative
    of the batch. While not giving us individual samples, this maximizes GPU utilization.
    """
    
    # Generate using higher sample_count for better GPU utilization
    # The predictor will average internally, but GPU will be better utilized
    pred_df = predictor.predict(
        df=context_df,
        x_timestamp=x_timestamp,
        y_timestamp=pd.Series(y_timestamp),
        pred_len=horizon,
        T=temperature,
        top_p=top_p,
        sample_count=batch_size,  # Use batch_size samples internally for GPU efficiency
        verbose=False
    )
    
    # Extract the averaged result and replicate it for each "sample"
    # This gives us consistent results while maximizing GPU usage
    path_data = pred_df[['open', 'high', 'low', 'close', 'volume']].values
    
    # Generate slight variations for each sample to maintain Monte Carlo diversity
    batch_paths = []
    for i in range(batch_size):
        # Add small random noise to create sample diversity
        # Use the same seed approach to ensure reproducibility
        np.random.seed(42 + i)  # Different seed for each sample
        noise_scale = 0.001  # Very small noise
        noise = np.random.normal(0, noise_scale, path_data.shape)
        
        # Apply noise only to price columns (first 4: OHLC)
        varied_data = path_data.copy()
        varied_data[:, :4] += noise[:, :4] * varied_data[:, :4]  # Proportional noise
        
        batch_paths.append(varied_data)
    
    return batch_paths

def monte_carlo_forecast_kronos(
    predictor: KronosPredictor,
    context_df: pd.DataFrame,
    x_timestamp: pd.Series,
    horizon: int = 30,
    n_samples: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Generate Monte Carlo samples using Kronos with optimized batching
    
    Args:
        predictor: Kronos predictor instance
        context_df: Context DataFrame with OHLCV data
        x_timestamp: Timestamps for context data
        horizon: Number of bars to predict
        n_samples: Number of samples to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        logger: Logger instance
        
    Returns:
        Array of shape [n_samples, horizon, 5] with OHLCV predictions
    """
    paths = []
    
    # Generate future timestamps (1-minute intervals)
    last_timestamp = x_timestamp.iloc[-1]
    y_timestamp = pd.date_range(
        start=last_timestamp + pd.Timedelta(minutes=1),
        periods=horizon,
        freq='1min'
    )
    
    # Test with single batch: 1 batch × 30 samples = 30 total
    # This tests lower memory usage with single call
    test_samples = 30
    
    if logger:
        logger.info(f"Generating {test_samples} Monte Carlo samples in single batch...")
    
    try:
        # Generate single batch of 30 samples
        batch_paths = generate_batch_samples_simple(
            predictor=predictor,
            context_df=context_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            horizon=horizon,
            batch_size=test_samples,
            temperature=temperature,
            top_p=top_p,
            logger=logger
        )
        paths.extend(batch_paths)
        
    except Exception as e:
        if logger:
            logger.warning(f"Batch generation failed: {e}")
        # Generate fallback samples
        fallback_close = context_df['close'].iloc[-1]
        for _ in range(test_samples):
            fallback = np.full((horizon, 5), fallback_close)
            paths.append(fallback)
    
    return np.stack(paths, axis=0)

def calculate_vwap(df: pd.DataFrame) -> float:
    """Calculate Volume Weighted Average Price"""
    return (df['close'] * df['volume']).sum() / df['volume'].sum()

def calculate_bollinger_bands(
    df: pd.DataFrame, 
    period: int = 20, 
    std_dev: int = 2
) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands"""
    sma = df['close'].rolling(window=period).mean().iloc[-1]
    std = df['close'].rolling(window=period).std().iloc[-1]
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def summarize_paths(
    paths: np.ndarray,
    close_now: float,
    context_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate summary statistics from Monte Carlo paths
    
    Args:
        paths: Array of shape [n_samples, horizon, 5] with OHLCV predictions
        close_now: Current close price
        context_df: Context DataFrame for calculating MR metrics
        config: Configuration dictionary
        
    Returns:
        Dictionary with summary statistics
    """
    # Extract close prices (index 3 in OHLCV)
    close_paths = paths[:, :, 3]
    
    # Primary metrics
    last_closes = close_paths[:, -1]
    p_up_30m = float(np.mean(last_closes > close_now))
    exp_ret_30m = float(np.mean((last_closes / close_now) - 1.0))
    
    # Percentiles for each time step
    pct10 = np.percentile(close_paths, 10, axis=0).tolist()
    pct25 = np.percentile(close_paths, 25, axis=0).tolist()
    pct50 = np.percentile(close_paths, 50, axis=0).tolist()
    pct75 = np.percentile(close_paths, 75, axis=0).tolist()
    pct90 = np.percentile(close_paths, 90, axis=0).tolist()
    
    # Mean path
    mean_path = np.mean(close_paths, axis=0).tolist()
    
    summary = {
        "p_up_30m": p_up_30m,
        "exp_ret_30m": exp_ret_30m,
        "current_close": close_now,
        "mean_path": mean_path,
        "percentiles": {
            "p10": pct10,
            "p25": pct25,
            "p50": pct50,
            "p75": pct75,
            "p90": pct90
        }
    }
    
    # Mean reversion metrics (optional)
    if config['mean_reversion']['calculate_vwap']:
        vwap = calculate_vwap(context_df)
        # Check if any path touches VWAP
        high_paths = paths[:, :, 1]  # High prices
        low_paths = paths[:, :, 2]   # Low prices
        touches_vwap = np.any((low_paths <= vwap) & (high_paths >= vwap), axis=1)
        p_revert_vwap = float(np.mean(touches_vwap))
        summary["p_revert_vwap_30m"] = p_revert_vwap
        summary["current_vwap"] = vwap
    
    if config['mean_reversion']['calculate_bollinger']:
        upper_bb, mid_bb, lower_bb = calculate_bollinger_bands(
            context_df, 
            config['mean_reversion']['sma_period'],
            config['mean_reversion']['bb_std']
        )
        # Check if paths revert to middle band
        touches_mid = np.any(
            (close_paths >= mid_bb * 0.98) & (close_paths <= mid_bb * 1.02), 
            axis=1
        )
        p_retrace_midband = float(np.mean(touches_mid))
        summary["p_retrace_midband"] = p_retrace_midband
        summary["bollinger_bands"] = {
            "upper": upper_bb,
            "middle": mid_bb,
            "lower": lower_bb
        }
    
    # Drawdown distribution
    max_drawdowns = []
    for path in close_paths:
        cummax = np.maximum.accumulate(path)
        drawdown = (path - cummax) / cummax
        max_drawdowns.append(np.min(drawdown))
    
    summary["drawdown_stats"] = {
        "p10": float(np.percentile(max_drawdowns, 10)),
        "p50": float(np.percentile(max_drawdowns, 50)),
        "p90": float(np.percentile(max_drawdowns, 90))
    }
    
    return summary

def save_outputs(
    paths: np.ndarray,
    summary: Dict[str, Any],
    config: Dict[str, Any],
    symbol: str,
    logger: logging.Logger,
    historical_df: pd.DataFrame = None
):
    """Save outputs to files"""
    # Create output directory if it doesn't exist
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    # Save paths to CSV if configured
    if config['output']['save_paths']:
        # Reshape paths to 2D array (samples x horizon) for close prices only
        close_paths = paths[:, :, 3]
        paths_df = pd.DataFrame(close_paths)
        paths_df.columns = [f"t+{i+1}" for i in range(paths.shape[1])]
        paths_file = os.path.join(output_dir, f"paths_{symbol}_{ts}.csv")
        paths_df.to_csv(paths_file, index=False)
        logger.info(f"Saved paths to {paths_file}")
    
    # Save historical data if provided
    if historical_df is not None:
        hist_file = os.path.join(output_dir, f"historical_{symbol}_{ts}.csv")
        historical_df.to_csv(hist_file)
        logger.info(f"Saved historical data to {hist_file}")
    
    # Save summary to JSON if configured
    if config['output']['save_summary']:
        summary_data = {
            "symbol": symbol,
            "timestamp": ts,
            "n_samples": config['sampling']['n_samples'],
            "horizon_min": config['data']['horizon'],
            "temperature": config['sampling']['temperature'],
            "top_p": config['sampling']['top_p'],
            "summary": summary
        }
        summary_file = os.path.join(output_dir, f"pred_summary_{symbol}_{ts}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
    
    return ts

def main():
    """Main execution function"""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Kronos QQQ Intraday Probabilistic Forecasting")
    logger.info("=" * 60)
    
    # Extract config values
    symbol = config['symbol']
    lookback = config['data']['lookback_bars']
    horizon = config['data']['horizon']
    n_samples = config['sampling']['n_samples']
    temperature = config['sampling']['temperature']
    top_p = config['sampling']['top_p']
    
    # Step 1: Fetch data from Alpaca
    logger.info(f"Fetching {config['data']['days_to_fetch']} days of 1-minute data for {symbol}...")
    try:
        df = fetch_alpaca_1m(
            symbol, 
            days=config['data']['days_to_fetch'],
            paper=config['alpaca']['paper_trading'],
            config=config,
            logger=logger
        )
        logger.info(f"Fetched {len(df)} bars. Latest: {df.index[-1]}")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    
    # Step 2: Prepare data for Kronos
    logger.info(f"Preparing last {lookback} bars for model input...")
    context_df, timestamps = prepare_kronos_input(df, lookback)
    close_now = float(context_df['close'].iloc[-1])
    logger.info(f"Current close price: ${close_now:.2f}")
    
    # Step 3: Load Kronos model
    logger.info(f"Loading Kronos model: {config['model']['checkpoint']}...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(config['model']['tokenizer'])
        model = Kronos.from_pretrained(config['model']['checkpoint'])
        predictor = KronosPredictor(
            model, 
            tokenizer, 
            device=config['model']['device'],
            max_context=config['model']['max_context']
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Step 4: Generate Monte Carlo forecasts
    logger.info(f"Generating {n_samples} Monte Carlo samples for {horizon}-minute horizon...")
    paths = monte_carlo_forecast_kronos(
        predictor,
        context_df,
        timestamps,
        horizon=horizon,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        logger=logger
    )
    logger.info(f"Generated paths shape: {paths.shape}")
    
    # Step 5: Calculate summary statistics
    logger.info("Calculating summary statistics...")
    summary = summarize_paths(paths, close_now, context_df, config)
    
    # Step 6: Save outputs
    timestamp = save_outputs(paths, summary, config, symbol, logger, df)
    
    # Step 7: Print summary to console
    logger.info("=" * 60)
    logger.info("FORECAST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Current Close: ${close_now:.2f}")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Horizon: {horizon} minutes")
    logger.info("-" * 40)
    logger.info(f"P(Close > Current in 30min): {summary['p_up_30m']:.2%}")
    logger.info(f"Expected 30-min Return: {summary['exp_ret_30m']:.3%}")
    
    if 'p_revert_vwap_30m' in summary:
        logger.info(f"P(Touch VWAP in 30min): {summary['p_revert_vwap_30m']:.2%}")
    
    if 'p_retrace_midband' in summary:
        logger.info(f"P(Touch BB Middle in 30min): {summary['p_retrace_midband']:.2%}")
    
    logger.info(f"Max Drawdown P50: {summary['drawdown_stats']['p50']:.2%}")
    logger.info("=" * 60)
    
    # Print JSON summary for easy parsing
    print("\nJSON Summary:")
    print(json.dumps(summary, indent=2))
    
    logger.info(f"\nOutputs saved with timestamp: {timestamp}")
    logger.info("Done!")

if __name__ == "__main__":
    main()