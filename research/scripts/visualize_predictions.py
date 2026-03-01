#!/usr/bin/env python
"""
Visualization Script for Kronos QQQ Predictions
Creates interactive charts similar to the Kronos BTC demo
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import argparse
from typing import Dict, Any, Optional
import pytz

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_next_trading_session(last_time: pd.Timestamp, horizon_minutes: int = 30) -> pd.DatetimeIndex:
    """
    Calculate the next trading session timestamps for predictions
    
    Args:
        last_time: Last timestamp from historical data
        horizon_minutes: Number of minutes to predict
        
    Returns:
        DatetimeIndex with prediction timestamps during next RTH
    """
    # Convert to ET timezone for market hours calculation
    et_tz = pytz.timezone('US/Eastern')
    if last_time.tz is None:
        last_time = last_time.tz_localize('UTC')
    last_time_et = last_time.tz_convert(et_tz)
    
    print(f"Last historical time (ET): {last_time_et}")
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open_hour = 9
    market_open_minute = 30
    market_close_hour = 16
    market_close_minute = 0
    
    # Determine next trading day
    next_trading_day = last_time_et.date()
    
    # If it's weekend or after market close, find next trading day
    while True:
        # Check if it's weekend
        if next_trading_day.weekday() >= 5:  # Saturday=5, Sunday=6
            next_trading_day += timedelta(days=1)
            continue
            
        # Check if we're past market close time for current day
        if next_trading_day == last_time_et.date():
            if (last_time_et.hour > market_close_hour or 
                (last_time_et.hour == market_close_hour and last_time_et.minute >= market_close_minute)):
                next_trading_day += timedelta(days=1)
                continue
        
        # Check for US market holidays (simplified - add more as needed)
        # Labor Day (first Monday in September), Memorial Day (last Monday in May), etc.
        if is_us_market_holiday(next_trading_day):
            next_trading_day += timedelta(days=1)
            continue
            
        break
    
    # Create prediction start time (market open of next trading day)
    pred_start = et_tz.localize(datetime.combine(
        next_trading_day, 
        datetime.min.time().replace(hour=market_open_hour, minute=market_open_minute)
    ))
    
    print(f"Next trading session starts (ET): {pred_start}")
    
    # Generate prediction timestamps
    pred_times_et = pd.date_range(
        start=pred_start,
        periods=horizon_minutes,
        freq='1min'
    )
    
    # Convert back to UTC for consistency
    pred_times_utc = pred_times_et.tz_convert('UTC')
    
    return pred_times_utc

def is_us_market_holiday(date_obj):
    """
    Check if a date is a US market holiday (simplified version)
    """
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    
    # Labor Day - first Monday in September
    if month == 9 and date_obj.weekday() == 0:
        # Calculate first Monday of September
        sep_1 = datetime(year, 9, 1)
        first_monday = 1 + (7 - sep_1.weekday()) % 7
        if date_obj.day == first_monday:
            return True
    
    # 2025 specific holidays (add more years/holidays as needed)
    if year == 2025:
        # Labor Day 2025 is September 1st (Monday)
        if month == 9 and day == 1:
            return True
        # New Year's Day
        if month == 1 and day == 1:
            return True
        # Independence Day
        if month == 7 and day == 4:
            return True
    
    return False

def load_prediction_data(
    summary_file: str, 
    paths_file: Optional[str] = None
) -> tuple:
    """
    Load prediction data from files
    
    Returns:
        Tuple of (summary_dict, paths_df)
    """
    # Load summary
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Load paths if available
    paths_df = None
    if paths_file and os.path.exists(paths_file):
        paths_df = pd.read_csv(paths_file)
    
    return summary_data, paths_df

def create_kronos_style_chart(
    historical_df: pd.DataFrame,
    summary_data: Dict[str, Any],
    paths_df: Optional[pd.DataFrame] = None,
    title: str = "QQQ Intraday Prediction"
) -> go.Figure:
    """
    Create an interactive chart similar to Kronos demo
    
    Args:
        historical_df: Historical OHLCV data
        summary_data: Prediction summary with statistics
        paths_df: Individual path predictions
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Extract data
    summary = summary_data['summary']
    current_close = summary['current_close']
    mean_path = summary['mean_path']
    percentiles = summary['percentiles']
    
    # Create timestamps for predictions
    last_time = historical_df.index[-1]
    
    # Calculate next trading session timestamps
    pred_times = get_next_trading_session(last_time, len(mean_path))
    
    print(f"Historical data ends at: {last_time}")
    print(f"Predictions start at: {pred_times[0]}")
    print(f"Predictions end at: {pred_times[-1]}")
    
    # Create single figure (no subplots for now to ensure predictions are visible)
    fig = go.Figure()
    
    # Historical candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=historical_df.index,
            open=historical_df['open'],
            high=historical_df['high'],
            low=historical_df['low'],
            close=historical_df['close'],
            name='Historical',
            increasing_line_color='rgba(0, 150, 0, 0.8)',
            decreasing_line_color='rgba(200, 0, 0, 0.8)',
            increasing_fillcolor='rgba(0, 150, 0, 0.3)',
            decreasing_fillcolor='rgba(200, 0, 0, 0.3)',
            showlegend=True
        )
    )
    
    # Add prediction start line as a vertical line trace
    y_min = min(historical_df['low'].min(), min(mean_path)) * 0.999
    y_max = max(historical_df['high'].max(), max(mean_path)) * 1.001
    fig.add_trace(
        go.Scatter(
            x=[last_time, last_time],
            y=[y_min, y_max],
            mode='lines',
            line=dict(width=3, dash='dash', color='red'),
            name='Market Close (Fri)',
            showlegend=True,
            hovertemplate='Market closed for Labor Day weekend<extra></extra>'
        )
    )
    
    # Add prediction start marker
    pred_y_min = min(mean_path) * 0.999
    pred_y_max = max(mean_path) * 1.001
    fig.add_trace(
        go.Scatter(
            x=[pred_times[0], pred_times[0]],
            y=[pred_y_min, pred_y_max],
            mode='lines',
            line=dict(width=3, dash='dash', color='green'),
            name='Market Open (Tue)',
            showlegend=True,
            hovertemplate='Market reopens Tuesday 9:30 AM ET<extra></extra>'
        )
    )
    
    # Add individual sample paths (if available, show subset for clarity)
    if paths_df is not None and len(paths_df) > 0:
        # Sample 5 paths for visualization (fewer but more visible)
        n_paths_to_show = min(5, len(paths_df))
        sample_indices = np.linspace(0, len(paths_df)-1, n_paths_to_show, dtype=int)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, idx in enumerate(sample_indices):
            path_values = paths_df.iloc[idx].values
            fig.add_trace(
                go.Scatter(
                    x=pred_times,
                    y=path_values,
                    mode='lines',
                    line=dict(width=2, color=colors[i], dash='dash'),
                    opacity=0.7,
                    showlegend=(i == 0),  # Only show legend for first path
                    name='Sample Paths',
                    hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                )
            )
    
    # Add confidence bands (10th to 90th percentile) - make more visible
    fig.add_trace(
        go.Scatter(
            x=list(pred_times) + list(pred_times[::-1]),
            y=list(percentiles['p90']) + list(percentiles['p10'][::-1]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Band',
            showlegend=True,
            hoverinfo='skip'
        )
    )
    
    # Add 25th to 75th percentile band (inner confidence)
    fig.add_trace(
        go.Scatter(
            x=list(pred_times) + list(pred_times[::-1]),
            y=list(percentiles['p75']) + list(percentiles['p25'][::-1]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='50% Confidence Band',
            showlegend=True,
            hoverinfo='skip'
        )
    )
    
    # Add connecting line from last historical price to first prediction
    fig.add_trace(
        go.Scatter(
            x=[last_time, pred_times[0]],
            y=[current_close, mean_path[0]],
            mode='lines',
            line=dict(width=3, color='blue', dash='dot'),
            name='Connection to Predictions',
            showlegend=True,
            hovertemplate='Connection from Friday close to Tuesday predictions<extra></extra>'
        )
    )
    
    # Add mean prediction path (make it very visible like in simple test)
    fig.add_trace(
        go.Scatter(
            x=pred_times,
            y=mean_path,
            mode='lines+markers',
            line=dict(width=5, color='blue'),
            marker=dict(size=8, color='blue'),
            name='Mean Prediction',
            showlegend=True,
            hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add median (50th percentile) path
    fig.add_trace(
        go.Scatter(
            x=pred_times,
            y=percentiles['p50'],
            mode='lines',
            line=dict(width=2, color='darkblue', dash='dash'),
            name='Median Prediction',
            showlegend=True,
            hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add current price horizontal line
    fig.add_hline(
        y=current_close,
        line_width=1,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Current: ${current_close:.2f}"
    )
    
    # Add VWAP line if available
    if 'current_vwap' in summary:
        fig.add_hline(
            y=summary['current_vwap'],
            line_width=1,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"VWAP: ${summary['current_vwap']:.2f}"
        )
    
    # Add Bollinger Bands if available
    if 'bollinger_bands' in summary:
        bb = summary['bollinger_bands']
        fig.add_hline(y=bb['upper'], line_width=1, line_dash="dot", 
                     line_color="gray")
        fig.add_hline(y=bb['middle'], line_width=1, line_dash="dot", 
                     line_color="gray")
        fig.add_hline(y=bb['lower'], line_width=1, line_dash="dot", 
                     line_color="gray")
    
    # Volume chart removed for now to focus on predictions visibility
    
    # Add probability annotations
    p_up = summary['p_up_30m']
    exp_ret = summary['exp_ret_30m']
    
    # Determine color based on probability
    prob_color = 'green' if p_up > 0.5 else 'red'
    
    # Add text box with statistics
    annotations = [
        dict(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            text=f"<b>Predictions (30-min horizon)</b><br>" +
                 f"P(Up): <b style='color:{prob_color}'>{p_up:.1%}</b><br>" +
                 f"Expected Return: <b>{exp_ret:+.2%}</b><br>" +
                 f"Samples: {summary_data.get('n_samples', 'N/A')}<br>" +
                 f"Temperature: {summary_data.get('temperature', 'N/A')}",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        ),
        dict(
            x=pred_times[15],  # Middle of prediction window
            y=mean_path[15],
            text="← 30-min Predictions",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue",
            font=dict(color="blue", size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="blue",
            borderwidth=1
        )
    ]
    
    # Add mean reversion probabilities if available
    if 'p_revert_vwap_30m' in summary:
        annotations[0]['text'] += f"<br>P(Touch VWAP): {summary['p_revert_vwap_30m']:.1%}"
    
    if 'p_retrace_midband' in summary:
        annotations[0]['text'] += f"<br>P(Touch BB Mid): {summary['p_retrace_midband']:.1%}"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title} - Predictions for {pred_times[0].strftime('%Y-%m-%d %H:%M')} UTC",
            font=dict(size=16)
        ),
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified',
        annotations=annotations,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    # Force the x-axis to show both historical and prediction data
    # Find the min and max times to set the range properly
    all_times = list(historical_df.index) + list(pred_times)
    x_range = [min(all_times), max(all_times)]
    fig.update_xaxes(range=x_range)
    
    # Update x-axis to show more time
    fig.update_xaxes(
        title_text="Time (UTC)",
        tickformat="%H:%M\n%m/%d",
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[21, 14.5], pattern="hour")  # Hide non-trading hours (UTC)
        ]
    )
    
    # Update y-axis formatting
    fig.update_yaxes(
        title_text="Price ($)",
        tickformat="$,.2f"
    )
    
    return fig

def create_simple_line_chart(
    historical_df: pd.DataFrame,
    summary_data: Dict[str, Any],
    paths_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Create a simpler line chart focusing on close prices"""
    
    summary = summary_data['summary']
    current_close = summary['current_close']
    mean_path = summary['mean_path']
    percentiles = summary['percentiles']
    
    # Create timestamps
    last_time = historical_df.index[-1]
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(minutes=1),
        periods=len(mean_path),
        freq='1min'
    )
    
    fig = go.Figure()
    
    # Historical close price
    fig.add_trace(
        go.Scatter(
            x=historical_df.index,
            y=historical_df['close'],
            mode='lines',
            line=dict(width=2, color='black'),
            name='Historical Close',
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
        )
    )
    
    # Confidence bands
    fig.add_trace(
        go.Scatter(
            x=list(pred_times) + list(pred_times[::-1]),
            y=list(percentiles['p90']) + list(percentiles['p10'][::-1]),
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.1)',
            line=dict(width=0),
            showlegend=True,
            name='90% Confidence',
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(pred_times) + list(pred_times[::-1]),
            y=list(percentiles['p75']) + list(percentiles['p25'][::-1]),
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.2)',
            line=dict(width=0),
            showlegend=True,
            name='50% Confidence',
            hoverinfo='skip'
        )
    )
    
    # Mean and median predictions
    fig.add_trace(
        go.Scatter(
            x=pred_times,
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Mean Prediction',
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=pred_times,
            y=percentiles['p50'],
            mode='lines',
            line=dict(width=2, color='darkblue', dash='dash'),
            name='Median Prediction',
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"QQQ Price Prediction - P(Up 30m): {summary['p_up_30m']:.1%}",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=600,
        hovermode='x unified',
        template="plotly_white"
    )
    
    return fig

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Visualize Kronos predictions')
    parser.add_argument('--summary', type=str, help='Path to summary JSON file')
    parser.add_argument('--paths', type=str, help='Path to paths CSV file')
    parser.add_argument('--historical', type=str, help='Path to historical data CSV')
    parser.add_argument('--output', type=str, help='Output HTML file path')
    parser.add_argument('--style', choices=['kronos', 'simple'], default='kronos',
                       help='Chart style to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Auto-detect latest files if not specified
    output_dir = config['output']['output_dir']
    
    if not args.summary:
        # Find latest summary file
        summary_files = sorted([f for f in os.listdir(output_dir) 
                               if f.startswith('pred_summary_') and f.endswith('.json')])
        if summary_files:
            args.summary = os.path.join(output_dir, summary_files[-1])
            print(f"Using latest summary: {args.summary}")
        else:
            print("No summary file found. Please run cli_kronos_prob_qqq.py first.")
            return
    
    if not args.paths:
        # Find corresponding paths file
        timestamp = args.summary.split('_')[-1].replace('.json', '')
        paths_file = os.path.join(output_dir, f"paths_{config['symbol']}_{timestamp}.csv")
        if os.path.exists(paths_file):
            args.paths = paths_file
            print(f"Using paths file: {args.paths}")
    
    if not args.historical:
        # Find corresponding historical file
        timestamp = args.summary.split('_')[-1].replace('.json', '')
        hist_file = os.path.join(output_dir, f"historical_{config['symbol']}_{timestamp}.csv")
        if os.path.exists(hist_file):
            args.historical = hist_file
            print(f"Using historical file: {args.historical}")
    
    # Load data
    summary_data, paths_df = load_prediction_data(args.summary, args.paths)
    
    # Load historical data
    if args.historical and os.path.exists(args.historical):
        print(f"Loading actual historical data from: {args.historical}")
        historical_df = pd.read_csv(args.historical, index_col=0, parse_dates=True)
        # Show all historical data (3 days) for proper context
        print(f"Loaded {len(historical_df)} bars of historical data")
    else:
        # Fallback: Create simulated historical data
        print("Warning: No historical data found. Using simulated data.")
        current_close = summary_data['summary']['current_close']
        hist_length = 100
        times = pd.date_range(end=datetime.now(), periods=hist_length, freq='1min')
        
        # Simulate historical prices
        np.random.seed(42)
        returns = np.random.normal(0, 0.002, hist_length)
        prices = current_close * np.exp(np.cumsum(returns))
        prices[-1] = current_close  # Ensure last price matches
        
        historical_df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, hist_length)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, hist_length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, hist_length))),
            'close': prices,
            'volume': np.random.uniform(1e6, 5e6, hist_length)
        }, index=times)
    
    # Create chart
    if args.style == 'kronos':
        fig = create_kronos_style_chart(historical_df, summary_data, paths_df)
    else:
        fig = create_simple_line_chart(historical_df, summary_data, paths_df)
    
    # Save chart
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        args.output = os.path.join(output_dir, f"chart_{config['symbol']}_{timestamp}.html")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)
    print(f"Chart saved to: {args.output}")
    
    # Also save as PNG if configured
    if 'png' in config['output'].get('chart_format', []):
        png_path = args.output.replace('.html', '.png')
        try:
            fig.write_image(png_path, width=1400, height=800)
            print(f"PNG saved to: {png_path}")
        except:
            print("Note: PNG export requires kaleido package")
    
    # Show chart
    fig.show()

if __name__ == "__main__":
    main()