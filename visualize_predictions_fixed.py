#!/usr/bin/env python
"""
Fixed Visualization Script for Kronos QQQ Predictions
Based on the working simple_viz_test.py
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import argparse
from typing import Dict, Any, Optional
import pytz

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prediction_data(summary_file: str, paths_file: Optional[str] = None) -> tuple:
    """Load prediction data from files"""
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    paths_df = None
    if paths_file and os.path.exists(paths_file):
        paths_df = pd.read_csv(paths_file)
    
    return summary_data, paths_df

def get_next_trading_session(last_time: pd.Timestamp, horizon_minutes: int = 30) -> pd.DatetimeIndex:
    """Calculate the next trading session timestamps for predictions"""
    # For now, just continue from last time (like in simple test)
    # This can be enhanced later for proper next trading day calculation
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(minutes=1),
        periods=horizon_minutes,
        freq='1min'
    )
    return pred_times

def create_visualization(
    historical_df: pd.DataFrame,
    summary_data: Dict[str, Any],
    paths_df: Optional[pd.DataFrame] = None,
    title: str = "QQQ Intraday Prediction"
) -> go.Figure:
    """
    Create visualization based on working simple test approach
    """
    # Extract data
    summary = summary_data['summary']
    current_close = summary['current_close']
    mean_path = summary['mean_path']
    percentiles = summary['percentiles']
    
    # Historical data should already be in Eastern Time from CLI script
    # Just ensure it's timezone aware
    est_tz = pytz.timezone('US/Eastern')
    historical_df = historical_df.copy()
    if historical_df.index.tz is None:
        # If for some reason it's still timezone naive, assume it's already Eastern Time
        historical_df.index = historical_df.index.tz_localize(est_tz)
    elif not (hasattr(historical_df.index.tz, 'zone') and historical_df.index.tz.zone == 'US/Eastern'):
        # Convert to Eastern if it's not already Eastern Time (handles different timezone object types)
        historical_df.index = historical_df.index.tz_convert(est_tz)
    
    # Get recent historical data (last 100-200 bars for context)
    recent_hist = historical_df.tail(200)
    
    # Create timestamps for predictions (continue from last historical time in EST)
    last_time = historical_df.index[-1]
    
    # Generate predictions starting from the next minute after historical data ends
    # Since last_time is already in EST, just continue from there
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(minutes=1),
        periods=len(mean_path),
        freq='1min',
        tz=est_tz
    )
    
    print(f"Historical data ends at: {last_time}")
    print(f"Predictions start at: {pred_times[0]}")
    print(f"Predictions end at: {pred_times[-1]}")
    print(f"Mean path range: ${min(mean_path):.2f} - ${max(mean_path):.2f}")
    
    # Create figure
    fig = go.Figure()
    
    # 1. Historical close price (just use close, not candlesticks for clarity)
    fig.add_trace(go.Scatter(
        x=recent_hist.index,
        y=recent_hist['close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='black', width=2)
    ))
    
    # 2. Confidence bands (make them very visible)
    # 90% confidence band
    fig.add_trace(go.Scatter(
        x=list(pred_times) + list(pred_times[::-1]),
        y=list(percentiles['p90']) + list(percentiles['p10'][::-1]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence Band',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # 50% confidence band (inner)
    fig.add_trace(go.Scatter(
        x=list(pred_times) + list(pred_times[::-1]),
        y=list(percentiles['p75']) + list(percentiles['p25'][::-1]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='50% Confidence Band',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # 3. Mean prediction with markers (very visible)
    fig.add_trace(go.Scatter(
        x=pred_times,
        y=mean_path,
        mode='lines+markers',
        name='Mean Prediction',
        line=dict(color='blue', width=5),
        marker=dict(size=8, color='blue'),
        hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # 4. Add sample paths (if available)
    if paths_df is not None and len(paths_df) > 0:
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        n_paths_to_show = min(5, len(paths_df))
        sample_indices = np.linspace(0, len(paths_df)-1, n_paths_to_show, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            fig.add_trace(go.Scatter(
                x=pred_times,
                y=paths_df.iloc[idx].values,
                mode='lines',
                name=f'Sample Path {i+1}',
                line=dict(color=colors[i], width=2, dash='dash'),
                opacity=0.7,
                showlegend=(i == 0),  # Only show first in legend
                hovertemplate='Sample Path<br>Price: $%{y:.2f}<extra></extra>'
            ))
    
    # 5. Add connecting line
    fig.add_trace(go.Scatter(
        x=[last_time, pred_times[0]],
        y=[historical_df['close'].iloc[-1], mean_path[0]],
        mode='lines',
        name='Connection',
        line=dict(color='blue', width=3, dash='dot'),
        showlegend=True
    ))
    
    # 6. Add vertical lines for clarity
    y_range = [
        min(recent_hist['close'].min(), min(mean_path)) * 0.999,
        max(recent_hist['close'].max(), max(mean_path)) * 1.001
    ]
    
    # Market close line
    fig.add_trace(go.Scatter(
        x=[last_time, last_time],
        y=y_range,
        mode='lines',
        name='Last Historical Data',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True
    ))
    
    # Prediction start line
    fig.add_trace(go.Scatter(
        x=[pred_times[0], pred_times[0]],
        y=y_range,
        mode='lines',
        name='Prediction Start',
        line=dict(color='green', width=2, dash='dash'),
        showlegend=True
    ))
    
    # 7. Add horizontal reference lines
    fig.add_hline(
        y=current_close,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Last Close: ${current_close:.2f}"
    )
    
    # 8. Add probability annotations
    p_up = summary['p_up_30m']
    exp_ret = summary['exp_ret_30m']
    
    prob_color = 'green' if p_up > 0.5 else 'red'
    
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=f"<b>30-min Predictions</b><br>" +
             f"P(Up): <b style='color:{prob_color}'>{p_up:.1%}</b><br>" +
             f"Expected Return: <b>{exp_ret:+.2%}</b><br>" +
             f"Current: ${current_close:.2f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{title} - Historical + 30-min Predictions (Eastern Time)",
        xaxis_title="Time (EST/EDT)",
        yaxis_title="Price ($)",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified',
        template="plotly_white"
    )
    
    # Format axes
    fig.update_xaxes(
        tickformat="%H:%M\n%m/%d",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    fig.update_yaxes(
        tickformat="$,.2f",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    return fig

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Visualize Kronos predictions (Fixed Version)')
    parser.add_argument('--summary', type=str, help='Path to summary JSON file')
    parser.add_argument('--paths', type=str, help='Path to paths CSV file')
    parser.add_argument('--historical', type=str, help='Path to historical data CSV')
    parser.add_argument('--output', type=str, help='Output HTML file path')
    
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
        print(f"Loaded {len(historical_df)} bars of historical data")
    else:
        print("Error: No historical data found.")
        return
    
    # Create visualization
    fig = create_visualization(historical_df, summary_data, paths_df)
    
    # Save chart
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        args.output = os.path.join(output_dir, f"chart_fixed_{config['symbol']}_{timestamp}.html")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)
    print(f"Chart saved to: {args.output}")
    
    # Also save as PNG
    png_path = args.output.replace('.html', '.png')
    try:
        fig.write_image(png_path, width=1400, height=700)
        print(f"PNG saved to: {png_path}")
    except:
        print("Note: PNG export requires kaleido package")
    
    # Show chart
    fig.show()

if __name__ == "__main__":
    main()