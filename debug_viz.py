#!/usr/bin/env python
"""
Debug visualization script to check why predictions aren't visible
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
import pytz

def main():
    # Load data
    with open('output/pred_summary_QQQ_20250830T194014Z.json', 'r') as f:
        summary_data = json.load(f)
    
    paths_df = pd.read_csv('output/paths_QQQ_20250830T194014Z.csv')
    historical_df = pd.read_csv('output/historical_QQQ_20250830T194014Z.csv', index_col=0, parse_dates=True)
    
    print("=== DEBUG INFO ===")
    print(f"Historical data: {len(historical_df)} bars")
    print(f"Historical range: {historical_df.index[0]} to {historical_df.index[-1]}")
    print(f"Historical close range: ${historical_df['close'].min():.2f} - ${historical_df['close'].max():.2f}")
    
    print(f"\nPaths data: {paths_df.shape}")
    print(f"Mean path range: ${min(summary_data['summary']['mean_path']):.2f} - ${max(summary_data['summary']['mean_path']):.2f}")
    
    # Generate prediction timestamps
    last_time = historical_df.index[-1]
    print(f"Last historical time: {last_time}")
    
    # Simple prediction times (just add 1 day for now to make them visible)
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(days=1),
        periods=30,
        freq='1min'
    )
    print(f"Prediction times: {pred_times[0]} to {pred_times[-1]}")
    
    # Create simple chart
    fig = go.Figure()
    
    # Add historical close prices (last 200 bars for clarity)
    hist_subset = historical_df.tail(200)
    fig.add_trace(go.Scatter(
        x=hist_subset.index,
        y=hist_subset['close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='black', width=1)
    ))
    
    # Add mean prediction
    mean_path = summary_data['summary']['mean_path']
    fig.add_trace(go.Scatter(
        x=pred_times,
        y=mean_path,
        mode='lines+markers',
        name='Mean Prediction',
        line=dict(color='blue', width=3)
    ))
    
    # Add a few sample paths
    for i in range(0, min(5, len(paths_df))):
        fig.add_trace(go.Scatter(
            x=pred_times,
            y=paths_df.iloc[i].values,
            mode='lines',
            name=f'Sample {i+1}',
            line=dict(color=f'rgba(255, 0, 0, 0.3)', width=1),
            showlegend=(i == 0)
        ))
    
    # Add confidence bands
    percentiles = summary_data['summary']['percentiles']
    fig.add_trace(go.Scatter(
        x=list(pred_times) + list(pred_times[::-1]),
        y=list(percentiles['p90']) + list(percentiles['p10'][::-1]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence Band',
        showlegend=True
    ))
    
    fig.update_layout(
        title="DEBUG: QQQ Historical + Predictions",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True
    )
    
    # Save and show
    fig.write_html("output/debug_chart.html")
    print(f"\nDebug chart saved to: output/debug_chart.html")
    
    # Print data ranges for comparison
    print(f"\n=== DATA RANGES ===")
    print(f"Historical close: ${historical_df['close'].iloc[-1]:.2f}")
    print(f"Mean prediction start: ${mean_path[0]:.2f}")
    print(f"Mean prediction end: ${mean_path[-1]:.2f}")
    print(f"Sample path 1 start: ${paths_df.iloc[0, 0]:.2f}")
    
    fig.show()

if __name__ == "__main__":
    main()