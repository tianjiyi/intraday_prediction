#!/usr/bin/env python
"""
Simple visualization test to make sure predictions are clearly visible
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

def main():
    # Load data
    with open('output/pred_summary_QQQ_20250830T194014Z.json', 'r') as f:
        summary_data = json.load(f)
    
    paths_df = pd.read_csv('output/paths_QQQ_20250830T194014Z.csv')
    historical_df = pd.read_csv('output/historical_QQQ_20250830T194014Z.csv', index_col=0, parse_dates=True)
    
    # Get last 50 bars of historical data for context
    recent_hist = historical_df.tail(50)
    
    # Create simple prediction times (just continue from last historical time)
    last_time = historical_df.index[-1]
    pred_times = pd.date_range(
        start=last_time + pd.Timedelta(minutes=1),
        periods=30,
        freq='1min'
    )
    
    mean_path = summary_data['summary']['mean_path']
    percentiles = summary_data['summary']['percentiles']
    
    # Create figure
    fig = go.Figure()
    
    # Historical close (recent)
    fig.add_trace(go.Scatter(
        x=recent_hist.index,
        y=recent_hist['close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='black', width=2)
    ))
    
    # Predictions - make them VERY visible
    
    # 1. Confidence band (wide)
    fig.add_trace(go.Scatter(
        x=list(pred_times) + list(pred_times[::-1]),
        y=list(percentiles['p90']) + list(percentiles['p10'][::-1]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence Band',
        hoverinfo='skip'
    ))
    
    # 2. Mean prediction with markers
    fig.add_trace(go.Scatter(
        x=pred_times,
        y=mean_path,
        mode='lines+markers',
        name='Mean Prediction',
        line=dict(color='blue', width=5),
        marker=dict(size=8, color='blue')
    ))
    
    # 3. Add a few sample paths
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i in range(min(5, len(paths_df))):
        fig.add_trace(go.Scatter(
            x=pred_times,
            y=paths_df.iloc[i].values,
            mode='lines',
            name=f'Sample Path {i+1}',
            line=dict(color=colors[i], width=2, dash='dash'),
            opacity=0.7
        ))
    
    # 4. Add connecting line
    fig.add_trace(go.Scatter(
        x=[last_time, pred_times[0]],
        y=[historical_df['close'].iloc[-1], mean_path[0]],
        mode='lines',
        name='Connection',
        line=dict(color='blue', width=3, dash='dot')
    ))
    
    # Add vertical line at prediction start
    y_range = [min(recent_hist['close'].min(), min(mean_path)) * 0.999,
               max(recent_hist['close'].max(), max(mean_path)) * 1.001]
    
    fig.add_trace(go.Scatter(
        x=[last_time, last_time],
        y=y_range,
        mode='lines',
        name='Prediction Start',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Layout
    fig.update_layout(
        title="SIMPLE TEST: QQQ Historical + Predictions (Next Trading Day)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Save
    fig.write_html("output/simple_viz_test.html")
    fig.write_image("output/simple_viz_test.png")
    
    print("Simple visualization saved to:")
    print("- output/simple_viz_test.html")
    print("- output/simple_viz_test.png")
    
    print(f"\nData summary:")
    print(f"Historical close: ${historical_df['close'].iloc[-1]:.2f}")
    print(f"Prediction range: ${min(mean_path):.2f} - ${max(mean_path):.2f}")
    print(f"Time gap: {pred_times[0] - last_time}")
    
    fig.show()

if __name__ == "__main__":
    main()