"""
Daily-Level OFI Divergence Analysis

The intraday divergence is too noisy. Let's aggregate to daily level:
- Daily High vs Daily OFI Sum
- Look for divergence across DAYS, not bars
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market_regime.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.WARNING)


def main():
    symbol = 'QQQ'
    start_date = date(2025, 11, 17)
    end_date = date(2025, 11, 21)
    volume_bar_size = 10000

    # Load features
    print("Loading volume bar data...")
    pipeline = FeaturePipeline(
        symbol=symbol, bucket_volume=volume_bar_size, vpin_buckets=50,
        bar_type='volume', volume_bar_size=volume_bar_size
    )

    features = pipeline.compute_features(
        start_date=start_date, end_date=end_date,
        timeframe='volume', rth_only=True, use_cache=True
    )
    print(f"Loaded {len(features)} volume bars")

    # Add date column
    features['date'] = features.index.date

    # Aggregate to daily level
    daily = aggregate_daily(features)
    print("\n" + "=" * 80)
    print("DAILY OFI SUMMARY")
    print("=" * 80)
    print(daily.to_string())

    # Detect daily divergence
    print("\n" + "=" * 80)
    print("DAILY DIVERGENCE ANALYSIS")
    print("=" * 80)
    analyze_daily_divergence(daily)

    # Create visualization
    create_daily_chart(features, daily, start_date, end_date)

    # Intraday evolution chart
    create_intraday_evolution_chart(features, start_date, end_date)


def aggregate_daily(features):
    """Aggregate volume bars to daily level."""

    daily_data = []

    for day in sorted(features['date'].unique()):
        day_df = features[features['date'] == day]

        daily_data.append({
            'date': day,
            'day_name': day.strftime('%a'),
            'open': day_df['open'].iloc[0],
            'high': day_df['high'].max(),
            'low': day_df['low'].min(),
            'close': day_df['close'].iloc[-1],
            'ofi_sum': day_df['ofi'].sum(),
            'ofi_mean': day_df['ofi'].mean(),
            'ofi_std': day_df['ofi'].std(),
            'vpin_mean': day_df['vpin'].mean(),
            'volume_bars': len(day_df),
            'pct_change': (day_df['close'].iloc[-1] - day_df['open'].iloc[0]) / day_df['open'].iloc[0] * 100
        })

    daily = pd.DataFrame(daily_data)

    # Add cumulative OFI
    daily['ofi_cum'] = daily['ofi_sum'].cumsum()

    # Add price change from start
    daily['price_from_start'] = (daily['close'] - daily['close'].iloc[0]) / daily['close'].iloc[0] * 100

    return daily


def analyze_daily_divergence(daily):
    """Analyze divergence at daily level."""

    print("\nDay-by-Day Analysis:")
    print("-" * 60)

    for i, row in daily.iterrows():
        print(f"\n{row['day_name']} {row['date'].strftime('%m/%d')}:")
        print(f"  Price: {row['open']:.2f} → {row['close']:.2f} ({row['pct_change']:+.2f}%)")
        print(f"  High: {row['high']:.2f} | Low: {row['low']:.2f}")
        print(f"  OFI Sum: {row['ofi_sum']:+,.0f}")
        print(f"  OFI Cumulative: {row['ofi_cum']:+,.0f}")
        print(f"  VPIN Avg: {row['vpin_mean']:.3f}")

    # Check for divergence patterns
    print("\n" + "-" * 60)
    print("DIVERGENCE PATTERNS:")
    print("-" * 60)

    for i in range(1, len(daily)):
        prev = daily.iloc[i-1]
        curr = daily.iloc[i]

        # Price direction
        price_up = curr['high'] > prev['high']
        price_down = curr['low'] < prev['low']

        # OFI direction
        ofi_up = curr['ofi_sum'] > prev['ofi_sum']
        ofi_down = curr['ofi_sum'] < prev['ofi_sum']

        # Cumulative OFI direction
        cum_ofi_up = curr['ofi_cum'] > prev['ofi_cum']
        cum_ofi_down = curr['ofi_cum'] < prev['ofi_cum']

        divergence = None
        if price_up and ofi_down:
            divergence = "BEARISH DIVERGENCE"
        elif price_down and ofi_up:
            divergence = "BULLISH DIVERGENCE"
        elif price_up and ofi_up:
            divergence = "Bullish Convergence (healthy uptrend)"
        elif price_down and ofi_down:
            divergence = "Bearish Convergence (healthy downtrend)"

        print(f"\n{prev['day_name']} → {curr['day_name']}:")
        print(f"  Price High: {prev['high']:.2f} → {curr['high']:.2f} ({'↑' if price_up else '↓'})")
        print(f"  OFI Sum: {prev['ofi_sum']:+,.0f} → {curr['ofi_sum']:+,.0f} ({'↑' if ofi_up else '↓'})")
        print(f"  OFI Cum: {prev['ofi_cum']:+,.0f} → {curr['ofi_cum']:+,.0f} ({'↑' if cum_ofi_up else '↓'})")
        if divergence:
            if "DIVERGENCE" in divergence:
                print(f"  >>> {divergence} <<<")
            else:
                print(f"  {divergence}")


def create_daily_chart(features, daily, start_date, end_date):
    """Create daily-level OFI chart."""

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    x = np.arange(len(daily))
    labels = [f"{r['day_name']}\n{r['date'].strftime('%m/%d')}" for _, r in daily.iterrows()]

    # Panel 0: Daily OHLC
    ax0 = axes[0]
    for i, row in daily.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Body
        ax0.bar(i, row['close'] - row['open'], bottom=row['open'],
               color=color, width=0.6, alpha=0.8)
        # Wick
        ax0.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)

    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=10)
    ax0.set_ylabel('Price', fontsize=10)
    ax0.set_title(f'QQQ Daily OFI Divergence Analysis: {start_date} to {end_date}', fontsize=14)
    ax0.grid(True, alpha=0.3)

    # Panel 1: Daily OFI Sum (bar chart)
    ax1 = axes[1]
    colors = ['green' if v >= 0 else 'red' for v in daily['ofi_sum']]
    ax1.bar(x, daily['ofi_sum'], color=colors, alpha=0.7, width=0.6)
    ax1.axhline(y=0, color='gray', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Daily OFI Sum', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(daily['ofi_sum']):
        ax1.text(i, v, f'{v/1e6:.1f}M', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

    # Panel 2: Cumulative OFI
    ax2 = axes[2]
    ax2.plot(x, daily['ofi_cum'], 'b-o', linewidth=2, markersize=8)
    ax2.fill_between(x, 0, daily['ofi_cum'],
                     where=daily['ofi_cum'] >= 0, alpha=0.3, color='green')
    ax2.fill_between(x, 0, daily['ofi_cum'],
                     where=daily['ofi_cum'] < 0, alpha=0.3, color='red')
    ax2.axhline(y=0, color='gray', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Cumulative OFI', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Price vs OFI comparison (normalized)
    ax3 = axes[3]

    # Normalize both to start at 0
    price_norm = (daily['high'] - daily['high'].iloc[0]) / daily['high'].iloc[0] * 100
    ofi_norm = daily['ofi_cum'] / abs(daily['ofi_cum']).max() * price_norm.abs().max()

    ax3.plot(x, price_norm, 'k-o', linewidth=2, markersize=8, label='Price High (normalized)')
    ax3.plot(x, ofi_norm, 'b-s', linewidth=2, markersize=8, label='OFI Cum (normalized)')

    # Highlight divergence
    for i in range(1, len(daily)):
        if price_norm.iloc[i] > price_norm.iloc[i-1] and ofi_norm.iloc[i] < ofi_norm.iloc[i-1]:
            ax3.axvspan(i-0.3, i+0.3, alpha=0.3, color='red', label='Bearish Div' if i == 1 else '')
        elif price_norm.iloc[i] < price_norm.iloc[i-1] and ofi_norm.iloc[i] > ofi_norm.iloc[i-1]:
            ax3.axvspan(i-0.3, i+0.3, alpha=0.3, color='green', label='Bullish Div' if i == 1 else '')

    ax3.axhline(y=0, color='gray', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_ylabel('Normalized', fontsize=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(__file__).parent / 'nov17_21_daily_ofi.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nDaily chart saved to: {save_path}")
    plt.close()


def create_intraday_evolution_chart(features, start_date, end_date):
    """Create chart showing intraday OFI evolution within each day."""

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=False)

    trading_days = sorted(features['date'].unique())

    for idx, day in enumerate(trading_days):
        ax = axes[idx]
        day_data = features[features['date'] == day].copy()

        x = np.arange(len(day_data))

        # Plot price
        ax_price = ax
        price_line = ax_price.plot(x, day_data['close'].values, 'k-', linewidth=1, label='Price')
        ax_price.set_ylabel('Price', fontsize=9)

        # Plot OFI cumulative (within day)
        ax_ofi = ax.twinx()
        ofi_cum = day_data['ofi'].cumsum().values

        ax_ofi.fill_between(x, 0, ofi_cum, where=ofi_cum >= 0, alpha=0.3, color='green')
        ax_ofi.fill_between(x, 0, ofi_cum, where=ofi_cum < 0, alpha=0.3, color='red')
        ofi_line = ax_ofi.plot(x, ofi_cum, 'b-', linewidth=1, alpha=0.7, label='OFI Cumsum')
        ax_ofi.axhline(y=0, color='blue', linewidth=0.5, linestyle='--')
        ax_ofi.set_ylabel('Intraday OFI Cum', fontsize=9, color='blue')

        # Price change for the day
        pct_change = (day_data['close'].iloc[-1] - day_data['open'].iloc[0]) / day_data['open'].iloc[0] * 100
        final_ofi = ofi_cum[-1]

        day_name = day.strftime('%a %m/%d')
        ax.set_title(f"{day_name}: Price {pct_change:+.2f}% | OFI Sum: {final_ofi:+,.0f}", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Legend on first plot
        if idx == 0:
            lines = price_line + ofi_line
            labels = ['Price', 'OFI Cumsum']
            ax.legend(lines, labels, loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Volume Bar Index (within day)', fontsize=10)

    plt.suptitle(f'QQQ Intraday OFI Evolution: {start_date} to {end_date}', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = Path(__file__).parent / 'nov17_21_intraday_ofi.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Intraday chart saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
