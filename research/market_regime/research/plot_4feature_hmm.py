"""
Plot 4-Feature Income Defender HMM chart for Sep 15-19
Uses optimized batch query for quote_imbalance (NO N+1 problem).
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'C:/Users/skysn/workspace/intraday_predication/live_chart_prediction')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from datetime import date
from pathlib import Path
from market_regime.feature_pipeline import FeaturePipeline
from market_regime.regime_detector import IncomeDefenderDetector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('=== 4-Feature Income Defender HMM ===')
print('Using optimized batch query + merge_asof for quote_imbalance')
print()

# Initialize pipeline
pipeline = FeaturePipeline(
    symbol='QQQ',
    bucket_volume=250000,
    vpin_buckets=50,
    bar_type='volume',
    volume_bar_size=250000
)

# Use compute_income_defender_features which has optimized batch quote_imbalance
# - Base features: use cache (fast)
# - quote_imbalance: ONE batch query + merge_asof (optimized, ~4 min)
print('Computing features with optimized quote_imbalance batch...')
features = pipeline.compute_income_defender_features(
    start_date=date(2025, 9, 15),
    end_date=date(2025, 9, 19),
    timeframe='volume',
    rth_only=True,
    use_cache=True  # Use cached base features, only compute quote_imbalance fresh
)

print(f'Got {len(features)} volume bars')

# Train detector
detector = IncomeDefenderDetector(n_states=3, n_iter=100, random_state=42)
detector.fit(features)

# Predict states
states = detector.predict(features)
features['state'] = states

# State colors
state_colors = {0: 'blue', 1: 'green', 2: 'red'}
state_names = {0: 'Bull', 1: 'Dip', 2: 'Stress'}

# Create sequential index for RTH-only (no gaps)
features = features.reset_index()
features['bar_idx'] = range(len(features))
features['date'] = pd.to_datetime(features['time']).dt.date

# Find day boundaries
day_starts = features.groupby('date')['bar_idx'].first().values[1:]  # Skip first day

print('Creating chart...')

# Create figure with 5 panels
fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

# Panel 1: Price with HMM State colors
ax1 = axes[0]
for state_id in [0, 1, 2]:
    mask = features['state'] == state_id
    ax1.scatter(features.loc[mask, 'bar_idx'], features.loc[mask, 'close'],
                c=state_colors[state_id], s=8, alpha=0.7, label=state_names[state_id])
ax1.set_ylabel('Price ($)')
ax1.set_title('QQQ Sep 15-19 2025 - 4-Feature Income Defender HMM (RTH Only)', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add day separators
for day_start in day_starts:
    ax1.axvline(x=day_start, color='gray', linestyle='--', alpha=0.5)

# Panel 2: OFI
ax2 = axes[1]
ax2.bar(features['bar_idx'], features['ofi'], color='steelblue', alpha=0.7, width=1)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('OFI')
ax2.grid(True, alpha=0.3)
for day_start in day_starts:
    ax2.axvline(x=day_start, color='gray', linestyle='--', alpha=0.5)

# Panel 3: VPIN Rank
ax3 = axes[2]
ax3.fill_between(features['bar_idx'], features['vpin_rank'], alpha=0.5, color='orange')
ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='High Toxicity (0.7)')
ax3.set_ylabel('VPIN Rank')
ax3.set_ylim(0, 1)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
for day_start in day_starts:
    ax3.axvline(x=day_start, color='gray', linestyle='--', alpha=0.5)

# Panel 4: VRP
ax4 = axes[3]
ax4.fill_between(features['bar_idx'], features['vrp'], alpha=0.5,
                  where=features['vrp'] >= 0, color='green')
ax4.fill_between(features['bar_idx'], features['vrp'], alpha=0.5,
                  where=features['vrp'] < 0, color='red')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_ylabel('VRP (%)')
ax4.grid(True, alpha=0.3)
for day_start in day_starts:
    ax4.axvline(x=day_start, color='gray', linestyle='--', alpha=0.5)

# Panel 5: Quote Imbalance (NEW!)
ax5 = axes[4]
ax5.fill_between(features['bar_idx'], features['quote_imbalance'], alpha=0.5,
                  where=features['quote_imbalance'] >= 0, color='green')
ax5.fill_between(features['bar_idx'], features['quote_imbalance'], alpha=0.5,
                  where=features['quote_imbalance'] < 0, color='red')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax5.axhline(y=-0.05, color='red', linestyle='--', linewidth=1, label='Danger Zone (-0.05)')
ax5.set_ylabel('Quote Imbalance')
ax5.set_xlabel('Bar Index (RTH)')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)
for day_start in day_starts:
    ax5.axvline(x=day_start, color='gray', linestyle='--', alpha=0.5)

# Add day labels at top
unique_dates = features['date'].unique()
for i, d in enumerate(unique_dates):
    day_data = features[features['date'] == d]
    mid_idx = day_data['bar_idx'].iloc[len(day_data)//2]
    ax1.text(mid_idx, ax1.get_ylim()[1], str(d), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('C:/Users/skysn/workspace/intraday_predication/live_chart_prediction/market_regime/research/hmm_4feature_sep15_19.png', dpi=150, bbox_inches='tight')
print('Chart saved to: hmm_4feature_sep15_19.png')

# Print Sep 17 analysis
print()
print('=== Sep 17 Analysis ===')
nov20 = features[features['date'] == date(2025, 9, 17)].copy()
print(f'Total bars: {len(nov20)}')
print(f'Price range: ${nov20["close"].min():.2f} - ${nov20["close"].max():.2f}')

# Find peak
peak_idx = nov20['close'].idxmax()
peak_bar = nov20.loc[peak_idx]
print(f'Peak: bar {peak_bar["bar_idx"]}, price ${peak_bar["close"]:.2f}, time {peak_bar["time"]}')
print(f'  State at peak: {state_names[int(peak_bar["state"])]}')
print(f'  Quote imbalance at peak: {peak_bar["quote_imbalance"]:.3f}')

# Show states around peak
print()
print('States around peak (bar -5 to +10):')
peak_local_idx = nov20.index.get_loc(peak_idx)
for i in range(max(0, peak_local_idx-5), min(len(nov20), peak_local_idx+11)):
    row = nov20.iloc[i]
    marker = ' <-- PEAK' if i == peak_local_idx else ''
    print(f'  Bar {row["bar_idx"]}: ${row["close"]:.2f} | State: {state_names[int(row["state"])]:6} | QIB: {row["quote_imbalance"]:+.3f}{marker}')

# Find when Stress first appeared on Sep 17
stress_bars = nov20[nov20['state'] == 2]
if len(stress_bars) > 0:
    first_stress = stress_bars.iloc[0]
    print()
    print(f'First Stress signal on Sep 17:')
    print(f'  Bar {first_stress["bar_idx"]}, time {first_stress["time"]}, price ${first_stress["close"]:.2f}')
    print(f'  Quote imbalance: {first_stress["quote_imbalance"]:.3f}')
