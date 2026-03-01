import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from ultralytics import YOLO
from pathlib import Path

# Load both models
our_model = YOLO('runs/detect/w_bottom_train/weights/best.pt')
original_model = YOLO('models/foduucom_original/model.pt')
w_bottom_class_idx = 5

output_dir = Path('w_bottom_full_comparison')
output_dir.mkdir(exist_ok=True)

def generate_temp_chart(df, output_path, image_size=640):
    """Generate clean chart for YOLO inference"""
    fig_size = image_size / 100
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    padding = price_range * 0.05
    ax.set_ylim(price_min - padding, price_max + padding)
    ax.set_xlim(-0.5, len(df) - 0.5)
    width = 0.8
    for i, (_, row) in enumerate(df.iterrows()):
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        body_bottom = min(row['open'], row['close'])
        body_height = max(abs(row['close'] - row['open']), 0.001)
        rect = Rectangle((i - width/2, body_bottom), width, body_height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
    plt.close()

def create_full_comparison(ticker, date_str, hist_file, output_path):
    # Load data
    df = pd.read_csv(hist_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    day_df = df[df['timestamp'].dt.strftime('%Y-%m-%d') == date_str].reset_index(drop=True)

    if len(day_df) < 60:
        print(f'  Skipping {ticker} {date_str} - only {len(day_df)} bars')
        return None

    # Generate full day chart for inference
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    full_chart_path = temp_dir / f'{ticker}_{date_str}_full.png'
    generate_temp_chart(day_df, full_chart_path)

    # Run inference on full chart
    our_result = our_model.predict(source=str(full_chart_path), conf=0.25, verbose=False)[0]
    orig_result = original_model.predict(source=str(full_chart_path), conf=0.25, verbose=False)[0]

    # Get boxes (normalized 0-1)
    our_boxes = [(float(b.xyxyn[0][0]), float(b.xyxyn[0][1]), float(b.xyxyn[0][2]), float(b.xyxyn[0][3]), float(b.conf))
                 for b in our_result.boxes]
    orig_boxes = [(float(b.xyxyn[0][0]), float(b.xyxyn[0][1]), float(b.xyxyn[0][2]), float(b.xyxyn[0][3]), float(b.conf))
                  for b in orig_result.boxes if int(b.cls) == w_bottom_class_idx]

    # Sample confidence over time (every 10 bars)
    bar_indices = list(range(30, len(day_df), 10))
    if len(day_df) - 1 not in bar_indices:
        bar_indices.append(len(day_df) - 1)

    our_confs = []
    orig_confs = []

    for bar_idx in bar_indices:
        chart_path = temp_dir / f'{ticker}_{date_str}_bar_{bar_idx}.png'
        subset = day_df.iloc[:bar_idx+1]
        generate_temp_chart(subset, chart_path)

        our_r = our_model.predict(source=str(chart_path), conf=0.1, verbose=False)[0]
        our_max = max([float(b.conf) for b in our_r.boxes], default=0)
        our_confs.append(our_max)

        orig_r = original_model.predict(source=str(chart_path), conf=0.1, verbose=False)[0]
        orig_w = [b for b in orig_r.boxes if int(b.cls) == w_bottom_class_idx]
        orig_max = max([float(b.conf) for b in orig_w], default=0)
        orig_confs.append(orig_max)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.12)

    # Top panel: Full day chart with boxes
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f'{ticker} {date_str} - W_Bottom Detection Comparison', fontsize=14, fontweight='bold')

    # Draw candlesticks
    n_bars = len(day_df)
    price_min = day_df['low'].min()
    price_max = day_df['high'].max()
    price_range = price_max - price_min
    padding = price_range * 0.08

    for i, (_, row) in enumerate(day_df.iterrows()):
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)
        body_bottom = min(row['open'], row['close'])
        body_height = max(abs(row['close'] - row['open']), 0.001)
        rect = Rectangle((i - 0.4, body_bottom), 0.8, body_height, facecolor=color, edgecolor=color)
        ax1.add_patch(rect)

    ax1.set_xlim(-2, n_bars + 2)
    ax1.set_ylim(price_min - padding, price_max + padding)

    # Convert normalized boxes to chart coordinates and draw
    def draw_box(ax, box, color, label, y_offset=0):
        x1_norm, y1_norm, x2_norm, y2_norm, conf = box
        # Convert normalized coords to chart coords
        x1 = x1_norm * n_bars - 0.5
        x2 = x2_norm * n_bars - 0.5
        # Y is inverted in image coords
        chart_height = price_max - price_min + 2*padding
        y1 = price_max + padding - y1_norm * chart_height
        y2 = price_max + padding - y2_norm * chart_height

        width = x2 - x1
        height = y1 - y2  # y1 > y2 due to inversion

        rect = Rectangle((x1, y2 + y_offset), width, height,
                         linewidth=2.5, edgecolor=color, facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.text(x1, y1 + y_offset + price_range*0.02, f'{label} {conf:.2f}',
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Draw our model boxes (blue)
    for box in our_boxes:
        draw_box(ax1, box, 'blue', 'Ours:', y_offset=0)

    # Draw original model boxes (magenta) with slight offset
    for box in orig_boxes:
        draw_box(ax1, box, 'magenta', 'Orig:', y_offset=price_range*0.01)

    ax1.set_ylabel('Price', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Time labels
    for i in range(0, len(day_df), 60):
        if i < len(day_df):
            t = day_df.iloc[i]['timestamp']
            time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else ''
            ax1.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
            ax1.text(i, price_min - padding*0.5, time_str, fontsize=8, ha='center')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2.5, label=f'Our Model ({len(our_boxes)} detections)'),
        Line2D([0], [0], color='magenta', linewidth=2.5, label=f'Original ({len(orig_boxes)} W_Bottom)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Bottom panel: Confidence over time
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('Detection Confidence Over Bar Index', fontsize=12)

    ax2.plot(bar_indices, our_confs, 'b-', linewidth=2, label='Our Model', marker='o', markersize=4)
    ax2.plot(bar_indices, orig_confs, 'm-', linewidth=2, label='Original', marker='s', markersize=4)

    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Min (0.25)')
    ax2.axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='High (0.80)')

    ax2.set_xlim(0, n_bars)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Bar Index', fontsize=11)
    ax2.set_ylabel('Confidence', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Stats
    our_first = next((bar_indices[i] for i, c in enumerate(our_confs) if c >= 0.25), None)
    orig_first = next((bar_indices[i] for i, c in enumerate(orig_confs) if c >= 0.25), None)
    our_high = next((bar_indices[i] for i, c in enumerate(our_confs) if c >= 0.80), None)
    orig_high = next((bar_indices[i] for i, c in enumerate(orig_confs) if c >= 0.80), None)

    stats = f'Our: First@{our_first}, High@{our_high}, Max={max(our_confs):.2f}\n'
    stats += f'Orig: First@{orig_first}, High@{orig_high}, Max={max(orig_confs):.2f}'
    ax2.text(0.98, 0.95, stats, transform=ax2.transAxes, fontsize=9,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path.name}')
    return True

if __name__ == '__main__':
    # Historical data
    hist_dir = Path('historical_data')
    hist_files = {f.name.split('_')[0]: f for f in hist_dir.glob('*_1min_rth.csv')}

    # 10 examples
    examples = [
        ('IWM', '2020-03-02'),
        ('QQQ', '2024-04-19'),
        ('IWM', '2024-11-06'),
        ('NVDA', '2023-05-25'),
        ('IWM', '2023-08-17'),
        ('QQQ', '2023-10-27'),
        ('NVDA', '2024-01-16'),
        ('IWM', '2022-09-08'),
        ('NVDA', '2022-01-05'),
        ('IWM', '2024-08-05'),
    ]

    print('Generating comparison charts with bounding boxes...')
    for ticker, date_str in examples:
        if ticker in hist_files:
            print(f'Processing {ticker} {date_str}...')
            output_path = output_dir / f'{ticker}_{date_str}_comparison.png'
            create_full_comparison(ticker, date_str, hist_files[ticker], output_path)

    print('\nDone!')
