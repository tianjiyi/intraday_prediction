"""Generate candlestick charts with entry/exit markers for best/worst days."""

import sys
import types
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)

_root = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _root)
sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
sys.modules["research.backtesting"].__path__ = [
    str(Path(_root) / "research" / "backtesting")
]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from research.backtesting.dayx_rl.config import RLFilterConfig, load_dayx_config
from research.backtesting.dayx_rl.compare import _prepare_data
from research.backtesting.dayx_rl.features import build_signal_observation
from research.backtesting.dayx.simulator import run_simulation


def main():
    dayx_cfg = load_dayx_config()
    cfg = RLFilterConfig(dayx_config=dayx_cfg)

    model = PPO.load(str(Path(_root) / "data/backtests/dayx_rl/models/final_model_combo_A"))
    df_full, data_cfg = _prepare_data(cfg, "test")

    # Collect decisions
    signal_mask = df_full["signal_any"].fillna(False).astype(bool)
    signal_rows_all = df_full[signal_mask].copy()

    decisions = {}
    for idx in signal_rows_all.index:
        row = df_full.loc[idx]
        obs = build_signal_observation(
            row, row["signal_name"], row.get("volume_sma_20", 1.0)
        )
        action, _ = model.predict(obs, deterministic=True)
        decisions[idx] = int(action)
        if int(action) == 0:
            df_full.at[idx, "signal_any"] = False

    trades, eq = run_simulation(df_full, data_cfg)

    trades_by_date = defaultdict(list)
    for t in trades:
        trades_by_date[t.entry_time.strftime("%Y-%m-%d")].append(t)

    day_pnl = {d: sum(t.pnl for t in ts) for d, ts in trades_by_date.items()}

    best_days = sorted(day_pnl.items(), key=lambda x: -x[1])[:3]
    worst_days = sorted(day_pnl.items(), key=lambda x: x[1])[:3]
    target_days = best_days + worst_days

    out_dir = Path(_root) / "data" / "backtests" / "dayx_rl"

    # Reload original df for price data (before signal_any was modified)
    df_orig, _ = _prepare_data(cfg, "test")

    for date_str, total_pnl in target_days:
        day_df = df_orig[df_orig.index.strftime("%Y-%m-%d") == date_str].copy()
        if len(day_df) == 0:
            continue

        day_trades = trades_by_date.get(date_str, [])
        day_signals = signal_rows_all[
            signal_rows_all.index.strftime("%Y-%m-%d") == date_str
        ]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 9), height_ratios=[3, 1],
            gridspec_kw={"hspace": 0.08},
        )

        # Candlestick chart
        times = day_df.index
        opens = day_df["open"].values
        highs = day_df["high"].values
        lows = day_df["low"].values
        closes = day_df["close"].values

        colors = [
            "#26a69a" if c >= o else "#ef5350" for o, c in zip(opens, closes)
        ]

        # Draw candle wicks
        for i, t in enumerate(times):
            ax1.plot([t, t], [lows[i], highs[i]], color=colors[i], linewidth=0.8)

        # Draw candle bodies
        bar_width = pd.Timedelta(seconds=40)
        for i, t in enumerate(times):
            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])
            body_h = max(body_high - body_low, 0.01)
            rect = Rectangle(
                (t - bar_width / 2, body_low), bar_width, body_h,
                facecolor=colors[i], edgecolor=colors[i], linewidth=0.5,
            )
            ax1.add_patch(rect)

        # VWAP line
        if "vwap" in day_df.columns:
            ax1.plot(
                times, day_df["vwap"], color="#2196F3",
                linewidth=1.2, alpha=0.7, label="VWAP",
            )

        # Plot rejected signals (small gray X)
        for idx in day_signals.index:
            if decisions.get(idx, 0) == 0:
                ax1.scatter(
                    idx, df_orig.loc[idx, "close"], marker="x",
                    color="gray", s=30, alpha=0.4, zorder=3,
                )

        # Plot trades: entry (triangle) and exit (square) with connecting line
        tc_map = {"W": "#4CAF50", "L": "#F44336"}
        for t in day_trades:
            win = "W" if t.pnl > 0 else "L"
            tc = tc_map[win]

            entry_marker = "^" if t.direction == "long" else "v"
            ax1.scatter(
                t.entry_time, t.entry_price, marker=entry_marker, color=tc,
                s=120, zorder=5, edgecolors="black", linewidth=0.8,
            )
            ax1.scatter(
                t.exit_time, t.exit_price, marker="s", color=tc,
                s=80, zorder=5, edgecolors="black", linewidth=0.8,
            )
            ax1.plot(
                [t.entry_time, t.exit_time], [t.entry_price, t.exit_price],
                color=tc, linewidth=1.5, alpha=0.6, linestyle="--",
            )

            pnl_str = f"${t.pnl:+,.0f}\n{t.exit_reason}"
            ax1.annotate(
                pnl_str,
                xy=(t.exit_time, t.exit_price),
                xytext=(10, 15 if t.pnl > 0 else -25),
                textcoords="offset points",
                fontsize=8, fontweight="bold", color=tc,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor=tc, alpha=0.9,
                ),
            )

        # Volume bars
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350" for o, c in zip(opens, closes)
        ]
        ax2.bar(times, day_df["volume"], width=bar_width, color=vol_colors, alpha=0.6)

        # Formatting
        tag = "WINNER" if total_pnl > 0 else "LOSER"
        tag_color = "#4CAF50" if total_pnl > 0 else "#F44336"

        n_accepted = sum(1 for idx in day_signals.index if decisions.get(idx, 0) == 1)
        n_total = len(day_signals)
        n_trades = len(day_trades)
        wins = sum(1 for t in day_trades if t.pnl > 0)
        losses = n_trades - wins

        ax1.set_title(
            f"QQQ {date_str}  \u2014  [{tag}] Day PnL: ${total_pnl:+,.2f}  |  "
            f"Signals: {n_accepted}/{n_total} accepted  |  "
            f"Trades: {wins}W/{losses}L",
            fontsize=13, fontweight="bold", color=tag_color,
        )

        ax1.set_ylabel("Price", fontsize=10)
        ax1.grid(True, alpha=0.2)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax2.set_ylabel("Volume", fontsize=10)
        ax2.grid(True, alpha=0.2)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.set_xlabel("Time (ET)", fontsize=10)

        legend_elements = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#4CAF50",
                   markersize=10, label="Long Entry (win)"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#4CAF50",
                   markersize=10, label="Short Entry (win)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
                   markersize=8, label="Exit (win)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#F44336",
                   markersize=10, label="Long Entry (loss)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#F44336",
                   markersize=8, label="Exit (loss)"),
            Line2D([0], [0], marker="x", color="gray", markersize=8,
                   linestyle="None", label="Rejected signal"),
        ]
        ax1.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=2)

        plt.tight_layout()
        fname = f"combo_A_{date_str}.png"
        plt.savefig(str(out_dir / fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")

    print("Done!")


if __name__ == "__main__":
    main()
