"""
Compare v1 (trend-following) vs v2 (mean reversion) strategies.
"""

import sys
import logging
from pathlib import Path

if __name__ == "__main__":
    _root = str(Path(__file__).resolve().parents[3])
    sys.path.insert(0, _root)
    import types
    sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
    sys.modules["research.backtesting"].__path__ = [
        str(Path(_root) / "research" / "backtesting")
    ]

import pandas as pd
import numpy as np

from research.backtesting.dayx.config import DayXConfig
from research.backtesting.dayx.data_loader import load_data
from research.backtesting.dayx.indicators import compute_all
from research.backtesting.dayx.signals import generate_signals
from research.backtesting.dayx.simulator import run_simulation
from research.backtesting.dayx.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_variants() -> dict[str, DayXConfig]:
    variants = {}

    # v1 baseline: original 3 strategies, no filters
    c = DayXConfig(symbol="QQQ", start_date="2020-01-01", end_date="2026-03-13",
                   require_exhaustion=False, strategies=["long_trend", "short_trend", "long_dip"])
    variants["v1_baseline"] = c

    # v1 + time filter + progressive trail (best v1)
    c = DayXConfig(symbol="QQQ", start_date="2020-01-01", end_date="2026-03-13",
                   require_exhaustion=False, strategies=["long_trend", "short_trend", "long_dip"],
                   time_filter=True, trail_mode="progressive", trail_progressive_step=0.5)
    variants["v1_best"] = c

    # v2: mean reversion + trend dip, fixed R targets, wider stops
    c = DayXConfig(symbol="QQQ", start_date="2020-01-01", end_date="2026-03-13",
                   require_exhaustion=False, strategies=["buy_dip", "sell_rip", "trend_dip"],
                   use_vwap_target=False, stop_atr_mult=2.0,
                   time_filter=True, trail_mode="progressive", trail_progressive_step=0.5)
    variants["v2_full"] = c

    return variants


def run_one(label: str, cfg: DayXConfig, df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    df = compute_all(df, cfg)
    df = generate_signals(df, cfg)
    trades, eq = run_simulation(df, cfg)
    m = compute_metrics(trades, eq, cfg.initial_capital)

    # Per-year breakdown
    yearly = {}
    for year in range(2020, 2027):
        yt = [t for t in trades if t.entry_time.year == year]
        if yt:
            yearly[year] = {
                "trades": len(yt),
                "pnl": sum(t.pnl for t in yt),
                "wr": sum(1 for t in yt if t.pnl > 0) / len(yt) * 100,
            }
    m["yearly"] = yearly
    return {"metrics": m, "trades": trades, "equity": eq}


def main():
    variants = make_variants()

    logger.info("Loading data...")
    base = DayXConfig(symbol="QQQ", start_date="2020-01-01", end_date="2026-03-13")
    df_raw = load_data(base)
    logger.info(f"Loaded {len(df_raw)} bars")

    results = {}
    for label, cfg in variants.items():
        logger.info(f"Running {label}...")
        results[label] = run_one(label, cfg, df_raw)

    # Print comparison
    years = list(range(2020, 2027))
    yr_cols = "".join(f" {y} PnL   {y} WR" for y in years)

    print("\n" + "=" * 200)
    print(f"{'VARIANT':<20} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'Ret%':>7} "
          f"{'Sharpe':>7} {'MaxDD%':>8} {'PF':>6} {'Stop%':>7}")
    print("-" * 100)

    for label, res in results.items():
        m = res["metrics"]
        if m["trade_count"] == 0:
            print(f"{label:<20} {'NO TRADES':>7}")
            continue
        stops = m.get("exit_reasons", {}).get("stop_loss", 0)
        stop_pct = stops / m["trade_count"] * 100
        print(f"{label:<20} {m['trade_count']:>7} {m['win_rate']:>5.1f}% "
              f"${m['total_pnl']:>9.0f} {m['total_return_pct']:>6.1f}% "
              f"{m['sharpe']:>7.2f} {m['max_drawdown_pct']:>7.1f}% "
              f"{m['profit_factor']:>5.2f} {stop_pct:>6.1f}%")

    # Per-year P&L breakdown
    print(f"\n{'VARIANT':<20}", end="")
    for y in years:
        print(f" {y:>10}", end="")
    print()
    print("-" * 100)
    for label, res in results.items():
        m = res["metrics"]
        if m["trade_count"] == 0:
            continue
        print(f"{label:<20}", end="")
        for y in years:
            ydata = m.get("yearly", {}).get(y, {})
            pnl = ydata.get("pnl", 0)
            print(f" ${pnl:>9.0f}", end="")
        print()

    print("=" * 100)

    # Per-strategy breakdown for v2
    print("\n--- v2_full per-strategy ---")
    if "v2_full" in results and results["v2_full"]["trades"]:
        for t_strat in sorted(set(t.strategy for t in results["v2_full"]["trades"])):
            st = [t for t in results["v2_full"]["trades"] if t.strategy == t_strat]
            wins = sum(1 for t in st if t.pnl > 0)
            total_pnl = sum(t.pnl for t in st)
            print(f"  {t_strat:15s}: n={len(st):4d}  wr={wins/len(st)*100:5.1f}%  pnl=${total_pnl:>9.0f}")

    # Generate chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import datetime
        from datetime import timezone

        fig, ax = plt.subplots(figsize=(16, 8))

        colors_map = {
            "v1_baseline": "#666666",
            "v1_best": "#aaaaaa",
            "v2_full": "#00ff88",
        }
        styles = {
            "v1_baseline": ("--", 1.5),
            "v1_best": ("--", 1.5),
            "v2_full": ("-", 3.0),
        }

        for label, res in results.items():
            trades = res["trades"]
            if not trades:
                continue
            tdf = pd.DataFrame([{"exit_time": t.exit_time, "pnl": t.pnl} for t in trades])
            tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)
            tdf = tdf.sort_values("exit_time")
            tdf["cum_pnl"] = tdf["pnl"].cumsum()

            ls, lw = styles.get(label, ("-", 1.0))
            ax.plot(tdf["exit_time"].dt.to_pydatetime(), tdf["cum_pnl"].values,
                    color=colors_map.get(label, "gray"), linewidth=lw,
                    linestyle=ls, alpha=0.9, label=label)

        for yr in range(2021, 2027):
            ax.axvline(x=datetime.datetime(yr, 1, 1, tzinfo=timezone.utc),
                       color="#555555", linestyle=":", alpha=0.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.set_ylabel("Cumulative P&L ($)", fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333333")

        fig.suptitle("Day_X v1 vs v2 — QQQ 2020-2026",
                     fontsize=15, fontweight="bold", color="white")
        plt.tight_layout()

        out_dir = Path(_root) / "data" / "backtests" / "dayx"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "v1_vs_v2_vs_v21_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Chart saved to {out}")

    except Exception as e:
        logger.warning(f"Could not generate chart: {e}")


if __name__ == "__main__":
    main()
