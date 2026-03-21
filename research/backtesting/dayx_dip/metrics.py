"""
Performance metrics for DayX Dip backtest.
"""

import pandas as pd
import numpy as np

from .strategy import Trade


def compute_metrics(trades: list[Trade], equity_curve: list[tuple],
                    initial_capital: float = 100_000.0) -> dict:
    """Compute performance metrics for a list of closed trades."""
    if not trades:
        return {"trade_count": 0, "note": "no trades"}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    trade_count = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / trade_count

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Equity curve metrics
    eq_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    eq_df.set_index("timestamp", inplace=True)

    final_equity = eq_df["equity"].iloc[-1] if len(eq_df) > 0 else initial_capital
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    running_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100 if len(drawdown) > 0 else 0

    eq_daily = eq_df["equity"].resample("D").last().dropna()
    daily_returns = eq_daily.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    durations = [
        (t.exit_time - t.entry_time).total_seconds() / 60
        for t in trades if t.exit_time
    ]
    avg_hold_min = np.mean(durations) if durations else 0

    avg_mae = np.mean([t.max_adverse for t in trades])
    avg_mfe = np.mean([t.max_favorable for t in trades])

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_win_loss_ratio": round(avg_win_loss_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "avg_hold_min": round(avg_hold_min, 1),
        "avg_mae": round(avg_mae, 4),
        "avg_mfe": round(avg_mfe, 4),
        "exit_reasons": exit_reasons,
    }


def hourly_pnl(trades: list[Trade]) -> dict[int, float]:
    """P&L grouped by entry hour."""
    by_hour: dict[int, float] = {}
    for t in trades:
        h = t.entry_time.hour
        by_hour[h] = by_hour.get(h, 0) + t.pnl
    return dict(sorted(by_hour.items()))


def weekday_pnl(trades: list[Trade]) -> dict[str, float]:
    """P&L grouped by day of week."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    by_day: dict[str, float] = {}
    for t in trades:
        d = days[t.entry_time.weekday()]
        by_day[d] = by_day.get(d, 0) + t.pnl
    return by_day


def print_summary(overall: dict, by_hour: dict, by_day: dict):
    """Pretty-print backtest results."""
    print("\n" + "=" * 70)
    print("DAYX DIP BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n{'OVERALL':^70}")
    print("-" * 70)
    for k, v in overall.items():
        if k == "exit_reasons":
            print(f"  Exit reasons:")
            for reason, count in v.items():
                print(f"    {reason:20s}: {count}")
        else:
            print(f"  {k:25s}: {v}")

    print(f"\n{'P&L BY HOUR':^70}")
    print("-" * 70)
    for h, pnl in by_hour.items():
        bar = "+" * int(abs(pnl) / 50) if pnl > 0 else "-" * int(abs(pnl) / 50)
        print(f"  {h:02d}:00  ${pnl:>10.2f}  {bar}")

    print(f"\n{'P&L BY DAY':^70}")
    print("-" * 70)
    for d, pnl in by_day.items():
        print(f"  {d}  ${pnl:>10.2f}")

    print("\n" + "=" * 70)
