"""Evaluate trained RL model on test split, collect trades and metrics."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .config import RLConfig
from .env import DipBuyEnv

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]
_out_dir = _root / "data" / "backtests" / "dayx_dip_rl"


def evaluate(cfg: RLConfig, model_path: str = None, split: str = "test",
             deterministic: bool = True):
    """
    Run trained model through a data split and collect trade-level results.

    Returns:
        trades: list of dicts with entry/exit info
        daily_stats: list of dicts with per-day stats
    """
    if model_path is None:
        model_path = str(_out_dir / "models" / "best_model")

    model = PPO.load(model_path)
    env = DipBuyEnv(cfg, mode=split, seed=99)

    trades = []
    daily_stats = []

    for day_idx in range(len(env._dates)):
        obs, info = env.reset()
        date = info["date"]
        day_entries = 0
        day_invalid_buys = 0
        day_rewards = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if "trade_entry" in info:
                day_entries += 1

            if "invalid_buy" in info:
                day_invalid_buys += 1

            if "trade_exit" in info:
                trades.append({
                    "date": date,
                    "entry_price": info.get("entry_price", 0),
                    "exit_price": info.get("exit_price", 0),
                    "exit_reason": info["trade_exit"],
                    "r_multiple": info.get("r_multiple", 0),
                })
                day_rewards.append(info.get("r_multiple", 0))

        daily_stats.append({
            "date": date,
            "entries": day_entries,
            "invalid_buys": day_invalid_buys,
            "total_r": sum(day_rewards),
            "trades": len(day_rewards),
        })

    # Summary
    total_trades = len(trades)
    if total_trades > 0:
        r_multiples = [t["r_multiple"] for t in trades]
        exit_reasons = {}
        for t in trades:
            exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1

        avg_entries_per_day = np.mean([d["entries"] for d in daily_stats])
        days_with_entry = sum(1 for d in daily_stats if d["entries"] > 0)

        print(f"\n{'='*60}")
        print(f"RL EVALUATION — {split.upper()} SPLIT")
        print(f"{'='*60}")
        print(f"  Days: {len(daily_stats)}")
        print(f"  Days with entry: {days_with_entry} ({days_with_entry/len(daily_stats)*100:.0f}%)")
        print(f"  Total trades: {total_trades}")
        print(f"  Avg entries/day: {avg_entries_per_day:.2f}")
        print(f"  Mean R-multiple: {np.mean(r_multiples):.3f}")
        print(f"  Median R-multiple: {np.median(r_multiples):.3f}")
        print(f"  Win rate: {sum(1 for r in r_multiples if r > 0)/total_trades*100:.1f}%")
        print(f"  Total R: {sum(r_multiples):.2f}")
        print(f"  Exit reasons:")
        for reason, count in sorted(exit_reasons.items()):
            print(f"    {reason:20s}: {count} ({count/total_trades*100:.0f}%)")
        print(f"  Avg invalid buys/day: {np.mean([d['invalid_buys'] for d in daily_stats]):.1f}")
        print(f"{'='*60}")
    else:
        print(f"\nNo trades on {split} split.")

    return trades, daily_stats
