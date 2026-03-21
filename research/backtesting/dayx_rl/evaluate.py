"""Evaluate trained RL signal filter — collect accept/reject stats."""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .config import RLFilterConfig
from .env import SignalFilterEnv

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]
_out_dir = _root / "data" / "backtests" / "dayx_rl"


def evaluate(cfg: RLFilterConfig, model_path: str = None, split: str = "test",
             deterministic: bool = True):
    """
    Run trained model through a data split and collect signal-level results.

    Returns:
        signals: list of dicts with per-signal accept/reject and outcome
        daily_stats: list of dicts with per-day summary
    """
    if model_path is None:
        model_path = str(_out_dir / "models" / "best_model")

    model = PPO.load(model_path)
    env = SignalFilterEnv(cfg, mode=split, seed=99)

    all_signals = []
    daily_stats = []

    for day_idx in range(len(env._dates)):
        obs, info = env.reset()
        date = info["date"]
        n_signals = info["n_signals"]
        day_accepted = 0
        day_rejected = 0
        day_r_multiples = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if "action" in info:
                signal_record = {
                    "date": date,
                    "signal_name": info.get("signal_name", ""),
                    "signal_direction": info.get("signal_direction", ""),
                    "signal_time": info.get("signal_time", ""),
                    "action": info["action"],
                    "r_multiple": info.get("r_multiple", 0),
                    "pnl": info.get("pnl", 0),
                    "exit_reason": info.get("exit_reason", ""),
                }
                all_signals.append(signal_record)

                if info["action"] == "accept":
                    day_accepted += 1
                    day_r_multiples.append(info.get("r_multiple", 0))
                else:
                    day_rejected += 1

        daily_stats.append({
            "date": date,
            "total_signals": n_signals,
            "accepted": day_accepted,
            "rejected": day_rejected,
            "total_r": sum(day_r_multiples),
        })

    # Summary
    accepted = [s for s in all_signals if s["action"] == "accept"]
    rejected = [s for s in all_signals if s["action"] == "reject"]

    print(f"\n{'='*60}")
    print(f"RL SIGNAL FILTER — {split.upper()} SPLIT")
    print(f"{'='*60}")
    print(f"  Days: {len(daily_stats)}")
    print(f"  Total signals: {len(all_signals)}")
    print(f"  Accepted: {len(accepted)} ({len(accepted)/max(len(all_signals),1)*100:.0f}%)")
    print(f"  Rejected: {len(rejected)} ({len(rejected)/max(len(all_signals),1)*100:.0f}%)")

    if accepted:
        r_mults = [s["r_multiple"] for s in accepted]
        wins = sum(1 for r in r_mults if r > 0)
        print(f"\n  --- Accepted Trades ---")
        print(f"  Win rate: {wins/len(accepted)*100:.1f}%")
        print(f"  Mean R: {np.mean(r_mults):.3f}")
        print(f"  Median R: {np.median(r_mults):.3f}")
        print(f"  Total R: {sum(r_mults):.2f}")

        # Exit reasons
        reasons = {}
        for s in accepted:
            r = s["exit_reason"]
            reasons[r] = reasons.get(r, 0) + 1
        print(f"  Exit reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"    {reason:20s}: {count} ({count/len(accepted)*100:.0f}%)")

    # Show what was rejected (counterfactual)
    if rejected:
        # Simulate rejected signals to see what would have happened
        print(f"\n  --- Rejected Signals (counterfactual) ---")
        print(f"  (Would need separate simulation to compute counterfactual P&L)")

    print(f"{'='*60}")

    return all_signals, daily_stats
