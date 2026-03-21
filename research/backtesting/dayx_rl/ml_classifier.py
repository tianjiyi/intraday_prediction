"""ML binary classifiers (XGBoost + LogReg) for signal filtering.

Compares supervised ML approach against RL (PPO) for accept/reject decisions.
Uses the same 15 features and real sequential backtest for evaluation.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from .compare import _prepare_data, run_unfiltered, run_filtered_simulation
from .config import RLFilterConfig
from .features import build_signal_observation, N_FEATURES

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]
_out_dir = _root / "data" / "backtests" / "dayx_rl"


def _simulate_signal(entry_ts, signal_row, day_df, dcfg):
    """Simulate a single signal independently. Returns R-multiple."""
    from research.backtesting.dayx.strategy import PositionManager

    pm = PositionManager(dcfg)
    atr_val = signal_row.get("atr", 0)
    if atr_val <= 0:
        return 0.0

    entered = pm.try_enter(
        timestamp=entry_ts,
        price=signal_row["close"],
        direction=signal_row["signal_direction"],
        strategy=signal_row["signal_name"],
        atr_val=atr_val,
        vwap_price=signal_row.get("vwap", 0.0),
        bb_upper=signal_row.get("bb_upper", 0.0),
        bb_lower=signal_row.get("bb_lower", 0.0),
    )
    if not entered:
        return 0.0

    entry_price = signal_row["close"]
    risk = abs(entry_price - pm.position.trade.stop_price)
    if risk <= 0:
        return 0.0

    flatten_h, flatten_m = map(int, dcfg.eod_flatten_time.split(":"))
    entry_loc = day_df.index.get_loc(entry_ts)
    remaining_bars = day_df.iloc[entry_loc + 1:]

    for ts, row in remaining_bars.iterrows():
        is_eod = (ts.hour > flatten_h or
                  (ts.hour == flatten_h and ts.minute >= flatten_m))
        closed = pm.check_exits(
            timestamp=ts, high=row["high"], low=row["low"],
            close=row["close"], is_eod=is_eod,
        )
        if closed:
            break

    if pm.has_position:
        last_ts = day_df.index[-1]
        pm._close_position(last_ts, day_df.iloc[-1]["close"], "backtest_end")

    trade = pm.closed_trades[0]
    rt_costs = 2 * (dcfg.commission_per_share + dcfg.slippage_per_share) * trade.size
    return (trade.pnl - rt_costs) / (risk * trade.size) if risk > 0 else 0.0


def generate_dataset(cfg: RLFilterConfig, split: str = "train"):
    """
    Build labeled dataset: features + R-multiple label for each signal.

    Returns (X, y, r_multiples) where:
        X: np.ndarray (n_signals, 14)
        y: np.ndarray (n_signals,) — 1 if R > 0, else 0
        r_multiples: np.ndarray (n_signals,) — raw R-multiples
    """
    from dataclasses import replace

    dcfg = cfg.dayx_config
    if split == "train":
        start, end = cfg.train_start, cfg.train_end
    elif split == "val":
        start, end = cfg.val_start, cfg.val_end
    else:
        start, end = cfg.test_start, cfg.test_end

    # Load full cached data and slice
    from research.backtesting.dayx.data_loader import (
        fetch_bars_cached, filter_rth, add_session_markers, compute_opening_range,
    )
    from research.backtesting.dayx.indicators import compute_all
    from research.backtesting.dayx.signals import generate_signals

    data_cfg = replace(dcfg, start_date="2016-01-01", end_date="2025-12-31")
    df = fetch_bars_cached(data_cfg)
    df = df.loc[start:end]
    df = filter_rth(df, data_cfg)
    df = add_session_markers(df, data_cfg)
    df = compute_opening_range(df, data_cfg)
    df = compute_all(df, data_cfg)
    df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()
    df = generate_signals(df, data_cfg)

    features = []
    r_multiples = []

    for date, group in df.groupby("session_date"):
        if len(group) < 20:
            continue
        signals = group[group["signal_any"]].copy()
        for i in range(len(signals)):
            row = signals.iloc[i]
            ts = signals.index[i]
            signal_name = row["signal_name"]
            volume_sma = row.get("volume_sma_20", 1.0)

            obs = build_signal_observation(row, signal_name, volume_sma)
            r_mult = _simulate_signal(ts, row, group, dcfg)

            features.append(obs)
            r_multiples.append(r_mult)

    X = np.array(features, dtype=np.float32)
    r_multiples = np.array(r_multiples, dtype=np.float32)
    y = (r_multiples > 0).astype(np.int32)

    logger.info(f"Dataset [{split}]: {len(X)} signals, "
                f"{y.sum()} positive ({y.mean()*100:.1f}%), "
                f"mean R={r_multiples.mean():.3f}")
    return X, y, r_multiples


def train_classifiers(X_train, y_train, X_val, y_val, r_val=None):
    """
    Train XGBoost and Logistic Regression classifiers.

    Uses class weights to handle imbalance, then finds optimal probability
    threshold on val set that maximizes mean R of accepted signals.
    """
    models = {}

    # Class weight: ratio of negative to positive samples
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / max(pos_count, 1)

    # XGBoost with balanced class weights
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models["XGBoost"] = xgb

    # Logistic Regression with balanced class weights
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0,
                            class_weight="balanced")
    lr.fit(X_train, y_train)
    models["LogReg"] = lr

    # Find optimal threshold on val set
    # Instead of default 0.5, sweep thresholds to find best accept rate
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]  # P(accept)

        print(f"\n  {name} — Threshold sweep on val set:")
        print(f"    {'Threshold':>10} {'Accept%':>8} {'WinRate':>8} {'MeanR':>8}")
        print(f"    {'-'*38}")

        best_threshold = 0.5
        best_mean_r = -999

        for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            y_pred = (probs >= thresh).astype(int)
            accept_mask = y_pred == 1
            n_accept = accept_mask.sum()
            if n_accept == 0:
                continue
            accept_rate = n_accept / len(y_val) * 100
            win_rate = y_val[accept_mask].mean() * 100

            if r_val is not None:
                mean_r = r_val[accept_mask].mean()
            else:
                mean_r = 0

            marker = ""
            if mean_r > best_mean_r:
                best_mean_r = mean_r
                best_threshold = thresh
                marker = " <-- best"

            print(f"    {thresh:>10.2f} {accept_rate:>7.0f}% {win_rate:>7.1f}% "
                  f"{mean_r:>8.3f}{marker}")

        # Store threshold on the model object
        model.optimal_threshold_ = best_threshold
        print(f"    Selected threshold: {best_threshold:.2f}")

    return models


def run_ml_filtered_simulation(cfg: RLFilterConfig, model, split: str = "test"):
    """Run ML-filtered backtest using the real sequential simulator.

    Uses predict_proba with the model's optimal threshold (from val tuning).
    """
    from research.backtesting.dayx.simulator import run_simulation
    from research.backtesting.dayx.metrics import compute_metrics

    df, data_cfg = _prepare_data(cfg, split)
    threshold = getattr(model, "optimal_threshold_", 0.5)

    signal_mask = df["signal_any"].fillna(False).astype(bool)
    signal_rows = df[signal_mask]

    total_signals = len(signal_rows)
    accepted = 0
    rejected = 0

    for idx in signal_rows.index:
        row = df.loc[idx]
        signal_name = row["signal_name"]
        volume_sma = row.get("volume_sma_20", 1.0)
        obs = build_signal_observation(row, signal_name, volume_sma)

        prob = model.predict_proba(obs.reshape(1, -1))[0, 1]
        if prob < threshold:  # reject
            df.at[idx, "signal_any"] = False
            rejected += 1
        else:
            accepted += 1

    print(f"  ML Filter (thresh={threshold:.2f}): {accepted}/{total_signals} accepted "
          f"({accepted/max(total_signals,1)*100:.0f}%), {rejected} rejected")

    trades, eq = run_simulation(df, data_cfg)
    metrics = compute_metrics(trades, eq, data_cfg.initial_capital)

    return trades, eq, metrics, {"accepted": accepted, "rejected": rejected,
                                  "total_signals": total_signals}


def compare_all(cfg: RLFilterConfig, rl_model_path: str = None, split: str = "test"):
    """4-way comparison: Unfiltered vs RL vs XGBoost vs LogReg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if rl_model_path is None:
        rl_model_path = str(_out_dir / "models" / "final_model")

    # Generate datasets
    print("Generating training dataset...")
    X_train, y_train, r_train = generate_dataset(cfg, "train")
    print("Generating validation dataset...")
    X_val, y_val, r_val = generate_dataset(cfg, "val")

    # Train ML models
    print("\nTraining classifiers...")
    models = train_classifiers(X_train, y_train, X_val, y_val, r_val=r_val)

    # Feature importance (XGBoost)
    feature_names = [
        "bb_position", "cci14", "cci20", "vwap_dist", "atr_pct",
        "exhaust_down", "exhaust_up", "hammer", "bull_engulf",
        "time_of_day", "rel_volume", "signal_type", "bb_lower_dist", "bb_upper_dist",
        "rsi",
    ]
    xgb_imp = models["XGBoost"].feature_importances_
    print("\n  XGBoost Feature Importance:")
    for name, imp in sorted(zip(feature_names, xgb_imp), key=lambda x: -x[1]):
        bar = "#" * int(imp * 100)
        print(f"    {name:<15} {imp:.3f} {bar}")

    # Run all 4 backtests
    print(f"\n{'='*80}")
    print(f"4-WAY COMPARISON — {split.upper()} SPLIT")
    print(f"(All use identical sequential backtest with position gating)")
    print(f"{'='*80}")

    results = {}

    print("\n--- 1. Unfiltered ---")
    uf_trades, uf_eq, uf_metrics = run_unfiltered(cfg, split)
    results["Unfiltered"] = (uf_trades, uf_eq, uf_metrics, None)

    print("\n--- 2. RL-Filtered (PPO) ---")
    rl_trades, rl_eq, rl_metrics, rl_stats = run_filtered_simulation(
        cfg, rl_model_path, split
    )
    results["RL (PPO)"] = (rl_trades, rl_eq, rl_metrics, rl_stats)

    for name, model in models.items():
        print(f"\n--- {len(results)+1}. {name}-Filtered ---")
        ml_trades, ml_eq, ml_metrics, ml_stats = run_ml_filtered_simulation(
            cfg, model, split
        )
        results[name] = (ml_trades, ml_eq, ml_metrics, ml_stats)

    # Print comparison table
    names = list(results.keys())
    print(f"\n{'METRIC':<20}", end="")
    for n in names:
        print(f"{n:>15}", end="")
    print()
    print("-" * (20 + 15 * len(names)))

    rows = [
        ("Trade count", "trade_count", "{:>15}"),
        ("Win rate", "win_rate", "{:>14.1f}%"),
        ("Total P&L", "total_pnl", "${:>14,.0f}"),
        ("Sharpe", "sharpe", "{:>15.2f}"),
        ("Profit Factor", "profit_factor", "{:>15.2f}"),
        ("Max DD", "max_drawdown_pct", "{:>14.1f}%"),
    ]

    for label, key, fmt in rows:
        print(f"{label:<20}", end="")
        for n in names:
            val = results[n][2].get(key, 0)
            print(fmt.format(val), end="")
        print()

    # Filter stats
    print(f"\n{'Accept rate':<20}", end="")
    for n in names:
        stats = results[n][3]
        if stats:
            rate = stats["accepted"] / max(stats["total_signals"], 1) * 100
            print(f"{rate:>14.0f}%", end="")
        else:
            print(f"{'100%':>15}", end="")
    print()

    # Equity curves
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = {"Unfiltered": "#888888", "RL (PPO)": "#2196F3",
              "XGBoost": "#4CAF50", "LogReg": "#FF9800"}

    for name, (trades, eq, metrics, _) in results.items():
        eq_df = pd.DataFrame(eq, columns=["timestamp", "equity"]).set_index("timestamp")
        daily = eq_df["equity"].resample("D").last().dropna()
        label = f"{name} (S={metrics.get('sharpe', 0):.2f}, PnL=${metrics.get('total_pnl', 0):,.0f})"
        ax.plot(daily.index, daily.values, label=label, linewidth=1.5,
                color=colors.get(name, "#666666"))

    ax.set_title(f"Signal Filter Comparison — {split.upper()} Split", fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    out_path = _out_dir / "ml_comparison.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nEquity curve saved to {out_path}")
    print(f"{'='*80}")

    return results
