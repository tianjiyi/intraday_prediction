"""
Kronos Model Backtesting Framework

A comprehensive backtesting system for evaluating time series predictions
from the Kronos model with classification and regression metrics.

Modules:
    backtest_engine: Core backtesting logic with rolling window evaluation
    metrics: Evaluation metrics (F1, AUC-ROC, MAE, RMSE, etc.)
    visualize: Visualization tools for results analysis
    run_backtest: CLI tool for running backtests

Usage:
    from backtesting import BacktestEngine, BacktestMetrics, BacktestVisualizer

    # Load config
    config = yaml.safe_load(open('config_backtest.yaml'))

    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest('QQQ', '1Min', '2025-09-01', '2025-10-01')

    # Calculate metrics
    metrics_calc = BacktestMetrics()
    metrics = metrics_calc.calculate_all_metrics(results[['predicted_price']], results[['actual_price']])

    # Visualize
    viz = BacktestVisualizer()
    viz.plot_roc_curve(metrics, save_path='roc.png')
"""

__version__ = "1.0.0"
__author__ = "Kronos Prediction Team"

from .backtest_engine import BacktestEngine, load_backtest_results
from .metrics import BacktestMetrics, compare_models
from .visualize import BacktestVisualizer

__all__ = [
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestVisualizer",
    "load_backtest_results",
    "compare_models",
]
