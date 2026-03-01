"""
Example Usage of Backtesting Framework

This script demonstrates how to use the backtesting system to evaluate
the Kronos model's predictions.
"""

import yaml
from pathlib import Path
from backtesting import BacktestEngine, BacktestMetrics, BacktestVisualizer


def example_simple_backtest():
    """
    Simple example: Run a backtest and print metrics.
    """
    print("=" * 80)
    print("EXAMPLE 1: Simple Backtest")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent / "config_backtest.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize engine
    engine = BacktestEngine(config)

    # Run backtest for QQQ (last 5 days)
    print("\nRunning backtest for QQQ (1Min)...")
    results_df = engine.run_backtest(
        symbol="QQQ", timeframe="1Min", start_date="2025-09-25", end_date="2025-10-03", step_size_minutes=60
    )

    print(f"\nGenerated {len(results_df)} predictions")
    print(f"\nSample results:")
    print(results_df.head())

    # Calculate metrics
    metrics_calc = BacktestMetrics()

    predictions = results_df[["timestamp", "predicted_price", "prob_up", "current_price"]]
    actuals = results_df[["timestamp", "actual_price"]]

    metrics = metrics_calc.calculate_all_metrics(predictions, actuals, threshold=0.0)

    # Print summary
    print("\n" + metrics_calc.get_summary_report())

    # Save results
    output_dir = Path(__file__).parent / "results" / "example_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"\nResults saved to: {output_dir}")

    return results_df, metrics


def example_with_visualization():
    """
    Example with full visualization pipeline.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Backtest with Visualizations")
    print("=" * 80)

    # Run backtest
    results_df, metrics = example_simple_backtest()

    # Create visualizer
    visualizer = BacktestVisualizer(style="dark_background", dpi=150)

    output_dir = Path(__file__).parent / "results" / "example_output"

    # Generate all plots
    print("\nGenerating visualizations...")

    print("  - ROC Curve...")
    visualizer.plot_roc_curve(metrics, save_path=str(output_dir / "roc_curve.png"))

    print("  - Confusion Matrix...")
    visualizer.plot_confusion_matrix(metrics, save_path=str(output_dir / "confusion_matrix.png"))

    print("  - Prediction vs Actual...")
    visualizer.plot_prediction_vs_actual(results_df, save_path=str(output_dir / "pred_vs_actual.png"))

    print("  - Price Trajectories...")
    visualizer.plot_price_trajectory(results_df, n_samples=10, save_path=str(output_dir / "trajectories.png"))

    print("  - Calibration Curve...")
    visualizer.plot_calibration_curve(metrics, save_path=str(output_dir / "calibration.png"))

    print("  - Residual Plot...")
    visualizer.plot_residuals(results_df, save_path=str(output_dir / "residuals.png"))

    print("  - Summary Dashboard...")
    visualizer.create_summary_dashboard(
        results_df, metrics, save_path=str(output_dir / "dashboard.png")
    )

    visualizer.close_all()

    print(f"\nAll visualizations saved to: {output_dir}")
    print("=" * 80)


def example_compare_timeframes():
    """
    Example: Compare model performance across different timeframes.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Timeframes")
    print("=" * 80)

    from backtesting.metrics import compare_models

    # Load config
    config_path = Path(__file__).parent / "config_backtest.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    timeframes = ["1Min", "5Min"]
    all_results = []
    all_metrics = []

    for tf in timeframes:
        print(f"\nBacktesting {tf}...")

        engine = BacktestEngine(config)
        results_df = engine.run_backtest(
            symbol="QQQ",
            timeframe=tf,
            start_date="2025-09-25",
            end_date="2025-10-03",
            step_size_minutes=60 if tf == "1Min" else 120,
        )

        # Calculate metrics
        metrics_calc = BacktestMetrics()
        predictions = results_df[["timestamp", "predicted_price", "prob_up", "current_price"]]
        actuals = results_df[["timestamp", "actual_price"]]
        metrics = metrics_calc.calculate_all_metrics(predictions, actuals)

        all_results.append(results_df)
        all_metrics.append(metrics)

    # Compare
    comparison_df = compare_models(all_metrics, timeframes)

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    print("\nKronos Model Backtesting Examples\n")

    # Example 1: Simple backtest
    try:
        example_simple_backtest()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    # Example 2: With visualizations
    try:
        example_with_visualization()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    # Example 3: Compare timeframes
    try:
        example_compare_timeframes()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    print("\n✅ Examples completed!")
