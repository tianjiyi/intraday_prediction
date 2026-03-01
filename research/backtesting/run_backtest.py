"""
CLI Tool for Running Kronos Model Backtests

Usage:
    python run_backtest.py --symbol QQQ --timeframe 1Min --start-date 2025-09-01 --end-date 2025-10-01
    python run_backtest.py --config config_backtest.yaml
    python run_backtest.py --symbol SPY --quick  # Quick test with default settings
"""

import argparse
import logging
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import BacktestEngine
from metrics import BacktestMetrics
from visualize import BacktestVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def run_backtest_pipeline(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    config: Dict,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run complete backtest pipeline.

    Args:
        symbol: Trading symbol
        timeframe: Bar timeframe
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Backtesting configuration
        output_dir: Output directory for results

    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 80)
    logger.info("STARTING BACKTEST PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("=" * 80)

    # Create output directory
    if output_dir is None:
        output_dir = config.get("output", {}).get("base_dir", "./backtesting/results")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"{symbol}_{timeframe}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {run_dir}")

    # Step 1: Run Backtest
    logger.info("\n[1/4] Running backtest engine...")
    engine = BacktestEngine(config)

    step_size = config.get("backtest", {}).get("rolling_window", {}).get("step_size_minutes", 30)

    results_df = engine.run_backtest(symbol, timeframe, start_date, end_date, step_size)

    if results_df.empty:
        logger.error("No results generated. Exiting.")
        return {"success": False, "error": "No predictions generated"}

    # Save raw results
    csv_path = run_dir / "predictions.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # Step 2: Calculate Metrics
    logger.info("\n[2/4] Calculating metrics...")
    metrics_calculator = BacktestMetrics()

    # Prepare dataframes for metrics
    predictions_df = results_df[
        ["timestamp", "predicted_price", "prob_up", "current_price"]
    ].copy()
    actuals_df = results_df[["timestamp", "actual_price"]].copy()

    threshold = config.get("metrics", {}).get("classification", {}).get("threshold", 0.0)
    metrics = metrics_calculator.calculate_all_metrics(
        predictions_df, actuals_df, threshold
    )

    # Print summary
    print("\n" + metrics_calculator.get_summary_report())

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Step 3: Generate Visualizations
    logger.info("\n[3/4] Generating visualizations...")
    visualizer = BacktestVisualizer(
        style=config.get("visualization", {}).get("style", "dark_background"),
        dpi=config.get("visualization", {}).get("dpi", 300),
    )

    plot_config = config.get("visualization", {}).get("plots", {})

    # ROC Curve
    if plot_config.get("roc_curve", True):
        logger.info("  - Creating ROC curve...")
        roc_path = run_dir / "roc_curve.png"
        visualizer.plot_roc_curve(metrics, save_path=str(roc_path))

    # Confusion Matrix
    if plot_config.get("confusion_matrix", True):
        logger.info("  - Creating confusion matrix...")
        cm_path = run_dir / "confusion_matrix.png"
        visualizer.plot_confusion_matrix(metrics, save_path=str(cm_path))

    # Prediction vs Actual
    if plot_config.get("prediction_vs_actual", True):
        logger.info("  - Creating prediction vs actual plot...")
        pred_actual_path = run_dir / "prediction_vs_actual.png"
        visualizer.plot_prediction_vs_actual(results_df, save_path=str(pred_actual_path))

    # Price Trajectory
    if plot_config.get("price_trajectory", True):
        logger.info("  - Creating price trajectory plot...")
        traj_path = run_dir / "price_trajectory.png"
        visualizer.plot_price_trajectory(results_df, save_path=str(traj_path))

    # Calibration Curve
    if plot_config.get("calibration_curve", True):
        logger.info("  - Creating calibration curve...")
        calib_path = run_dir / "calibration_curve.png"
        visualizer.plot_calibration_curve(metrics, save_path=str(calib_path))

    # Residual Plot
    if plot_config.get("residual_plot", True):
        logger.info("  - Creating residual plot...")
        resid_path = run_dir / "residual_plot.png"
        visualizer.plot_residuals(results_df, save_path=str(resid_path))

    # Summary Dashboard
    logger.info("  - Creating summary dashboard...")
    dashboard_path = run_dir / "dashboard.png"
    visualizer.create_summary_dashboard(
        results_df, metrics, save_path=str(dashboard_path)
    )

    visualizer.close_all()

    # Step 4: Generate Summary Report
    logger.info("\n[4/4] Generating summary report...")
    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(metrics_calculator.get_summary_report())
    logger.info(f"Summary report saved to: {summary_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total Predictions: {len(results_df)}")
    logger.info(f"F1 Score: {metrics.get('classification', {}).get('f1_score', 0):.4f}")
    logger.info(f"AUC-ROC: {metrics.get('classification', {}).get('auc_roc', 0):.4f}")
    logger.info(f"MAE: {metrics.get('regression', {}).get('mae', 0):.4f}")
    logger.info(f"RMSE: {metrics.get('regression', {}).get('rmse', 0):.4f}")
    logger.info(f"\nAll results saved to: {run_dir}")
    logger.info("=" * 80)

    return {
        "success": True,
        "results_df": results_df,
        "metrics": metrics,
        "output_dir": str(run_dir),
    }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run Kronos Model Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python run_backtest.py --symbol QQQ --timeframe 1Min --start-date 2025-09-01 --end-date 2025-10-01

  # Use config file
  python run_backtest.py --config config_backtest.yaml

  # Quick test (last 5 days)
  python run_backtest.py --symbol SPY --quick

  # Multiple symbols
  python run_backtest.py --symbols QQQ SPY AAPL --timeframe 5Min --start-date 2025-09-01

  # Custom output directory
  python run_backtest.py --symbol TSLA --output-dir ./my_results
        """,
    )

    # Required arguments (or config file)
    parser.add_argument(
        "--config", "-c", type=str, help="Path to config YAML file (overrides other args)"
    )

    # Symbol and timeframe
    parser.add_argument("--symbol", "-s", type=str, help="Trading symbol (e.g., QQQ, SPY)")
    parser.add_argument(
        "--symbols", type=str, nargs="+", help="Multiple symbols to backtest"
    )
    parser.add_argument(
        "--timeframe",
        "-t",
        type=str,
        choices=["1Min", "5Min", "15Min", "30Min"],
        help="Bar timeframe",
    )

    # Date range
    parser.add_argument(
        "--start-date", type=str, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test: last 5 trading days"
    )

    # Output
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for results")

    # Options
    parser.add_argument(
        "--step-size",
        type=int,
        default=30,
        help="Minutes between predictions (default: 30)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip visualization generation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging (DEBUG level)"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        # Use default config
        default_config_path = Path(__file__).parent / "config_backtest.yaml"
        if default_config_path.exists():
            config = load_config(str(default_config_path))
        else:
            logger.warning("No config file found, using minimal defaults")
            config = {
                "backtest": {"horizon_minutes": 30, "rolling_window": {"step_size_minutes": 30}},
                "metrics": {"classification": {"threshold": 0.0}},
                "visualization": {"plots": {}},
                "output": {"base_dir": "./backtesting/results"},
            }

    # Determine symbols to test
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    elif not args.config:
        parser.error("Must specify --symbol, --symbols, or --config")
    else:
        # Use symbols from config
        symbols = config.get("backtest", {}).get("symbols", ["QQQ"])

    # Determine timeframe
    if args.timeframe:
        timeframe = args.timeframe
    else:
        timeframes = config.get("backtest", {}).get("timeframes", ["1Min"])
        timeframe = timeframes[0]

    # Determine date range
    if args.quick:
        # Use recent historical dates (4 trading days in September 2025)
        # Note: Using dates where market data is available
        start_date_str = "2025-09-02"  # Tuesday (after Labor Day)
        end_date_str = "2025-09-05"    # Friday (4 trading days)
    elif args.start_date and args.end_date:
        start_date_str = args.start_date
        end_date_str = args.end_date
    else:
        # Use config dates
        start_date_str = config.get("backtest", {}).get("start_date", "2025-09-02")
        end_date_str = config.get("backtest", {}).get("end_date", "2025-09-05")

    # Override config with CLI args
    if args.step_size:
        config.setdefault("backtest", {}).setdefault("rolling_window", {})[
            "step_size_minutes"
        ] = args.step_size

    if args.no_plots:
        config.setdefault("visualization", {})["plots"] = {}

    # Run backtest for each symbol
    all_results = []
    for symbol in symbols:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BACKTESTING {symbol}")
        logger.info(f"{'=' * 80}\n")

        result = run_backtest_pipeline(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date_str,
            end_date=end_date_str,
            config=config,
            output_dir=args.output_dir,
        )

        all_results.append(result)

    # Summary across all symbols
    if len(symbols) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-SYMBOL SUMMARY")
        logger.info("=" * 80)
        for i, (symbol, result) in enumerate(zip(symbols, all_results)):
            if result.get("success"):
                metrics = result["metrics"]
                logger.info(f"\n{symbol}:")
                logger.info(
                    f"  F1 Score: {metrics.get('classification', {}).get('f1_score', 0):.4f}"
                )
                logger.info(
                    f"  AUC-ROC:  {metrics.get('classification', {}).get('auc_roc', 0):.4f}"
                )
                logger.info(f"  MAE:      {metrics.get('regression', {}).get('mae', 0):.4f}")
        logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nBacktest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nBacktest failed with error: {e}", exc_info=True)
        sys.exit(1)
