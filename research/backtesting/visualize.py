"""
Visualization Module for Backtesting Results

Creates publication-quality plots for model evaluation:
- ROC curves
- Confusion matrices
- Prediction vs Actual scatter plots
- Price trajectory comparisons
- Calibration curves
- Residual plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Create visualizations for backtesting results"""

    def __init__(self, style: str = "dark_background", dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style (default: dark_background)
            dpi: DPI for saved images (default: 300)
        """
        self.style = style
        self.dpi = dpi

        # Set style
        plt.style.use(self.style)

        # Set seaborn defaults
        sns.set_palette("husl")

    def plot_roc_curve(
        self, metrics: Dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve with AUC score.

        Args:
            metrics: Metrics dictionary containing ROC curve data
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract ROC data
        roc_data = metrics.get("classification", {}).get("roc_curve")
        auc_score = metrics.get("classification", {}).get("auc_roc")

        if roc_data is None:
            logger.warning("No ROC curve data available")
            return fig

        fpr = np.array(roc_data["fpr"])
        tpr = np.array(roc_data["tpr"])

        # Plot ROC curve
        ax.plot(
            fpr, tpr, color="#2962FF", linewidth=2, label=f"ROC (AUC = {auc_score:.3f})"
        )

        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")

        # Labels and title
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve - Direction Prediction", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_confusion_matrix(
        self, metrics: Dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.

        Args:
            metrics: Metrics dictionary containing confusion matrix
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract confusion matrix
        cm_data = metrics.get("classification", {}).get("confusion_matrix", {})

        cm = np.array(
            [[cm_data.get("tn", 0), cm_data.get("fp", 0)], [cm_data.get("fn", 0), cm_data.get("tp", 0)]]
        )

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            square=True,
            ax=ax,
            annot_kws={"size": 16},
        )

        # Labels
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xticklabels(["Down", "Up"])
        ax.set_yticklabels(["Down", "Up"])

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_prediction_vs_actual(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot predicted vs actual prices (scatter plot).

        Args:
            df: DataFrame with 'predicted_price' and 'actual_price' columns
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(
            df["actual_price"],
            df["predicted_price"],
            alpha=0.6,
            s=30,
            color="#2962FF",
            edgecolors="white",
            linewidth=0.5,
        )

        # Perfect prediction line (y=x)
        min_val = min(df["actual_price"].min(), df["predicted_price"].min())
        max_val = max(df["actual_price"].max(), df["predicted_price"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

        # Labels and title
        ax.set_xlabel("Actual Price", fontsize=12)
        ax.set_ylabel("Predicted Price", fontsize=12)
        ax.set_title("Predicted vs Actual Prices", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(df["actual_price"], df["predicted_price"])[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="gray", alpha=0.3),
        )

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_price_trajectory(
        self, df: pd.DataFrame, n_samples: int = 20, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot price trajectories over time (sample of predictions).

        Args:
            df: DataFrame with timestamp, current_price, predicted_price, actual_price
            n_samples: Number of sample predictions to plot
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Sample random predictions
        if len(df) > n_samples:
            sample_df = df.sample(n=n_samples, random_state=42)
        else:
            sample_df = df

        # Plot each prediction
        for idx, row in sample_df.iterrows():
            timestamp = row["timestamp"]
            outcome_time = row.get("outcome_timestamp", timestamp + pd.Timedelta(minutes=30))

            # Plot current -> predicted (blue line)
            ax.plot(
                [timestamp, outcome_time],
                [row["current_price"], row["predicted_price"]],
                "b-",
                alpha=0.3,
                linewidth=1,
            )

            # Plot current -> actual (green line)
            ax.plot(
                [timestamp, outcome_time],
                [row["current_price"], row["actual_price"]],
                "g-",
                alpha=0.5,
                linewidth=1.5,
            )

        # Create custom legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="b", linewidth=2, alpha=0.5, label="Predicted"),
            Line2D([0], [0], color="g", linewidth=2, alpha=0.7, label="Actual"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        # Labels and title
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_title(
            f"Price Trajectories (Sample of {len(sample_df)} predictions)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_calibration_curve(
        self, metrics: Dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot calibration curve (predicted probability vs observed frequency).

        Args:
            metrics: Metrics dictionary containing calibration data
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract calibration data
        calib_data = metrics.get("probabilistic", {}).get("calibration", {})

        if not calib_data:
            logger.warning("No calibration data available")
            return fig

        bin_centers = calib_data.get("bin_centers", [])
        observed_freq = calib_data.get("observed_frequency", [])
        expected_prob = calib_data.get("expected_probability", [])

        # Filter out None values manually (before converting to numpy arrays)
        valid_data = [(bc, of, ep) for bc, of, ep in zip(bin_centers, observed_freq, expected_prob)
                      if of is not None and ep is not None]

        if not valid_data:
            logger.warning("No valid calibration data to plot")
            return fig

        bin_centers, observed_freq, expected_prob = zip(*valid_data)

        # Now convert to numpy arrays (all values are guaranteed to be numbers)
        bin_centers = np.array(bin_centers)
        observed_freq = np.array(observed_freq)
        expected_prob = np.array(expected_prob)

        # Plot calibration curve
        ax.plot(
            expected_prob,
            observed_freq,
            "s-",
            color="#2962FF",
            linewidth=2,
            markersize=8,
            label="Model Calibration",
        )

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")

        # Labels and title
        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Observed Frequency", fontsize=12)
        ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        # Add ECE (Expected Calibration Error)
        ece = calib_data.get("expected_calibration_error")
        if ece is not None:
            ax.text(
                0.95,
                0.05,
                f"ECE: {ece:.4f}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="gray", alpha=0.3),
            )

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_residuals(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot residual analysis (prediction errors).

        Args:
            df: DataFrame with predicted_price and actual_price
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Calculate residuals
        residuals = df["actual_price"] - df["predicted_price"]

        # Plot 1: Residuals vs Predicted
        ax1.scatter(
            df["predicted_price"],
            residuals,
            alpha=0.6,
            s=30,
            color="#2962FF",
            edgecolors="white",
            linewidth=0.5,
        )
        ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax1.set_xlabel("Predicted Price", fontsize=12)
        ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
        ax1.set_title("Residual Plot", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residual histogram
        ax2.hist(residuals, bins=30, color="#2962FF", alpha=0.7, edgecolor="white")
        ax2.axvline(x=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("Residuals", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Residual Distribution", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        ax2.text(
            0.95,
            0.95,
            f"Mean: {mean_res:.4f}\nStd: {std_res:.4f}",
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="gray", alpha=0.3),
        )

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_summary_dashboard(
        self, df: pd.DataFrame, metrics: Dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with all key plots.

        Args:
            df: Results DataFrame
            metrics: Metrics dictionary
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure with multiple subplots
        """
        fig = plt.figure(figsize=(20, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        roc_data = metrics.get("classification", {}).get("roc_curve")
        auc_score = metrics.get("classification", {}).get("auc_roc")
        if roc_data:
            fpr = np.array(roc_data["fpr"])
            tpr = np.array(roc_data["tpr"])
            ax1.plot(fpr, tpr, color="#2962FF", linewidth=2)
            ax1.plot([0, 1], [0, 1], "r--", linewidth=1)
            ax1.set_title(f"ROC Curve (AUC={auc_score:.3f})", fontsize=10, fontweight="bold")
            ax1.set_xlabel("FPR", fontsize=9)
            ax1.set_ylabel("TPR", fontsize=9)
            ax1.grid(True, alpha=0.3)

        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        cm_data = metrics.get("classification", {}).get("confusion_matrix", {})
        cm = np.array(
            [[cm_data.get("tn", 0), cm_data.get("fp", 0)], [cm_data.get("fn", 0), cm_data.get("tp", 0)]]
        )
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax2)
        ax2.set_title("Confusion Matrix", fontsize=10, fontweight="bold")
        ax2.set_xticklabels(["Down", "Up"], fontsize=8)
        ax2.set_yticklabels(["Down", "Up"], fontsize=8)

        # 3. Metrics Table
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")
        metrics_text = self._create_metrics_table(metrics)
        ax3.text(0.1, 0.9, metrics_text, fontsize=9, verticalalignment="top", family="monospace")

        # 4. Prediction vs Actual
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(df["actual_price"], df["predicted_price"], alpha=0.5, s=20, color="#2962FF")
        min_val = min(df["actual_price"].min(), df["predicted_price"].min())
        max_val = max(df["actual_price"].max(), df["predicted_price"].max())
        ax4.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        ax4.set_title("Predicted vs Actual", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Actual", fontsize=9)
        ax4.set_ylabel("Predicted", fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 5. Residuals
        ax5 = fig.add_subplot(gs[1, 1])
        residuals = df["actual_price"] - df["predicted_price"]
        ax5.scatter(df["predicted_price"], residuals, alpha=0.5, s=20, color="#2962FF")
        ax5.axhline(y=0, color="r", linestyle="--", linewidth=1)
        ax5.set_title("Residual Plot", fontsize=10, fontweight="bold")
        ax5.set_xlabel("Predicted", fontsize=9)
        ax5.set_ylabel("Residuals", fontsize=9)
        ax5.grid(True, alpha=0.3)

        # 6. Residual Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(residuals, bins=30, color="#2962FF", alpha=0.7, edgecolor="white")
        ax6.axvline(x=0, color="r", linestyle="--", linewidth=1)
        ax6.set_title("Residual Distribution", fontsize=10, fontweight="bold")
        ax6.set_xlabel("Residuals", fontsize=9)
        ax6.set_ylabel("Frequency", fontsize=9)
        ax6.grid(True, alpha=0.3, axis="y")

        # 7-9. Time series plots (bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        # Plot predictions over time
        ax7.plot(df["timestamp"], df["current_price"], "gray", alpha=0.5, label="Current Price")
        ax7.scatter(
            df["timestamp"],
            df["predicted_price"],
            s=10,
            color="blue",
            alpha=0.5,
            label="Predicted",
        )
        ax7.scatter(df["timestamp"], df["actual_price"], s=10, color="green", alpha=0.7, label="Actual")
        ax7.set_title("Predictions Over Time", fontsize=10, fontweight="bold")
        ax7.set_xlabel("Time", fontsize=9)
        ax7.set_ylabel("Price", fontsize=9)
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

        # Main title
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "Unknown"
        timeframe = df["timeframe"].iloc[0] if "timeframe" in df.columns else "Unknown"
        fig.suptitle(
            f"Backtesting Results Dashboard - {symbol} ({timeframe})",
            fontsize=16,
            fontweight="bold",
        )

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _create_metrics_table(self, metrics: Dict) -> str:
        """Create formatted metrics table as text"""
        lines = []
        lines.append("METRICS SUMMARY")
        lines.append("=" * 30)

        if "classification" in metrics:
            cls = metrics["classification"]
            lines.append("\nClassification:")
            lines.append(f"  F1 Score:   {cls.get('f1_score', 0):.4f}")
            lines.append(f"  AUC-ROC:    {cls.get('auc_roc', 0):.4f}")
            lines.append(f"  Precision:  {cls.get('precision', 0):.4f}")
            lines.append(f"  Recall:     {cls.get('recall', 0):.4f}")

        if "regression" in metrics:
            reg = metrics["regression"]
            lines.append("\nRegression:")
            lines.append(f"  MAE:        {reg.get('mae', 0):.4f}")
            lines.append(f"  RMSE:       {reg.get('rmse', 0):.4f}")
            lines.append(f"  MAPE:       {reg.get('mape', 0):.2f}%")
            lines.append(f"  R²:         {reg.get('r2_score', 0):.4f}")

        return "\n".join(lines)

    def _save_figure(self, fig: plt.Figure, save_path: str):
        """Save figure to file"""
        try:
            # Create directory if needed
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save figure: {e}")

    def close_all(self):
        """Close all matplotlib figures"""
        plt.close("all")
