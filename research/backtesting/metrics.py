"""
Evaluation Metrics for Kronos Model Backtesting

Provides comprehensive metrics for evaluating time series predictions:
- Classification metrics: F1, AUC-ROC, Precision, Recall, Accuracy
- Regression metrics: MAE, RMSE, MAPE, R²
- Probabilistic metrics: Brier Score, Log Loss, Calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    brier_score_loss,
    log_loss,
    average_precision_score,
    precision_recall_curve,
)
import logging

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """Calculate all evaluation metrics for backtesting results"""

    def __init__(self):
        self.results = {}

    @staticmethod
    def calculate_volatility_threshold(
        recent_returns: np.ndarray,
        base_threshold: float = 0.002,
        multiplier: float = 1.5
    ) -> float:
        """
        Calculate adaptive threshold based on recent volatility.

        Args:
            recent_returns: Recent price returns (e.g., last 20 bars)
            base_threshold: Minimum threshold (default 0.2%)
            multiplier: Volatility multiplier (default 1.5x)

        Returns:
            Adaptive threshold value
        """
        if len(recent_returns) == 0:
            return base_threshold

        recent_vol = np.std(recent_returns)
        adaptive_threshold = max(base_threshold, multiplier * recent_vol)
        return adaptive_threshold

    @staticmethod
    def generate_smart_labels(
        predicted_prices: np.ndarray,
        current_price: float,
        recent_returns: np.ndarray,
        threshold_pct: float = 0.002
    ) -> Tuple[int, float, float]:
        """
        Generate labels with noise filtering.

        Args:
            predicted_prices: Array of predicted future prices (Monte Carlo samples or trajectory)
            current_price: Current close price
            recent_returns: Recent price returns for volatility calculation
            threshold_pct: Base threshold percentage

        Returns:
            Tuple of (label, signal_strength, threshold)
            - label: 1 (strong up), -1 (strong down), 0 (neutral)
            - signal_strength: Continuous signal value
            - threshold: Applied threshold value
        """
        # Mean predicted price (reduces noise from Monte Carlo or trajectory)
        mean_pred_price = np.mean(predicted_prices)

        # Calculate return
        predicted_return = (mean_pred_price - current_price) / current_price

        # Adaptive threshold
        threshold = BacktestMetrics.calculate_volatility_threshold(
            recent_returns, threshold_pct
        )

        # Generate label
        if predicted_return > threshold:
            label = 1  # Strong UP (tradeable)
            signal_strength = predicted_return
        elif predicted_return < -threshold:
            label = -1  # Strong DOWN (tradeable)
            signal_strength = predicted_return
        else:
            label = 0  # NEUTRAL (no trade - filtered as noise)
            signal_strength = 0.0

        return label, signal_strength, threshold

    def calculate_all_metrics(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        threshold: float = 0.0,
    ) -> Dict:
        """
        Calculate all metrics for backtesting results.

        Args:
            predictions: DataFrame with columns ['timestamp', 'predicted_price', 'prob_up', 'current_price']
            actuals: DataFrame with columns ['timestamp', 'actual_price']
            threshold: Price change threshold for classification (default: 0.0)

        Returns:
            Dictionary with all calculated metrics
        """
        # Merge predictions with actuals
        df = pd.merge(predictions, actuals, on="timestamp", how="inner")

        if len(df) == 0:
            logger.warning("No matching predictions and actuals found")
            return {}

        # Clean data: remove rows with NaN values
        original_count = len(df)
        df = df.dropna(subset=['predicted_price', 'actual_price', 'prob_up', 'current_price'])
        cleaned_count = len(df)
        dropped_count = original_count - cleaned_count

        if dropped_count > 0:
            logger.warning(
                f"Dropped {dropped_count} predictions with NaN values "
                f"({dropped_count/original_count*100:.1f}%)"
            )

        if len(df) == 0:
            logger.error("No valid predictions after removing NaN values")
            return {}

        # Calculate metrics
        results = {
            "classification": self._calculate_classification_metrics(df, threshold),
            "regression": self._calculate_regression_metrics(df),
            "probabilistic": self._calculate_probabilistic_metrics(df),
            "summary": self._calculate_summary_stats(df),
        }

        self.results = results
        return results

    def _calculate_classification_metrics(
        self, df: pd.DataFrame, threshold: float = 0.0
    ) -> Dict:
        """
        Calculate classification metrics (direction prediction).

        Uses volatility-adaptive thresholds to filter noise and focus on tradeable moves.
        """
        # Calculate adaptive thresholds for each prediction
        df = df.copy()

        # Calculate returns for volatility estimation (using 20-bar lookback)
        lookback_bars = 20
        adaptive_thresholds = []
        predicted_labels = []
        actual_labels = []

        for idx in range(len(df)):
            # Get recent returns for volatility calculation
            start_idx = max(0, idx - lookback_bars)
            recent_prices = df.iloc[start_idx:idx + 1]["current_price"].values

            if len(recent_prices) > 1:
                recent_returns = np.diff(recent_prices) / recent_prices[:-1]
            else:
                recent_returns = np.array([])

            # Calculate adaptive threshold
            adaptive_threshold = self.calculate_volatility_threshold(
                recent_returns,
                base_threshold=0.0008,  # 0.08% minimum (realistic for 30-min QQQ moves)
                multiplier=1.2          # 1.2× volatility
            )
            adaptive_thresholds.append(adaptive_threshold)

            # Calculate predicted return
            current_price = df.iloc[idx]["current_price"]
            predicted_price = df.iloc[idx]["predicted_price"]
            actual_price = df.iloc[idx]["actual_price"]

            predicted_return = (predicted_price - current_price) / current_price
            actual_return = (actual_price - current_price) / current_price

            # Generate predicted labels based on adaptive threshold
            if predicted_return > adaptive_threshold:
                predicted_labels.append(1)  # Strong UP
            elif predicted_return < -adaptive_threshold:
                predicted_labels.append(-1)  # Strong DOWN
            else:
                predicted_labels.append(0)  # NEUTRAL

            # Generate actual labels based on simple direction (no threshold)
            # We want to know: when we predict strong move, does price move in that direction?
            if actual_return > 0:
                actual_labels.append(1)  # UP (any positive move)
            elif actual_return < 0:
                actual_labels.append(-1)  # DOWN (any negative move)
            else:
                actual_labels.append(0)  # No change

        # Add to dataframe
        df["adaptive_threshold"] = adaptive_thresholds
        df["predicted_label"] = predicted_labels
        df["actual_label"] = actual_labels

        # Log filtering statistics
        n_total = len(df)
        n_neutral = (df["predicted_label"] == 0).sum()
        n_up = (df["predicted_label"] == 1).sum()
        n_down = (df["predicted_label"] == -1).sum()
        avg_threshold = np.mean(adaptive_thresholds)

        logger.info(f"Smart labeling: {n_total} predictions -> {n_up} UP, {n_down} DOWN, {n_neutral} NEUTRAL")
        logger.info(f"Average adaptive threshold: {avg_threshold:.4f} ({avg_threshold*100:.2f}%)")

        # Convert to binary: UP (1) vs NOT-UP (0)
        # This evaluates the model as a long-only signal generator
        df["predicted_direction"] = (df["predicted_label"] == 1).astype(int)
        df["actual_direction"] = (df["actual_label"] == 1).astype(int)

        # Extract arrays
        y_true = df["actual_direction"].values
        y_pred = df["predicted_direction"].values
        y_prob = df["prob_up"].values

        # Log class distribution
        n_actual_up = y_true.sum()
        n_actual_not_up = len(y_true) - n_actual_up
        logger.info(f"Actual: {n_actual_up} UP, {n_actual_not_up} NOT-UP")

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        # AUC-ROC (requires probabilities)
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            # Store ROC curve data for plotting
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            metrics["auc_roc"] = None
            metrics["roc_curve"] = None

        # PR-AUC (Precision-Recall AUC - better for imbalanced classes)
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            # Store PR curve data for plotting
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
            metrics["pr_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
        except Exception as e:
            logger.warning(f"Could not calculate PR-AUC: {e}")
            metrics["pr_auc"] = None
            metrics["pr_curve"] = None

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        }

        # Additional metrics
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        else:
            metrics["specificity"] = 0.0
            metrics["npv"] = 0.0

        # Directional accuracy (ignoring magnitude)
        metrics["n_predictions"] = len(y_true)
        metrics["n_up_actual"] = int(np.sum(y_true))
        metrics["n_down_actual"] = int(len(y_true) - np.sum(y_true))
        metrics["n_up_predicted"] = int(np.sum(y_pred))
        metrics["n_down_predicted"] = int(len(y_pred) - np.sum(y_pred))

        # Smart labeling statistics
        metrics["n_total_predictions"] = n_total
        metrics["n_predicted_up"] = int(n_up)
        metrics["n_predicted_down"] = int(n_down)
        metrics["n_predicted_neutral"] = int(n_neutral)
        metrics["n_actual_up"] = int(n_actual_up)
        metrics["n_actual_not_up"] = int(n_actual_not_up)
        metrics["avg_adaptive_threshold"] = float(avg_threshold)

        return metrics

    def _calculate_regression_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate regression metrics (price level prediction).
        """
        y_true = df["actual_price"].values
        y_pred = df["predicted_price"].values

        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R² Score
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        residuals = y_true - y_pred

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2_score": float(r2),
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "max_error": float(np.max(np.abs(residuals))),
            "median_absolute_error": float(np.median(np.abs(residuals))),
        }

        # Residual statistics
        metrics["residuals"] = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "q25": float(np.percentile(residuals, 25)),
            "q50": float(np.percentile(residuals, 50)),
            "q75": float(np.percentile(residuals, 75)),
        }

        return metrics

    def _calculate_probabilistic_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate probabilistic metrics (uncertainty calibration).
        """
        # Binary classification for probabilistic metrics
        y_true = (df["actual_price"] > df["current_price"]).astype(int).values
        y_prob = df["prob_up"].values

        # Brier Score (lower is better, range 0-1)
        try:
            brier = brier_score_loss(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate Brier score: {e}")
            brier = None

        # Log Loss (lower is better)
        try:
            # Clip probabilities to avoid log(0)
            y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
            logloss = log_loss(y_true, y_prob_clipped)
        except Exception as e:
            logger.warning(f"Could not calculate Log Loss: {e}")
            logloss = None

        # Calibration analysis (Expected vs Observed)
        calibration = self._calculate_calibration(y_true, y_prob)

        metrics = {
            "brier_score": float(brier) if brier is not None else None,
            "log_loss": float(logloss) if logloss is not None else None,
            "calibration": calibration,
        }

        return metrics

    def _calculate_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict:
        """
        Calculate calibration curve data.

        Divides predictions into bins by probability and compares
        expected probability vs observed frequency.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        observed_freq = []
        expected_prob = []
        counts = []

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            n_samples = np.sum(mask)

            if n_samples > 0:
                observed = np.mean(y_true[mask])
                expected = np.mean(y_prob[mask])
                observed_freq.append(float(observed))
                expected_prob.append(float(expected))
                counts.append(int(n_samples))
            else:
                observed_freq.append(None)
                expected_prob.append(None)
                counts.append(0)

        # Calculate calibration error (Expected Calibration Error)
        valid_indices = [i for i, c in enumerate(counts) if c > 0]
        if valid_indices:
            ece = np.sum(
                [
                    (counts[i] / len(y_true))
                    * np.abs(observed_freq[i] - expected_prob[i])
                    for i in valid_indices
                ]
            )
        else:
            ece = None

        return {
            "bin_centers": bin_centers.tolist(),
            "observed_frequency": observed_freq,
            "expected_probability": expected_prob,
            "counts": counts,
            "expected_calibration_error": float(ece) if ece is not None else None,
        }

    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics.
        """
        # Price changes
        actual_changes = df["actual_price"] - df["current_price"]
        predicted_changes = df["predicted_price"] - df["current_price"]

        summary = {
            "n_predictions": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            },
            "price_statistics": {
                "current_price_mean": float(df["current_price"].mean()),
                "current_price_std": float(df["current_price"].std()),
                "actual_price_mean": float(df["actual_price"].mean()),
                "predicted_price_mean": float(df["predicted_price"].mean()),
            },
            "change_statistics": {
                "actual_change_mean": float(actual_changes.mean()),
                "actual_change_std": float(actual_changes.std()),
                "predicted_change_mean": float(predicted_changes.mean()),
                "predicted_change_std": float(predicted_changes.std()),
            },
        }

        return summary

    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report.
        """
        if not self.results:
            return "No metrics calculated yet."

        lines = []
        lines.append("=" * 80)
        lines.append("BACKTESTING METRICS SUMMARY")
        lines.append("=" * 80)

        # Classification Metrics
        if "classification" in self.results:
            cls = self.results["classification"]
            lines.append("\n📊 CLASSIFICATION METRICS (Direction Prediction)")
            lines.append("-" * 80)

            # Smart labeling statistics
            if cls.get("n_total_predictions"):
                lines.append(f"  Smart Labeling (Adaptive Thresholds):")
                lines.append(f"    Total predictions:    {cls['n_total_predictions']}")
                lines.append(f"    Predicted UP:         {cls.get('n_predicted_up', 0)}")
                lines.append(f"    Predicted DOWN:       {cls.get('n_predicted_down', 0)}")
                lines.append(f"    Predicted NEUTRAL:    {cls.get('n_predicted_neutral', 0)}")
                lines.append(f"    Actual UP:            {cls.get('n_actual_up', 0)}")
                lines.append(f"    Actual NOT-UP:        {cls.get('n_actual_not_up', 0)}")
                lines.append(f"    Avg threshold:        {cls.get('avg_adaptive_threshold', 0)*100:.2f}%")
                lines.append("")
                lines.append(f"  Evaluation: UP vs NOT-UP (Long-Only Signals)")
                lines.append("")

            if cls.get("pr_auc"):
                lines.append(f"  PR-AUC:         {cls['pr_auc']:.4f}  ⭐ PRIMARY METRIC")
            if cls.get("auc_roc"):
                lines.append(f"  ROC-AUC:        {cls['auc_roc']:.4f}")
            lines.append(f"  Accuracy:       {cls.get('accuracy', 0):.4f}")
            lines.append(f"  Precision:      {cls.get('precision', 0):.4f}")
            lines.append(f"  Recall:         {cls.get('recall', 0):.4f}")
            lines.append(f"  F1 Score:       {cls.get('f1_score', 0):.4f}")
            lines.append(f"  Specificity:    {cls.get('specificity', 0):.4f}")

            cm = cls.get("confusion_matrix", {})
            lines.append(f"\n  Confusion Matrix:")
            lines.append(f"    True Positives:  {cm.get('tp', 0)}")
            lines.append(f"    True Negatives:  {cm.get('tn', 0)}")
            lines.append(f"    False Positives: {cm.get('fp', 0)}")
            lines.append(f"    False Negatives: {cm.get('fn', 0)}")

        # Regression Metrics
        if "regression" in self.results:
            reg = self.results["regression"]
            lines.append("\n📈 REGRESSION METRICS (Price Level Prediction)")
            lines.append("-" * 80)
            lines.append(f"  MAE:            {reg.get('mae', 0):.4f}")
            lines.append(f"  RMSE:           {reg.get('rmse', 0):.4f}")
            lines.append(f"  MAPE:           {reg.get('mape', 0):.2f}%")
            lines.append(f"  R² Score:       {reg.get('r2_score', 0):.4f}")
            lines.append(f"  Max Error:      {reg.get('max_error', 0):.4f}")
            lines.append(
                f"  Median Abs Err: {reg.get('median_absolute_error', 0):.4f}"
            )

        # Probabilistic Metrics
        if "probabilistic" in self.results:
            prob = self.results["probabilistic"]
            lines.append("\n🎲 PROBABILISTIC METRICS (Uncertainty Calibration)")
            lines.append("-" * 80)
            if prob.get("brier_score"):
                lines.append(f"  Brier Score:    {prob['brier_score']:.4f}")
            if prob.get("log_loss"):
                lines.append(f"  Log Loss:       {prob['log_loss']:.4f}")
            if prob.get("calibration", {}).get("expected_calibration_error"):
                ece = prob["calibration"]["expected_calibration_error"]
                lines.append(f"  Calibration Err:{ece:.4f}")

        # Summary
        if "summary" in self.results:
            summ = self.results["summary"]
            lines.append(f"\n📋 SUMMARY STATISTICS")
            lines.append("-" * 80)
            lines.append(f"  Total Predictions: {summ.get('n_predictions', 0)}")
            lines.append(
                f"  Date Range: {summ.get('date_range', {}).get('start', 'N/A')} to {summ.get('date_range', {}).get('end', 'N/A')}"
            )

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def compare_models(results_list: List[Dict], model_names: List[str]) -> pd.DataFrame:
    """
    Compare metrics across multiple models.

    Args:
        results_list: List of metric dictionaries
        model_names: List of model names

    Returns:
        DataFrame with comparison table
    """
    comparison = []

    for i, (results, name) in enumerate(zip(results_list, model_names)):
        row = {"Model": name}

        # Classification
        if "classification" in results:
            cls = results["classification"]
            row["F1"] = cls.get("f1_score", 0)
            row["AUC-ROC"] = cls.get("auc_roc", 0)
            row["Precision"] = cls.get("precision", 0)
            row["Recall"] = cls.get("recall", 0)

        # Regression
        if "regression" in results:
            reg = results["regression"]
            row["MAE"] = reg.get("mae", 0)
            row["RMSE"] = reg.get("rmse", 0)
            row["R²"] = reg.get("r2_score", 0)

        comparison.append(row)

    return pd.DataFrame(comparison)
