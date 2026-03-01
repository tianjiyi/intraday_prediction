"""
HMM-based Market Regime Detector

Uses a 4-state Gaussian Hidden Markov Model to classify market regimes:
1. Range (low volatility, no clear direction)
2. Bull Trend (positive returns, moderate volatility)
3. Bear Trend (negative returns, moderate volatility)
4. Stress/Reversal (high VPIN, high volatility, potential turning point)

The model auto-labels states based on their statistical properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
import logging
from pathlib import Path

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    GaussianHMM = None

logger = logging.getLogger(__name__)


# State labels (4-state directional model)
REGIME_LABELS = {
    'range': 'Range',
    'bull': 'Bull Trend',
    'bear': 'Bear Trend',
    'stress': 'Stress/Reversal'
}

# State labels (3-state Sell Put model - VPIN toxicity based)
SELL_PUT_LABELS = {
    'safe': 'Safe',        # Low VPIN, neutral flow - Enter position
    'toxic': 'Toxic',      # High VPIN, negative OFI - AVOID!
    'reversal': 'Reversal' # Moderate VPIN, positive OFI - Golden entry
}


class RegimeDetector:
    """
    HMM-based market regime detector.

    Uses GaussianHMM with 4 hidden states to model different market regimes.
    States are automatically labeled based on feature means.

    Features used:
        - OFI (Order Flow Imbalance) - z-score normalized
        - VPIN (Volume-Synchronized Probability of Informed Trading)
        - Log Return - z-score normalized
        - Volatility - z-score normalized
    """

    FEATURE_NAMES = ['ofi', 'vpin', 'log_return', 'volatility']

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = 'diag',  # Changed from 'full' for numerical stability
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the regime detector.

        Args:
            n_states: Number of hidden states (default 4)
            covariance_type: HMM covariance type ('full', 'diag', 'tied', 'spherical')
                           'diag' is recommended for numerical stability
            n_iter: Maximum EM iterations for training
            random_state: Random seed for reproducibility
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is not installed. "
                "Install with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.state_labels: Dict[int, str] = {}
        self.is_fitted = False

        # Feature scaling parameters (set during fit)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        logger.info(
            f"RegimeDetector initialized: {n_states} states, "
            f"covariance={covariance_type}"
        )

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance.

        Args:
            X: Feature matrix
            fit: If True, compute and store scaling parameters

        Returns:
            Normalized feature matrix
        """
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds < 1e-8] = 1.0

        if self.feature_means is None or self.feature_stds is None:
            return X

        return (X - self.feature_means) / self.feature_stds

    def fit(self, features_df: pd.DataFrame) -> 'RegimeDetector':
        """
        Train the HMM on historical feature data.

        Args:
            features_df: DataFrame with columns [ofi, vpin, log_return, volatility]

        Returns:
            self (for chaining)
        """
        # Validate features
        for col in self.FEATURE_NAMES:
            if col not in features_df.columns:
                raise ValueError(f"Missing required feature: {col}")

        # Extract feature matrix
        X = features_df[self.FEATURE_NAMES].values

        # Remove any rows with NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < self.n_states * 10:
            raise ValueError(
                f"Insufficient data: {len(X_clean)} samples for {self.n_states} states. "
                f"Need at least {self.n_states * 10} samples."
            )

        logger.info(f"Training HMM on {len(X_clean)} samples...")

        # Normalize features for numerical stability
        X_normalized = self._normalize_features(X_clean, fit=True)
        logger.info(f"Features normalized. Means: {self.feature_means}, Stds: {self.feature_stds}")

        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )

        self.model.fit(X_normalized)

        # Auto-label states
        self.state_labels = self._auto_label_states()

        self.is_fitted = True
        logger.info(f"HMM training complete. State labels: {self.state_labels}")

        return self

    def _auto_label_states(self) -> Dict[int, str]:
        """
        Automatically assign semantic labels to HMM states based on means.

        Dispatches to appropriate labeling method based on n_states:
        - 3 states: Sell Put strategy (Safe/Toxic/Reversal based on VPIN)
        - 4 states: Directional strategy (Range/Bull/Bear/Stress)

        Returns:
            Dict mapping state_id (0 to n_states-1) to label string
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if self.n_states == 3:
            return self._auto_label_states_sell_put()
        else:
            return self._auto_label_states_directional()

    def _auto_label_states_sell_put(self) -> Dict[int, str]:
        """
        Label states for Sell Put strategy based on VPIN toxicity.

        For 3-state model:
        - Toxic: Highest VPIN mean + Most negative OFI (AVOID!)
        - Safe: Lowest VPIN mean + Neutral OFI (ENTER)
        - Reversal: Moderate VPIN + Positive OFI (GOLDEN ENTRY)

        Returns:
            Dict mapping state_id to label string
        """
        means = self.model.means_

        # Feature indices
        idx_ofi = 0
        idx_vpin = 1

        vpin_means = means[:, idx_vpin]
        ofi_means = means[:, idx_ofi]

        labels = {}
        assigned = set()

        # Step 1: Toxic = Highest VPIN + Negative OFI
        # Score: high VPIN + low OFI = most toxic
        toxicity_score = vpin_means - ofi_means  # Higher VPIN, lower OFI = higher score
        toxic_state = int(np.argmax(toxicity_score))
        labels[toxic_state] = SELL_PUT_LABELS['toxic']
        assigned.add(toxic_state)

        # Step 2: Safe = Lowest VPIN + Near-zero OFI
        remaining = [i for i in range(self.n_states) if i not in assigned]
        # Score: low VPIN + neutral OFI = most safe
        safe_scores = []
        for i in remaining:
            score = -vpin_means[i] - abs(ofi_means[i])  # Lower VPIN, OFI near 0 = higher score
            safe_scores.append((i, score))
        safe_scores.sort(key=lambda x: x[1], reverse=True)
        safe_state = safe_scores[0][0]
        labels[safe_state] = SELL_PUT_LABELS['safe']
        assigned.add(safe_state)

        # Step 3: Reversal = Remaining (moderate VPIN + positive OFI)
        remaining = [i for i in range(self.n_states) if i not in assigned]
        for state_id in remaining:
            labels[state_id] = SELL_PUT_LABELS['reversal']

        return labels

    def _auto_label_states_directional(self) -> Dict[int, str]:
        """
        Label states for directional trading strategy (original 4-state model).

        Labeling Rules:
            1. Highest VPIN mean → 'Stress/Reversal'
            2. Lowest |log_return| AND lowest volatility → 'Range'
            3. Remaining states:
               - Positive log_return mean → 'Bull Trend'
               - Negative log_return mean → 'Bear Trend'

        Returns:
            Dict mapping state_id (0 to n_states-1) to label string
        """
        means = self.model.means_

        # Feature indices
        idx_ofi = 0
        idx_vpin = 1
        idx_return = 2
        idx_vol = 3

        labels = {}
        assigned = set()

        # Step 1: Highest VPIN → Stress/Reversal
        vpin_means = means[:, idx_vpin]
        stress_state = int(np.argmax(vpin_means))
        labels[stress_state] = REGIME_LABELS['stress']
        assigned.add(stress_state)

        # Step 2: Lowest |return| AND lowest vol → Range
        remaining = [i for i in range(self.n_states) if i not in assigned]
        if remaining:
            # Score = |return| + volatility (lower is more range-like)
            range_scores = []
            for i in remaining:
                score = abs(means[i, idx_return]) + abs(means[i, idx_vol])
                range_scores.append((i, score))

            range_scores.sort(key=lambda x: x[1])
            range_state = range_scores[0][0]
            labels[range_state] = REGIME_LABELS['range']
            assigned.add(range_state)

        # Step 3: Remaining states → Bull or Bear
        remaining = [i for i in range(self.n_states) if i not in assigned]

        if len(remaining) == 2:
            # Special case: 2 remaining states
            # Check if both have same return sign (both near zero)
            r0, r1 = means[remaining[0], idx_return], means[remaining[1], idx_return]
            same_sign = (r0 >= 0 and r1 >= 0) or (r0 < 0 and r1 < 0)
            both_near_zero = abs(r0) < 0.05 and abs(r1) < 0.05

            if same_sign or both_near_zero:
                # Use volatility to differentiate: higher vol = Bull, lower vol = Bear
                # Rationale: Active trends have higher volatility, quiet distribution is bearish
                v0, v1 = means[remaining[0], idx_vol], means[remaining[1], idx_vol]
                if v0 > v1:
                    labels[remaining[0]] = REGIME_LABELS['bull']
                    labels[remaining[1]] = REGIME_LABELS['bear']
                else:
                    labels[remaining[0]] = REGIME_LABELS['bear']
                    labels[remaining[1]] = REGIME_LABELS['bull']
            else:
                # Different return signs - use return sign
                for state_id in remaining:
                    if means[state_id, idx_return] >= 0:
                        labels[state_id] = REGIME_LABELS['bull']
                    else:
                        labels[state_id] = REGIME_LABELS['bear']
        else:
            # Fall back to return sign for other cases
            for state_id in remaining:
                if means[state_id, idx_return] >= 0:
                    labels[state_id] = REGIME_LABELS['bull']
                else:
                    labels[state_id] = REGIME_LABELS['bear']

        return labels

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime states for a sequence of observations.

        Uses the Viterbi algorithm to find the most likely state sequence.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of state IDs (0 to n_states-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features_df[self.FEATURE_NAMES].values

        # Handle NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_clean = X.copy()
        X_clean[~valid_mask] = 0  # Replace invalid with zeros

        # Normalize using stored parameters
        X_normalized = self._normalize_features(X_clean)

        states = self.model.predict(X_normalized)

        return states

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get state probabilities for each observation.

        Uses the Forward algorithm to compute posterior probabilities.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of shape (n_samples, n_states) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features_df[self.FEATURE_NAMES].values

        # Handle NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_clean = X.copy()
        X_clean[~valid_mask] = 0

        # Normalize using stored parameters
        X_normalized = self._normalize_features(X_clean)

        probs = self.model.predict_proba(X_normalized)

        return probs

    def predict_live(
        self,
        current_features: np.ndarray,
        history: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Predict current regime for real-time inference.

        Uses the Forward algorithm to compute current state probabilities
        given the observation history.

        Args:
            current_features: Shape (4,) array [ofi, vpin, log_return, volatility]
            history: Optional previous observations for context

        Returns:
            Dict with:
                - state: str (semantic label)
                - state_id: int (0 to n_states-1)
                - state_probs: dict mapping label to probability
                - confidence: float (max probability)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Ensure 2D array
        if current_features.ndim == 1:
            current_features = current_features.reshape(1, -1)

        # If history provided, concatenate
        if history is not None:
            if history.ndim == 1:
                history = history.reshape(-1, len(self.FEATURE_NAMES))
            X = np.vstack([history, current_features])
        else:
            X = current_features

        # Normalize using stored parameters
        X_normalized = self._normalize_features(X)

        # Get probabilities for last observation
        probs = self.model.predict_proba(X_normalized)
        current_probs = probs[-1]

        # Get most likely state
        state_id = int(np.argmax(current_probs))
        state_label = self.state_labels.get(state_id, f"State_{state_id}")
        confidence = float(current_probs[state_id])

        # Create probability dict with labels
        state_probs = {}
        for sid, prob in enumerate(current_probs):
            label = self.state_labels.get(sid, f"State_{sid}")
            state_probs[label] = float(prob)

        return {
            'state': state_label,
            'state_id': state_id,
            'state_probs': state_probs,
            'confidence': confidence
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model parameters and statistics.

        Returns:
            Dict with model information
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'n_states': self.n_states
            }

        return {
            'is_fitted': True,
            'n_states': self.n_states,
            'covariance_type': self.covariance_type,
            'state_labels': self.state_labels,
            'feature_names': self.FEATURE_NAMES,
            'means': {
                self.state_labels.get(i, f"State_{i}"): {
                    feat: float(self.model.means_[i, j])
                    for j, feat in enumerate(self.FEATURE_NAMES)
                }
                for i in range(self.n_states)
            },
            'transition_matrix': self.model.transmat_.tolist()
        }

    def save(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_data = {
            'model': self.model,
            'state_labels': self.state_labels,
            'n_states': self.n_states,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RegimeDetector':
        """
        Load trained model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            RegimeDetector instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        detector = cls(
            n_states=save_data['n_states'],
            covariance_type=save_data['covariance_type'],
            n_iter=save_data['n_iter'],
            random_state=save_data['random_state']
        )

        detector.model = save_data['model']
        detector.state_labels = save_data['state_labels']
        detector.is_fitted = save_data['is_fitted']
        detector.feature_means = save_data.get('feature_means')
        detector.feature_stds = save_data.get('feature_stds')

        logger.info(f"Model loaded from {filepath}")

        return detector


# =============================================================================
# Guided HMM Regime Detector for Sell Put Strategy
# =============================================================================

class GuidedRegimeDetector:
    """
    Guided HMM Regime Detector for Sell Put Strategy.

    Key Improvements over RegimeDetector:
    1. Guided Initialization: Forces State 2 to be 'Toxic' (High VPIN).
    2. Input Smoothing: Applies EMA to stop state flickering (56% -> ~15%).
    3. VPIN Rank: Uses CDF (0-1) to handle relative toxicity thresholds.

    Features used (dropped log_return):
        - OFI (Order Flow Imbalance) - z-score normalized
        - VPIN (Volume-Synchronized Probability of Informed Trading) - percentile rank
        - Volatility - z-score normalized
    """

    # Dropped log_return - OFI + Vol captures same info with less noise
    FEATURE_NAMES = ['ofi', 'vpin', 'volatility']

    # Fixed state labels - no auto-labeling needed
    STATE_LABELS = {
        0: 'Range (Safe)',
        1: 'Bull (Trend)',
        2: 'Stress (Toxic)'
    }

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        ema_span: int = 5
    ):
        """
        Initialize the guided regime detector.

        Args:
            n_states: Number of hidden states (default 3)
            n_iter: Maximum EM iterations for training
            random_state: Random seed for reproducibility
            ema_span: EMA smoothing span (default 5 for ~1 week memory)
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is not installed. "
                "Install with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.ema_span = ema_span

        self.model: Optional[GaussianHMM] = None
        self.state_labels: Dict[int, str] = self.STATE_LABELS.copy()
        self.is_fitted = False

        # Scaling parameters
        self.scaler_params: Dict[str, np.ndarray] = {}

        logger.info(
            f"GuidedRegimeDetector initialized: {n_states} states, "
            f"EMA span={ema_span}"
        )

    def _preprocess(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for HMM.

        Steps:
        1. Apply EMA smoothing to reduce transition noise
        2. Convert VPIN to percentile rank (0.0-1.0)
        3. Z-score normalize OFI and Volatility

        Args:
            df: DataFrame with ofi, vpin, volatility columns
            fit: If True, compute and store scaling parameters

        Returns:
            Preprocessed feature matrix
        """
        # Work on copy
        data = df[self.FEATURE_NAMES].copy()

        # Step 1: EMA Smoothing (fixes 56% transition rate)
        data = data.ewm(span=self.ema_span, min_periods=1).mean()

        # Step 2: VPIN to Percentile Rank (fixes threshold issue)
        # During fit, we calculate global rank
        # During predict, we use rolling rank from the data
        if fit:
            data['vpin'] = data['vpin'].rank(pct=True)

        # Step 3: Z-score normalization
        X = data.values

        if fit:
            self.scaler_params['mean'] = np.nanmean(X, axis=0)
            self.scaler_params['std'] = np.nanstd(X, axis=0)

            # Keep VPIN as raw rank (0-1), don't z-score it
            self.scaler_params['mean'][1] = 0.0
            self.scaler_params['std'][1] = 1.0

            # Prevent division by zero
            self.scaler_params['std'][self.scaler_params['std'] < 1e-8] = 1.0

        # Apply scaling
        X_scaled = (X - self.scaler_params['mean']) / self.scaler_params['std']

        # Ensure VPIN column keeps raw rank values during fit
        if fit:
            X_scaled[:, 1] = data['vpin'].values

        return np.nan_to_num(X_scaled)

    def fit(self, features_df: pd.DataFrame) -> 'GuidedRegimeDetector':
        """
        Train the Guided HMM on historical feature data.

        Uses guided priors to force specific state meanings:
        - State 0: Range (Safe) - neutral OFI, low VPIN, low vol
        - State 1: Bull (Trend) - positive OFI, low VPIN, avg vol
        - State 2: Stress (Toxic) - negative OFI, HIGH VPIN, high vol

        Args:
            features_df: DataFrame with columns [ofi, vpin, volatility]

        Returns:
            self (for chaining)
        """
        # Validate features
        for col in self.FEATURE_NAMES:
            if col not in features_df.columns:
                raise ValueError(f"Missing required feature: {col}")

        # Preprocess
        X = self._preprocess(features_df, fit=True)

        # Remove rows with NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < self.n_states * 10:
            raise ValueError(
                f"Insufficient data: {len(X_clean)} samples for {self.n_states} states"
            )

        logger.info(f"Training Guided HMM on {len(X_clean)} samples...")

        # Define Guided Priors [OFI (z-score), VPIN (rank 0-1), Vol (z-score)]
        priors = np.array([
            [0.0, 0.3, -1.0],    # State 0: Range (Safe) - neutral, low VPIN, low vol
            [1.0, 0.3, 0.0],     # State 1: Bull (Trend) - positive OFI, low VPIN
            [-1.0, 0.9, 1.5]     # State 2: Stress (Toxic) - negative OFI, HIGH VPIN
        ])

        # Initialize HMM with guided priors
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",  # Full captures (High Vol + Neg OFI) correlation
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stc"  # Init StartProb, TransMat, Covars - NOT means
        )

        # Set guided means BEFORE fitting
        self.model.means_ = priors

        # Train - EM will fine-tune but stay close to guided priors
        self.model.fit(X_clean)

        self.is_fitted = True

        logger.info(f"Guided HMM training complete.")
        logger.info(f"Final means:\n{self.model.means_}")
        logger.info(f"State labels: {self.state_labels}")

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime states for a sequence of observations.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of state IDs (0, 1, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # For prediction, we need to handle VPIN ranking differently
        # Use rolling rank within the prediction window
        df_copy = features_df.copy()
        df_copy['vpin'] = df_copy['vpin'].rank(pct=True)

        X = self._preprocess_for_predict(df_copy)
        states = self.model.predict(X)

        return states

    def _preprocess_for_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess for prediction (VPIN already ranked)."""
        data = df[self.FEATURE_NAMES].copy()

        # EMA smoothing
        data = data.ewm(span=self.ema_span, min_periods=1).mean()

        X = data.values

        # Apply stored scaling (but keep VPIN as rank)
        X_scaled = X.copy()
        X_scaled[:, 0] = (X[:, 0] - self.scaler_params['mean'][0]) / self.scaler_params['std'][0]
        X_scaled[:, 1] = X[:, 1]  # Keep VPIN rank as-is
        X_scaled[:, 2] = (X[:, 2] - self.scaler_params['mean'][2]) / self.scaler_params['std'][2]

        return np.nan_to_num(X_scaled)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get state probabilities for each observation.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of shape (n_samples, n_states) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        df_copy = features_df.copy()
        df_copy['vpin'] = df_copy['vpin'].rank(pct=True)

        X = self._preprocess_for_predict(df_copy)
        probs = self.model.predict_proba(X)

        return probs

    def predict_live(
        self,
        current_features: np.ndarray,
        history: Optional[np.ndarray] = None,
        history_vpin_rank: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predict current regime for real-time inference.

        Args:
            current_features: Shape (3,) array [ofi, vpin, volatility]
            history: Optional previous observations for EMA smoothing
            history_vpin_rank: Optional pre-computed VPIN rank for current bar

        Returns:
            Dict with state, state_id, is_toxic, confidence
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Build sequence for EMA smoothing
        if history is not None:
            full_seq = np.vstack([history, current_features.reshape(1, -1)])
        else:
            full_seq = current_features.reshape(1, -1)

        df_seq = pd.DataFrame(full_seq, columns=self.FEATURE_NAMES)

        # Use provided VPIN rank or compute rolling rank
        if history_vpin_rank is not None:
            # Use provided rank for current bar
            df_seq['vpin'] = df_seq['vpin'].rank(pct=True)
            df_seq.iloc[-1, 1] = history_vpin_rank
        else:
            df_seq['vpin'] = df_seq['vpin'].rank(pct=True)

        X = self._preprocess_for_predict(df_seq)
        X_curr = X[-1:, :]

        state_id = int(self.model.predict(X_curr)[0])
        probs = self.model.predict_proba(X_curr)[0]
        confidence = float(probs[state_id])

        return {
            'state': self.state_labels[state_id],
            'state_id': state_id,
            'is_toxic': state_id == 2,  # State 2 is ALWAYS Toxic
            'confidence': confidence,
            'state_probs': {
                self.state_labels[i]: float(probs[i])
                for i in range(self.n_states)
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model parameters and statistics."""
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'n_states': self.n_states,
                'detector_type': 'GuidedRegimeDetector'
            }

        return {
            'is_fitted': True,
            'n_states': self.n_states,
            'detector_type': 'GuidedRegimeDetector',
            'ema_span': self.ema_span,
            'state_labels': self.state_labels,
            'feature_names': self.FEATURE_NAMES,
            'means': {
                self.state_labels[i]: {
                    feat: float(self.model.means_[i, j])
                    for j, feat in enumerate(self.FEATURE_NAMES)
                }
                for i in range(self.n_states)
            },
            'transition_matrix': self.model.transmat_.tolist()
        }

    def save(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_data = {
            'detector_type': 'GuidedRegimeDetector',
            'model': self.model,
            'state_labels': self.state_labels,
            'n_states': self.n_states,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'ema_span': self.ema_span,
            'is_fitted': self.is_fitted,
            'scaler_params': self.scaler_params
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"GuidedRegimeDetector saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GuidedRegimeDetector':
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Verify it's a GuidedRegimeDetector
        if save_data.get('detector_type') != 'GuidedRegimeDetector':
            raise ValueError(
                f"File contains {save_data.get('detector_type', 'unknown')}, "
                "not GuidedRegimeDetector"
            )

        detector = cls(
            n_states=save_data['n_states'],
            n_iter=save_data['n_iter'],
            random_state=save_data['random_state'],
            ema_span=save_data.get('ema_span', 5)
        )

        detector.model = save_data['model']
        detector.state_labels = save_data['state_labels']
        detector.is_fitted = save_data['is_fitted']
        detector.scaler_params = save_data.get('scaler_params', {})

        logger.info(f"GuidedRegimeDetector loaded from {filepath}")

        return detector


# =============================================================================
# Income Defender Detector - 4-Feature HMM for Sell Put Strategy
# =============================================================================

class IncomeDefenderDetector:
    """
    4-Feature HMM Regime Detector for Weekly Sell Put Strategy.

    Features:
        - ofi: Order Flow Imbalance (z-score, EMA smoothed)
        - vpin_rank: VPIN percentile rank (0-1)
        - vrp: Volatility Risk Premium proxy (MA20_vol - current_vol)
        - quote_imbalance: Quote-level bid/ask imbalance (LEADING indicator!)

    States:
        0: Bull (鸡肋) - Safe but thin premium, light position
        1: Dip (黄金)  - Golden zone, heavy position
        2: Stress (死亡) - Death zone, CASH

    Key Insights:
        - State 1 (Dip/黄金) is the "golden pit" where VIX is inflated vs realized vol
        - This is where you harvest Vega + Theta with ATM puts
        - quote_imbalance is a LEADING indicator - negative values warn of crashes
          BEFORE they happen (market makers adjust quotes before price moves)
    """

    FEATURE_NAMES = ['ofi', 'vpin_rank', 'vrp', 'quote_imbalance']

    STATE_LABELS = {
        0: 'Bull (鸡肋)',      # Safe but thin premium - light position
        1: 'Dip (黄金)',       # Golden zone - HEAVY position!
        2: 'Stress (死亡)'     # Death zone - CASH
    }

    TRADING_ACTIONS = {
        0: "Light position - far OTM or spreads",
        1: "HEAVY position - ATM puts, harvest Vega+Theta",
        2: "CASH or buy puts for hedge"
    }

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        ema_span: int = 3
    ):
        """
        Initialize the Income Defender detector.

        Args:
            n_states: Number of hidden states (fixed at 3)
            n_iter: Maximum EM iterations for training
            random_state: Random seed for reproducibility
            ema_span: EMA smoothing span for OFI
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is not installed. "
                "Install with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.ema_span = ema_span

        self.model: Optional[GaussianHMM] = None
        self.state_labels: Dict[int, str] = self.STATE_LABELS.copy()
        self.is_fitted = False

        # Scaling parameters
        self.scaler_params: Dict[str, np.ndarray] = {}

        logger.info(
            f"IncomeDefenderDetector initialized: {n_states} states, "
            f"features={self.FEATURE_NAMES}"
        )

    def _preprocess(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for HMM.

        Steps:
        1. Extract 4 features: ofi, vpin_rank, vrp, quote_imbalance
        2. Apply EMA smoothing to OFI
        3. Keep VPIN as rank 0-1, keep quote_imbalance as raw (-1 to 1)
        4. Z-score normalize OFI, VRP only

        Args:
            df: DataFrame with ofi, vpin_rank, vrp, quote_imbalance columns
            fit: If True, compute and store scaling parameters

        Returns:
            Preprocessed feature matrix
        """
        # Check required columns
        required = ['ofi', 'vpin_rank', 'vrp', 'quote_imbalance']
        df_copy = df.copy()

        for col in required:
            if col not in df_copy.columns:
                # Try alternative column names
                if col == 'ofi' and 'ofi_smooth' in df_copy.columns:
                    df_copy = df_copy.rename(columns={'ofi_smooth': 'ofi'})
                else:
                    raise ValueError(f"Missing required column: {col}")

        # Work on copy
        data = df_copy[self.FEATURE_NAMES].copy()

        # EMA Smoothing for OFI
        data['ofi'] = data['ofi'].ewm(span=self.ema_span, min_periods=1).mean()

        # Get feature matrix
        X = data.values

        if fit:
            self.scaler_params['mean'] = np.nanmean(X, axis=0)
            self.scaler_params['std'] = np.nanstd(X, axis=0)

            # Keep vpin_rank as raw 0-1 (don't z-score) - index 1
            self.scaler_params['mean'][1] = 0.0
            self.scaler_params['std'][1] = 1.0

            # Keep quote_imbalance as raw -1 to 1 (don't z-score) - index 3
            self.scaler_params['mean'][3] = 0.0
            self.scaler_params['std'][3] = 1.0

            # Prevent division by zero
            self.scaler_params['std'][self.scaler_params['std'] < 1e-8] = 1.0

        # Apply scaling
        X_scaled = (X - self.scaler_params['mean']) / self.scaler_params['std']

        # Ensure raw values stay raw during fit
        if fit:
            X_scaled[:, 1] = data['vpin_rank'].values  # VPIN rank
            X_scaled[:, 3] = data['quote_imbalance'].values  # Quote imbalance

        return np.nan_to_num(X_scaled)

    def fit(self, features_df: pd.DataFrame) -> 'IncomeDefenderDetector':
        """
        Train the Income Defender HMM on historical feature data.

        Uses guided priors to force specific state meanings:
        - State 0: Bull - positive OFI, low VPIN, LOW VRP, positive quote_imbalance
        - State 1: Dip  - neutral OFI, low VPIN, HIGH VRP, neutral quote_imbalance
        - State 2: Stress - negative OFI, HIGH VPIN, VRP collapse, NEGATIVE quote_imbalance

        Key insight: quote_imbalance is a LEADING indicator!
        - Negative quote_imbalance = market makers posting more asks than bids
        - This happens BEFORE the price crash (7+ minutes early on Nov 20)

        Args:
            features_df: DataFrame with columns [ofi, vpin_rank, vrp, quote_imbalance]

        Returns:
            self (for chaining)
        """
        # Preprocess
        X = self._preprocess(features_df, fit=True)

        # Remove rows with NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < self.n_states * 10:
            raise ValueError(
                f"Insufficient data: {len(X_clean)} samples for {self.n_states} states"
            )

        logger.info(f"Training Income Defender HMM on {len(X_clean)} samples...")

        # Guided Priors [OFI (z), VPIN_Rank (0-1), VRP (z), Quote_Imbalance (-1 to 1)]
        # Key insight: quote_imbalance NEGATIVE = early warning of crash!
        # TUNED: More sensitive to negative quote_imbalance for earlier Stress detection
        # Nov 20 showed QIB = -0.425 at peak, so Stress threshold should be ~ -0.30
        priors = np.array([
            [+1.0, 0.3, -0.5, +0.15],   # State 0: Bull - buying, positive imbalance (raised from +0.10)
            [0.0, 0.3, +1.5, +0.05],    # State 1: Dip  - neutral, slightly positive imbalance
            [-0.5, 0.7, -1.0, -0.35]    # State 2: Stress - TUNED: QIB -0.35 (was -0.20), VPIN 0.7 (was 0.9)
        ])

        # Sticky transition matrix (80% stay in same state)
        # This reduces noisy state transitions
        stick = 0.80
        transmat = np.array([
            [stick, (1-stick)/2, (1-stick)/2],
            [(1-stick)/2, stick, (1-stick)/2],
            [(1-stick)/2, (1-stick)/2, stick]
        ])

        # Initialize HMM with guided priors and sticky transitions
        # Use diag covariance for numerical stability
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",  # More stable than "full"
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="sc",  # Init StartProb, Covars
            params="stmc"      # Update all params - let EM fine-tune
        )

        # Set guided means and sticky transmat BEFORE fitting
        self.model.means_ = priors
        self.model.transmat_ = transmat

        # Train - EM will fine-tune but start from guided priors
        self.model.fit(X_clean)

        # POST-TRAINING: Apply rule-based override for early Stress detection
        # If quote_imbalance is very negative, adjust state assignment
        self._qib_stress_threshold = -0.05  # Trigger Stress when QIB < -0.05 (smoothed scale)

        self.is_fitted = True

        logger.info(f"Income Defender HMM training complete.")
        logger.info(f"Final means:\n{self.model.means_}")
        logger.info(f"State labels: {self.state_labels}")

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime states for a sequence of observations.

        Includes rule-based override: If quote_imbalance < threshold,
        force state to Stress (2) regardless of HMM output.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of state IDs (0, 1, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._preprocess_for_predict(features_df)
        states = self.model.predict(X)

        # Rule-based override: Force Stress when quote_imbalance is very negative
        # This is the LEADING indicator - catches crashes before HMM does
        qib_threshold = getattr(self, '_qib_stress_threshold', -0.05)
        if 'quote_imbalance' in features_df.columns:
            qib_values = features_df['quote_imbalance'].values
            # Force Stress (state 2) when QIB is below threshold
            stress_override = qib_values < qib_threshold
            states[stress_override] = 2  # Stress state

        return states

    def _preprocess_for_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess for prediction."""
        # Check required columns
        required = ['ofi', 'vpin_rank', 'vrp', 'quote_imbalance']
        df_copy = df.copy()

        for col in required:
            if col not in df_copy.columns:
                if col == 'ofi' and 'ofi_smooth' in df_copy.columns:
                    df_copy = df_copy.rename(columns={'ofi_smooth': 'ofi'})
                else:
                    raise ValueError(f"Missing required column: {col}")

        data = df_copy[self.FEATURE_NAMES].copy()

        # EMA smoothing for OFI
        data['ofi'] = data['ofi'].ewm(span=self.ema_span, min_periods=1).mean()

        X = data.values

        # Apply stored scaling (but keep VPIN rank and quote_imbalance as-is)
        X_scaled = X.copy()
        X_scaled[:, 0] = (X[:, 0] - self.scaler_params['mean'][0]) / self.scaler_params['std'][0]  # OFI
        X_scaled[:, 1] = X[:, 1]  # Keep VPIN rank as-is (0-1)
        X_scaled[:, 2] = (X[:, 2] - self.scaler_params['mean'][2]) / self.scaler_params['std'][2]  # VRP
        X_scaled[:, 3] = X[:, 3]  # Keep quote_imbalance as-is (-1 to 1)

        return np.nan_to_num(X_scaled)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get state probabilities for each observation.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            np.ndarray of shape (n_samples, n_states) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._preprocess_for_predict(features_df)
        probs = self.model.predict_proba(X)

        return probs

    def get_action(self, state_id: int) -> str:
        """Get trading action for a state."""
        return self.TRADING_ACTIONS.get(state_id, "Unknown")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model parameters and statistics."""
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'n_states': self.n_states,
                'detector_type': 'IncomeDefenderDetector'
            }

        return {
            'is_fitted': True,
            'n_states': self.n_states,
            'detector_type': 'IncomeDefenderDetector',
            'ema_span': self.ema_span,
            'state_labels': self.state_labels,
            'feature_names': self.FEATURE_NAMES,
            'means': {
                self.state_labels[i]: {
                    feat: float(self.model.means_[i, j])
                    for j, feat in enumerate(self.FEATURE_NAMES)
                }
                for i in range(self.n_states)
            },
            'transition_matrix': self.model.transmat_.tolist(),
            'trading_actions': self.TRADING_ACTIONS
        }

    def save(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_data = {
            'detector_type': 'IncomeDefenderDetector',
            'model': self.model,
            'state_labels': self.state_labels,
            'n_states': self.n_states,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'ema_span': self.ema_span,
            'is_fitted': self.is_fitted,
            'scaler_params': self.scaler_params
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"IncomeDefenderDetector saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'IncomeDefenderDetector':
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Verify it's an IncomeDefenderDetector
        if save_data.get('detector_type') != 'IncomeDefenderDetector':
            raise ValueError(
                f"File contains {save_data.get('detector_type', 'unknown')}, "
                "not IncomeDefenderDetector"
            )

        detector = cls(
            n_states=save_data['n_states'],
            n_iter=save_data['n_iter'],
            random_state=save_data['random_state'],
            ema_span=save_data.get('ema_span', 3)
        )

        detector.model = save_data['model']
        detector.state_labels = save_data['state_labels']
        detector.is_fitted = save_data['is_fitted']
        detector.scaler_params = save_data.get('scaler_params', {})

        logger.info(f"IncomeDefenderDetector loaded from {filepath}")

        return detector


if __name__ == "__main__":
    # Example usage with synthetic data
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== HMM Regime Detector Test ===\n")

    # Generate synthetic market data with different regimes
    np.random.seed(42)

    def generate_regime_data(regime: str, n: int) -> pd.DataFrame:
        """Generate synthetic data for a specific regime."""
        if regime == 'range':
            ofi = np.random.randn(n) * 0.5
            vpin = np.random.uniform(0.2, 0.4, n)
            log_return = np.random.randn(n) * 0.001
            volatility = np.random.uniform(0.005, 0.01, n)
        elif regime == 'bull':
            ofi = np.random.randn(n) * 0.5 + 1.0  # Positive OFI
            vpin = np.random.uniform(0.3, 0.5, n)
            log_return = np.random.randn(n) * 0.005 + 0.002  # Positive returns
            volatility = np.random.uniform(0.01, 0.02, n)
        elif regime == 'bear':
            ofi = np.random.randn(n) * 0.5 - 1.0  # Negative OFI
            vpin = np.random.uniform(0.3, 0.5, n)
            log_return = np.random.randn(n) * 0.005 - 0.002  # Negative returns
            volatility = np.random.uniform(0.01, 0.02, n)
        else:  # stress
            ofi = np.random.randn(n) * 2.0  # High variance
            vpin = np.random.uniform(0.6, 0.9, n)  # High VPIN
            log_return = np.random.randn(n) * 0.01  # High vol returns
            volatility = np.random.uniform(0.025, 0.04, n)  # High volatility

        return pd.DataFrame({
            'ofi': ofi,
            'vpin': vpin,
            'log_return': log_return,
            'volatility': volatility
        })

    # Generate training data with all regimes
    regimes = ['range', 'bull', 'bear', 'stress']
    dfs = [generate_regime_data(r, 100) for r in regimes]
    train_df = pd.concat(dfs, ignore_index=True)

    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Training data shape: {train_df.shape}")
    print(train_df.describe())

    # Initialize and train detector
    detector = RegimeDetector(n_states=4, random_state=42)
    detector.fit(train_df)

    # Print model info
    print("\n=== Model Info ===")
    info = detector.get_model_info()
    print(f"State labels: {info['state_labels']}")
    print("\nState means:")
    for state, means in info['means'].items():
        print(f"  {state}: {means}")

    # Test predictions
    print("\n=== Predictions ===")

    # Generate test data for each regime
    for regime in regimes:
        test_df = generate_regime_data(regime, 10)
        states = detector.predict(test_df)
        probs = detector.predict_proba(test_df)

        print(f"\nTrue regime: {regime}")
        print(f"Predicted states: {states}")
        print(f"Average confidence: {np.max(probs, axis=1).mean():.3f}")

    # Test live prediction
    print("\n=== Live Prediction ===")
    current = np.array([1.5, 0.35, 0.003, 0.015])  # Bull-like features
    result = detector.predict_live(current)
    print(f"Current features: OFI={current[0]:.2f}, VPIN={current[1]:.2f}, "
          f"Return={current[2]:.4f}, Vol={current[3]:.4f}")
    print(f"Predicted state: {result['state']} (confidence: {result['confidence']:.2%})")
    print(f"State probabilities: {result['state_probs']}")

    # Test save/load
    print("\n=== Save/Load Test ===")
    detector.save("test_regime_model.pkl")
    loaded_detector = RegimeDetector.load("test_regime_model.pkl")
    print(f"Loaded model state labels: {loaded_detector.state_labels}")

    # Cleanup
    import os
    os.remove("test_regime_model.pkl")
    print("Test complete!")
