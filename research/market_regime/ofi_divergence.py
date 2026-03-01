"""
OFI Divergence Detection System

Detects divergence between Price and OFI (Order Flow Imbalance):
- Bearish Divergence: Price makes higher high, OFI makes lower high
- Bullish Divergence: Price makes lower low, OFI makes higher low

OFI acts as "shadow of price" - when they diverge, a reversal is likely.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Divergence:
    """Represents a detected divergence event."""
    bar_idx: int              # Bar index where divergence detected
    divergence_type: str      # 'bearish' or 'bullish'
    price_peak1_idx: int      # First peak/trough index
    price_peak2_idx: int      # Second peak/trough index
    price_peak1_val: float    # First peak/trough value
    price_peak2_val: float    # Second peak/trough value
    ofi_peak1_val: float      # OFI at first peak/trough
    ofi_peak2_val: float      # OFI at second peak/trough
    strength: float           # Divergence strength (0-1)


class OFIDivergenceDetector:
    """
    Detects price-OFI divergences for reversal signals.

    Divergence = Price and OFI moving in opposite directions
    - Bullish: Price lower low + OFI higher low = buyers absorbing
    - Bearish: Price higher high + OFI lower high = sellers distributing
    """

    def __init__(
        self,
        peak_window: int = 50,
        ofi_cum_window: int = 100,
        min_peak_distance: int = 100,
        divergence_threshold: float = 0.15,
        prominence_pct: float = 0.002,
        min_price_change_pct: float = 0.2
    ):
        """
        Initialize divergence detector.

        Parameters:
        -----------
        peak_window : int
            Window for local peak/trough detection
        ofi_cum_window : int
            Rolling sum window for cumulative OFI
        min_peak_distance : int
            Minimum bars between peaks to compare for divergence
        divergence_threshold : float
            Minimum relative change in OFI to count as divergence
        prominence_pct : float
            Minimum prominence as percentage of price range
        min_price_change_pct : float
            Minimum price change between peaks to count as significant (%)
        """
        self.peak_window = peak_window
        self.ofi_cum_window = ofi_cum_window
        self.min_peak_distance = min_peak_distance
        self.divergence_threshold = divergence_threshold
        self.prominence_pct = prominence_pct
        self.min_price_change_pct = min_price_change_pct

    def compute_cumulative_ofi(self, ofi_series: pd.Series) -> pd.Series:
        """
        Compute cumulative OFI using rolling sum.

        Cumulative OFI is better than raw OFI because:
        - Less noisy (single-bar OFI is too volatile)
        - Shows sustained buying/selling pressure
        - Better for detecting trend strength
        """
        return ofi_series.rolling(window=self.ofi_cum_window).sum()

    def detect_peaks(
        self,
        series: np.ndarray,
        prominence: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect local peaks (maxima) in a series.

        Returns:
        --------
        peaks : ndarray
            Indices of peaks
        properties : dict
            Peak properties including prominence
        """
        if prominence is None:
            price_range = np.nanmax(series) - np.nanmin(series)
            prominence = price_range * self.prominence_pct

        peaks, properties = find_peaks(
            series,
            distance=self.peak_window,
            prominence=prominence
        )
        return peaks, properties

    def detect_troughs(
        self,
        series: np.ndarray,
        prominence: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect local troughs (minima) in a series.

        Uses peak detection on inverted series.
        """
        inverted = -series
        if prominence is None:
            price_range = np.nanmax(series) - np.nanmin(series)
            prominence = price_range * self.prominence_pct

        troughs, properties = find_peaks(
            inverted,
            distance=self.peak_window,
            prominence=prominence
        )
        return troughs, properties

    def detect_bearish_divergence(
        self,
        price: np.ndarray,
        ofi_cum: np.ndarray,
        lookback: int = 500
    ) -> List[Divergence]:
        """
        Detect bearish divergence (topping signal).

        Bearish Divergence:
        - Price makes HIGHER high
        - OFI makes LOWER high (or declining)

        This means buyers are retreating while price continues up on inertia.
        """
        divergences = []

        # Find price peaks
        peaks, _ = self.detect_peaks(price)

        if len(peaks) < 2:
            return divergences

        # Compare consecutive peaks within lookback window
        for i in range(1, len(peaks)):
            peak2_idx = peaks[i]
            peak1_idx = peaks[i-1]

            # Check if within lookback window
            if peak2_idx - peak1_idx > lookback:
                continue

            # Check minimum distance
            if peak2_idx - peak1_idx < self.min_peak_distance:
                continue

            # Get price values at peaks
            price1 = price[peak1_idx]
            price2 = price[peak2_idx]

            # Get OFI values at peaks
            ofi1 = ofi_cum[peak1_idx]
            ofi2 = ofi_cum[peak2_idx]

            # Skip if NaN
            if np.isnan(ofi1) or np.isnan(ofi2):
                continue

            # Bearish divergence: Price higher high + OFI lower high
            if price2 > price1:
                # Check minimum price change
                price_change_pct = (price2 - price1) / price1 * 100
                if price_change_pct < self.min_price_change_pct:
                    continue

                # Calculate OFI change relative to its range
                ofi_range = np.nanmax(ofi_cum) - np.nanmin(ofi_cum)
                if ofi_range == 0:
                    continue

                ofi_decline = (ofi1 - ofi2) / abs(ofi_range)

                if ofi_decline > self.divergence_threshold:
                    # Strength based on both price gain and OFI decline
                    strength = min(1.0, ofi_decline * 3)  # Scale to 0-1

                    divergences.append(Divergence(
                        bar_idx=peak2_idx,
                        divergence_type='bearish',
                        price_peak1_idx=peak1_idx,
                        price_peak2_idx=peak2_idx,
                        price_peak1_val=price1,
                        price_peak2_val=price2,
                        ofi_peak1_val=ofi1,
                        ofi_peak2_val=ofi2,
                        strength=strength
                    ))

        return divergences

    def detect_bullish_divergence(
        self,
        price: np.ndarray,
        ofi_cum: np.ndarray,
        lookback: int = 500
    ) -> List[Divergence]:
        """
        Detect bullish divergence (bottoming signal).

        Bullish Divergence:
        - Price makes LOWER low
        - OFI makes HIGHER low (less negative)

        This means sellers are exhausting while price continues down.
        """
        divergences = []

        # Find price troughs
        troughs, _ = self.detect_troughs(price)

        if len(troughs) < 2:
            return divergences

        # Compare consecutive troughs within lookback window
        for i in range(1, len(troughs)):
            trough2_idx = troughs[i]
            trough1_idx = troughs[i-1]

            # Check if within lookback window
            if trough2_idx - trough1_idx > lookback:
                continue

            # Check minimum distance
            if trough2_idx - trough1_idx < self.min_peak_distance:
                continue

            # Get price values at troughs
            price1 = price[trough1_idx]
            price2 = price[trough2_idx]

            # Get OFI values at troughs
            ofi1 = ofi_cum[trough1_idx]
            ofi2 = ofi_cum[trough2_idx]

            # Skip if NaN
            if np.isnan(ofi1) or np.isnan(ofi2):
                continue

            # Bullish divergence: Price lower low + OFI higher low
            if price2 < price1:
                # Check minimum price change
                price_change_pct = (price1 - price2) / price1 * 100
                if price_change_pct < self.min_price_change_pct:
                    continue

                # Calculate OFI improvement relative to its range
                ofi_range = np.nanmax(ofi_cum) - np.nanmin(ofi_cum)
                if ofi_range == 0:
                    continue

                ofi_improvement = (ofi2 - ofi1) / abs(ofi_range)

                if ofi_improvement > self.divergence_threshold:
                    # Strength based on both price drop and OFI improvement
                    strength = min(1.0, ofi_improvement * 3)

                    divergences.append(Divergence(
                        bar_idx=trough2_idx,
                        divergence_type='bullish',
                        price_peak1_idx=trough1_idx,
                        price_peak2_idx=trough2_idx,
                        price_peak1_val=price1,
                        price_peak2_val=price2,
                        ofi_peak1_val=ofi1,
                        ofi_peak2_val=ofi2,
                        strength=strength
                    ))

        return divergences

    def detect_all_divergences(
        self,
        price: np.ndarray,
        ofi: np.ndarray,
        lookback: int = 500
    ) -> Tuple[List[Divergence], np.ndarray]:
        """
        Detect all divergences (bearish and bullish).

        Parameters:
        -----------
        price : ndarray
            Price series (close prices)
        ofi : ndarray
            Raw OFI series (will be converted to cumulative)
        lookback : int
            Maximum bars between peaks to compare

        Returns:
        --------
        divergences : List[Divergence]
            All detected divergences
        ofi_cum : ndarray
            Cumulative OFI series used for detection
        """
        # Compute cumulative OFI
        ofi_series = pd.Series(ofi)
        ofi_cum = self.compute_cumulative_ofi(ofi_series).values

        # Detect both types
        bearish = self.detect_bearish_divergence(price, ofi_cum, lookback)
        bullish = self.detect_bullish_divergence(price, ofi_cum, lookback)

        # Combine and sort by bar index
        all_divergences = bearish + bullish
        all_divergences.sort(key=lambda d: d.bar_idx)

        return all_divergences, ofi_cum

    def compute_divergence_score(
        self,
        price: np.ndarray,
        ofi: np.ndarray,
        window: int = 100
    ) -> np.ndarray:
        """
        Compute continuous divergence score for each bar.

        Score range: -1 (strong bullish divergence) to +1 (strong bearish divergence)
        0 = convergence (price and OFI moving together)

        This is useful for visualization and continuous monitoring.
        """
        n = len(price)
        score = np.zeros(n)

        # Compute cumulative OFI
        ofi_series = pd.Series(ofi)
        ofi_cum = self.compute_cumulative_ofi(ofi_series).values

        for i in range(window, n):
            # Get local window
            price_window = price[i-window:i+1]
            ofi_window = ofi_cum[i-window:i+1]

            # Skip if NaN in OFI
            if np.isnan(ofi_window).any():
                continue

            # Compute price direction (normalized)
            price_change = (price_window[-1] - price_window[0]) / price_window[0]

            # Compute OFI direction (normalized)
            ofi_range = np.nanmax(ofi_window) - np.nanmin(ofi_window)
            if ofi_range > 0:
                ofi_change = (ofi_window[-1] - ofi_window[0]) / ofi_range
            else:
                ofi_change = 0

            # Divergence = opposite directions
            # Positive score = bearish divergence (price up, OFI down)
            # Negative score = bullish divergence (price down, OFI up)
            if price_change > 0 and ofi_change < 0:
                # Bearish divergence
                score[i] = min(1.0, abs(ofi_change) * 2)
            elif price_change < 0 and ofi_change > 0:
                # Bullish divergence
                score[i] = -min(1.0, abs(ofi_change) * 2)
            else:
                # Convergence or neutral
                score[i] = 0

        return score
