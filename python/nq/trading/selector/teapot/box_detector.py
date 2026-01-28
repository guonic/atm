"""
Box detection algorithms for Teapot pattern recognition.

Provides multiple implementations of box detection algorithms for evaluation.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl

logger = logging.getLogger(__name__)


class BoxDetector(ABC):
    """
    Abstract base class for box detection algorithms.

    All box detectors should implement the detect_box method.
    """

    def __init__(
        self,
        box_window: int = 40,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize box detector.

        Args:
            box_window: Window size for box calculation (default: 40 days).
            smooth_window: Window size for box filter smoothing (default: None, disabled).
            smooth_threshold: Minimum number of days in smooth_window that must be box candidates
                (default: None, disabled). If set, requires smooth_threshold out of smooth_window
                days to be candidates.
        """
        self.box_window = box_window
        self.smooth_window = smooth_window
        self.smooth_threshold = smooth_threshold

    @abstractmethod
    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box pattern in DataFrame.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.

        Returns:
            DataFrame with added columns:
            - box_h: Box upper bound
            - box_l: Box lower bound
            - box_width: Box width (relative value)
            - is_box_candidate: Boolean indicating if current row is a box candidate
        """
        pass

    def _apply_smoothing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply box filter smoothing to reduce false positives.

        Only marks as box candidate if past M days have N days as candidates.

        Args:
            df: DataFrame with is_box_candidate column.

        Returns:
            DataFrame with smoothed is_box_candidate column.
        """
        if self.smooth_window is None or self.smooth_threshold is None:
            return df

        return df.with_columns(
            [
                (
                    pl.col("is_box_candidate")
                    .cast(pl.Int32)
                    .rolling_sum(window_size=self.smooth_window)
                    .over("ts_code")
                    >= self.smooth_threshold
                ).alias("is_box_candidate")
            ]
        )


class SimpleBoxDetector(BoxDetector):
    """
    Simple box detector using rolling max/min.

    This is the original implementation that uses rolling max/min to define box bounds.
    """

    def __init__(
        self,
        box_window: int = 40,
        box_width_threshold: float = 0.15,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize simple box detector.

        Args:
            box_window: Window size for box calculation (default: 40 days).
            box_width_threshold: Maximum box width threshold (default: 0.15, i.e., 15%).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.box_width_threshold = box_width_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using simple rolling max/min method.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Group by stock code
        df = df.with_columns(
            [
                # Box upper bound (rolling max of high)
                pl.col("high")
                .rolling_max(window_size=self.box_window)
                .shift(1)
                .over("ts_code")
                .alias("box_h"),
                # Box lower bound (rolling min of low)
                pl.col("low")
                .rolling_min(window_size=self.box_window)
                .shift(1)
                .over("ts_code")
                .alias("box_l"),
            ]
        )

        # Compute box width
        df = df.with_columns(
            [
                # Box width (relative)
                ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias(
                    "box_width"
                ),
            ]
        )

        # Mark box candidates
        df = df.with_columns(
            [
                (
                    (pl.col("box_width") < self.box_width_threshold)
                    & (pl.col("box_width") > 0)
                    & pl.col("box_h").is_not_null()
                    & pl.col("box_l").is_not_null()
                ).alias("is_box_candidate")
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class MeanReversionBoxDetectorV2(BoxDetector):
    """
    Improved mean reversion box detector (V2) for better coarse recall.

    Improvements:
    1. Uses rolling_quantile(0.95/0.05) instead of rolling_max/min to filter extreme outliers
    2. Calculates slope relative to volatility (unit-time return vs volatility)
    3. More flexible parameters for coarse recall

    Recommended parameters for coarse recall:
    - box_window: 30-40
    - max_total_return: 0.07 (allows 7% center displacement)
    - max_relative_box_height: 0.12 (allows 12% total fluctuation space)
    - smooth_window: 10
    - smooth_threshold: 6
    """

    def __init__(
        self,
        box_window: int = 40,
        max_total_return: Optional[float] = None,
        max_relative_box_height: float = 0.12,
        volatility_ratio: float = 0.25,
        quantile_high: float = 0.95,
        quantile_low: float = 0.05,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize improved mean reversion box detector (V2).

        Args:
            box_window: Window size for box calculation (default: 40 days).
            max_total_return: Maximum total return over window. If None, uses volatility-based
                calculation (slope < volatility * volatility_ratio).
            max_relative_box_height: Maximum relative box height (default: 0.12, i.e., 12%).
            volatility_ratio: Ratio for volatility-based slope threshold (default: 0.25).
                If slope < volatility * volatility_ratio, considered no trend.
            quantile_high: High quantile for box upper bound (default: 0.95).
            quantile_low: Low quantile for box lower bound (default: 0.05).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height
        self.volatility_ratio = volatility_ratio
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using improved mean reversion method (V2).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Group by stock code
        df = df.with_columns(
            [
                # 1. Calculate volatility (std of returns)
                pl.col("close")
                .pct_change()
                .rolling_std(window_size=self.box_window)
                .over("ts_code")
                .alias("volatility"),
                # 2. Calculate unit-time return (slope per day)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                    / self.box_window
                )
                .over("ts_code")
                .alias("unit_time_return"),
                # 3. Calculate total return over window (for reference)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                )
                .over("ts_code")
                .alias("total_return"),
                # 4. Calculate average deviation from mean (mean reversion strength)
                (
                    (pl.col("close") - pl.col("close").rolling_mean(self.box_window))
                    .abs()
                    .rolling_mean(self.box_window)
                    .over("ts_code")
                ).alias("avg_deviation"),
            ]
        )

        # Calculate box bounds using quantiles (filters extreme outliers)
        df = df.with_columns(
            [
                # Box upper bound (95th percentile, filters 5% extreme highs)
                pl.col("high")
                .rolling_quantile(
                    quantile=self.quantile_high, window_size=self.box_window
                )
                .shift(1)
                .over("ts_code")
                .alias("box_h"),
                # Box lower bound (5th percentile, filters 5% extreme lows)
                pl.col("low")
                .rolling_quantile(
                    quantile=self.quantile_low, window_size=self.box_window
                )
                .shift(1)
                .over("ts_code")
                .alias("box_l"),
            ]
        )

        # Calculate relative box height using quantile-based bounds
        df = df.with_columns(
            [
                (
                    (pl.col("box_h") - pl.col("box_l"))
                    / pl.col("close").rolling_mean(self.box_window)
                )
                .over("ts_code")
                .alias("relative_box_height"),
            ]
        )

        # Compute box width (for compatibility)
        df = df.with_columns(
            [
                ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias(
                    "box_width"
                ),
            ]
        )

        # Determine slope threshold: use volatility-based if max_total_return is None
        if self.max_total_return is None:
            # Slope should be less than volatility * volatility_ratio
            # Convert to total return: unit_time_return * box_window < volatility * volatility_ratio
            slope_ok = (
                pl.col("unit_time_return").abs() * self.box_window
                < pl.col("volatility") * self.volatility_ratio
            )
        else:
            # Use fixed threshold
            slope_ok = pl.col("total_return").abs() < self.max_total_return

        # Define box candidate using vectorized operations
        df = df.with_columns(
            [
                (
                    slope_ok
                    & (pl.col("relative_box_height") < self.max_relative_box_height)
                    & pl.col("total_return").is_not_null()
                    & pl.col("relative_box_height").is_not_null()
                    & pl.col("avg_deviation").is_not_null()
                    & pl.col("volatility").is_not_null()
                ).alias("is_box_candidate")
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class HybridBoxDetectorV2(BoxDetector):
    """
    Improved hybrid box detector (V2) using scoring system for coarse recall.

    Instead of requiring all detectors to agree (AND logic), this version:
    1. Calculates scores from multiple dimensions
    2. Weighted sum of scores from three detectors
    3. Recalls if total score > threshold

    This approach is better for coarse recall as it's more inclusive.
    """

    def __init__(
        self,
        box_window: int = 40,
        box_width_threshold: float = 0.15,
        max_total_return: Optional[float] = None,
        max_relative_box_height: float = 0.12,
        volatility_ratio: float = 0.25,
        score_threshold: float = 0.5,
        weight_simple: float = 0.3,
        weight_mean_reversion: float = 0.4,
        weight_mean_reversion_v2: float = 0.3,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize improved hybrid box detector (V2).

        Args:
            box_window: Window size for box calculation (default: 40 days).
            box_width_threshold: Maximum box width threshold for simple method (default: 0.15).
            max_total_return: Maximum total return for mean reversion method.
                If None, uses volatility-based calculation.
            max_relative_box_height: Maximum relative box height (default: 0.12).
            volatility_ratio: Ratio for volatility-based slope threshold (default: 0.25).
            score_threshold: Minimum total score to be considered a box candidate (default: 0.5).
            weight_simple: Weight for simple detector score (default: 0.3).
            weight_mean_reversion: Weight for mean reversion detector score (default: 0.4).
            weight_mean_reversion_v2: Weight for mean reversion V2 detector score (default: 0.3).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.box_width_threshold = box_width_threshold
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height
        self.volatility_ratio = volatility_ratio
        self.score_threshold = score_threshold
        self.weight_simple = weight_simple
        self.weight_mean_reversion = weight_mean_reversion
        self.weight_mean_reversion_v2 = weight_mean_reversion_v2

        # Normalize weights to sum to 1.0
        total_weight = weight_simple + weight_mean_reversion + weight_mean_reversion_v2
        if total_weight > 0:
            self.weight_simple = weight_simple / total_weight
            self.weight_mean_reversion = weight_mean_reversion / total_weight
            self.weight_mean_reversion_v2 = weight_mean_reversion_v2 / total_weight

        # Initialize detectors
        self.simple_detector = SimpleBoxDetector(
            box_window=box_window,
            box_width_threshold=box_width_threshold,
            smooth_window=None,  # Apply smoothing at the end
            smooth_threshold=None,
        )
        self.mean_reversion_detector = MeanReversionBoxDetector(
            box_window=box_window,
            max_total_return=max_total_return or 0.05,
            max_relative_box_height=max_relative_box_height,
            smooth_window=None,
            smooth_threshold=None,
        )
        self.mean_reversion_v2_detector = MeanReversionBoxDetectorV2(
            box_window=box_window,
            max_total_return=max_total_return,
            max_relative_box_height=max_relative_box_height,
            volatility_ratio=volatility_ratio,
            smooth_window=None,
            smooth_threshold=None,
        )

    def _calculate_score(self, is_candidate: pl.Series) -> pl.Series:
        """
        Calculate score from boolean candidate series.

        Args:
            is_candidate: Boolean series indicating candidates.

        Returns:
            Score series (0.0 or 1.0).
        """
        return is_candidate.cast(pl.Float64)

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using hybrid scoring method (V2).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Apply all three detectors
        df_simple = self.simple_detector.detect_box(df)
        df_mean_rev = self.mean_reversion_detector.detect_box(df)
        df_mean_rev_v2 = self.mean_reversion_v2_detector.detect_box(df)

        # Calculate scores from each detector
        score_simple = self._calculate_score(df_simple["is_box_candidate"])
        score_mean_rev = self._calculate_score(df_mean_rev["is_box_candidate"])
        score_mean_rev_v2 = self._calculate_score(df_mean_rev_v2["is_box_candidate"])

        # Calculate weighted total score
        total_score = (
            score_simple * self.weight_simple
            + score_mean_rev * self.weight_mean_reversion
            + score_mean_rev_v2 * self.weight_mean_reversion_v2
        )

        # Use V2 detector's box bounds (more robust with quantiles)
        df = df.with_columns(
            [
                df_mean_rev_v2["box_h"].alias("box_h"),
                df_mean_rev_v2["box_l"].alias("box_l"),
                df_mean_rev_v2["box_width"].alias("box_width"),
                (total_score >= self.score_threshold).alias("is_box_candidate"),
                total_score.alias("box_score"),  # Add score for analysis
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class MeanReversionBoxDetector(BoxDetector):
    """
    Mean reversion box detector using slope significance and mean reversion strength.

    This improved version uses:
    - Slope significance: Total return over window should be near zero
    - Mean reversion strength: Price deviation from mean should be small
    - Box compactness: Relative box height should be small
    """

    def __init__(
        self,
        box_window: int = 40,
        max_total_return: float = 0.05,
        max_relative_box_height: float = 0.08,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize mean reversion box detector.

        Args:
            box_window: Window size for box calculation (default: 40 days).
            max_total_return: Maximum total return over window (default: 0.05, i.e., 5%).
            max_relative_box_height: Maximum relative box height (default: 0.08, i.e., 8%).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using mean reversion and slope significance.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Group by stock code
        df = df.with_columns(
            [
                # 1. Calculate total return over window (slope significance)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                )
                .over("ts_code")
                .alias("total_return"),
                # 2. Calculate average deviation from mean (mean reversion strength)
                (
                    (pl.col("close") - pl.col("close").rolling_mean(self.box_window))
                    .abs()
                    .rolling_mean(self.box_window)
                    .over("ts_code")
                ).alias("avg_deviation"),
                # 3. Calculate relative box height (compactness)
                (
                    (
                        pl.col("high").rolling_max(self.box_window)
                        - pl.col("low").rolling_min(self.box_window)
                    )
                    / pl.col("close").rolling_mean(self.box_window)
                )
                .over("ts_code")
                .alias("relative_box_height"),
                # Box bounds for reference
                pl.col("high")
                .rolling_max(window_size=self.box_window)
                .shift(1)
                .over("ts_code")
                .alias("box_h"),
                pl.col("low")
                .rolling_min(window_size=self.box_window)
                .shift(1)
                .over("ts_code")
                .alias("box_l"),
            ]
        )

        # Compute box width (for compatibility)
        df = df.with_columns(
            [
                ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias(
                    "box_width"
                ),
            ]
        )

        # Define box candidate using vectorized operations (faster than map_elements)
        df = df.with_columns(
            [
                (
                    (pl.col("total_return").abs() < self.max_total_return)
                    & (pl.col("relative_box_height") < self.max_relative_box_height)
                    & pl.col("total_return").is_not_null()
                    & pl.col("relative_box_height").is_not_null()
                    & pl.col("avg_deviation").is_not_null()
                ).alias("is_box_candidate")
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class MeanReversionBoxDetectorV2(BoxDetector):
    """
    Improved mean reversion box detector (V2) for better coarse recall.

    Improvements:
    1. Uses rolling_quantile(0.95/0.05) instead of rolling_max/min to filter extreme outliers
    2. Calculates slope relative to volatility (unit-time return vs volatility)
    3. More flexible parameters for coarse recall

    Recommended parameters for coarse recall:
    - box_window: 30-40
    - max_total_return: 0.07 (allows 7% center displacement)
    - max_relative_box_height: 0.12 (allows 12% total fluctuation space)
    - smooth_window: 10
    - smooth_threshold: 6
    """

    def __init__(
        self,
        box_window: int = 40,
        max_total_return: Optional[float] = None,
        max_relative_box_height: float = 0.12,
        volatility_ratio: float = 0.25,
        quantile_high: float = 0.95,
        quantile_low: float = 0.05,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize improved mean reversion box detector (V2).

        Args:
            box_window: Window size for box calculation (default: 40 days).
            max_total_return: Maximum total return over window. If None, uses volatility-based
                calculation (slope < volatility * volatility_ratio).
            max_relative_box_height: Maximum relative box height (default: 0.12, i.e., 12%).
            volatility_ratio: Ratio for volatility-based slope threshold (default: 0.25).
                If slope < volatility * volatility_ratio, considered no trend.
            quantile_high: High quantile for box upper bound (default: 0.95).
            quantile_low: Low quantile for box lower bound (default: 0.05).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height
        self.volatility_ratio = volatility_ratio
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using improved mean reversion method (V2).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Group by stock code
        df = df.with_columns(
            [
                # 1. Calculate volatility (std of returns)
                pl.col("close")
                .pct_change()
                .rolling_std(window_size=self.box_window)
                .over("ts_code")
                .alias("volatility"),
                # 2. Calculate unit-time return (slope per day)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                    / self.box_window
                )
                .over("ts_code")
                .alias("unit_time_return"),
                # 3. Calculate total return over window (for reference)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                )
                .over("ts_code")
                .alias("total_return"),
                # 4. Calculate average deviation from mean (mean reversion strength)
                (
                    (pl.col("close") - pl.col("close").rolling_mean(self.box_window))
                    .abs()
                    .rolling_mean(self.box_window)
                    .over("ts_code")
                ).alias("avg_deviation"),
            ]
        )

        # Calculate box bounds using quantiles (filters extreme outliers)
        df = df.with_columns(
            [
                # Box upper bound (95th percentile, filters 5% extreme highs)
                pl.col("high")
                .rolling_quantile(
                    quantile=self.quantile_high, window_size=self.box_window
                )
                .shift(1)
                .over("ts_code")
                .alias("box_h"),
                # Box lower bound (5th percentile, filters 5% extreme lows)
                pl.col("low")
                .rolling_quantile(
                    quantile=self.quantile_low, window_size=self.box_window
                )
                .shift(1)
                .over("ts_code")
                .alias("box_l"),
            ]
        )

        # Calculate relative box height using quantile-based bounds
        df = df.with_columns(
            [
                (
                    (pl.col("box_h") - pl.col("box_l"))
                    / pl.col("close").rolling_mean(self.box_window)
                )
                .over("ts_code")
                .alias("relative_box_height"),
            ]
        )

        # Compute box width (for compatibility)
        df = df.with_columns(
            [
                ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias(
                    "box_width"
                ),
            ]
        )

        # Determine slope threshold: use volatility-based if max_total_return is None
        if self.max_total_return is None:
            # Slope should be less than volatility * volatility_ratio
            # Convert to total return: unit_time_return * box_window < volatility * volatility_ratio
            slope_ok = (
                pl.col("unit_time_return").abs() * self.box_window
                < pl.col("volatility") * self.volatility_ratio
            )
        else:
            # Use fixed threshold
            slope_ok = pl.col("total_return").abs() < self.max_total_return

        # Define box candidate using vectorized operations
        df = df.with_columns(
            [
                (
                    slope_ok
                    & (pl.col("relative_box_height") < self.max_relative_box_height)
                    & pl.col("total_return").is_not_null()
                    & pl.col("relative_box_height").is_not_null()
                    & pl.col("avg_deviation").is_not_null()
                    & pl.col("volatility").is_not_null()
                ).alias("is_box_candidate")
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class HybridBoxDetectorV2(BoxDetector):
    """
    Improved hybrid box detector (V2) using scoring system for coarse recall.

    Instead of requiring all detectors to agree (AND logic), this version:
    1. Calculates scores from multiple dimensions
    2. Weighted sum of scores from three detectors
    3. Recalls if total score > threshold

    This approach is better for coarse recall as it's more inclusive.
    """

    def __init__(
        self,
        box_window: int = 40,
        box_width_threshold: float = 0.15,
        max_total_return: Optional[float] = None,
        max_relative_box_height: float = 0.12,
        volatility_ratio: float = 0.25,
        score_threshold: float = 0.5,
        weight_simple: float = 0.3,
        weight_mean_reversion: float = 0.4,
        weight_mean_reversion_v2: float = 0.3,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize improved hybrid box detector (V2).

        Args:
            box_window: Window size for box calculation (default: 40 days).
            box_width_threshold: Maximum box width threshold for simple method (default: 0.15).
            max_total_return: Maximum total return for mean reversion method.
                If None, uses volatility-based calculation.
            max_relative_box_height: Maximum relative box height (default: 0.12).
            volatility_ratio: Ratio for volatility-based slope threshold (default: 0.25).
            score_threshold: Minimum total score to be considered a box candidate (default: 0.5).
            weight_simple: Weight for simple detector score (default: 0.3).
            weight_mean_reversion: Weight for mean reversion detector score (default: 0.4).
            weight_mean_reversion_v2: Weight for mean reversion V2 detector score (default: 0.3).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.box_width_threshold = box_width_threshold
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height
        self.volatility_ratio = volatility_ratio
        self.score_threshold = score_threshold
        self.weight_simple = weight_simple
        self.weight_mean_reversion = weight_mean_reversion
        self.weight_mean_reversion_v2 = weight_mean_reversion_v2

        # Normalize weights to sum to 1.0
        total_weight = weight_simple + weight_mean_reversion + weight_mean_reversion_v2
        if total_weight > 0:
            self.weight_simple = weight_simple / total_weight
            self.weight_mean_reversion = weight_mean_reversion / total_weight
            self.weight_mean_reversion_v2 = weight_mean_reversion_v2 / total_weight

        # Initialize detectors
        self.simple_detector = SimpleBoxDetector(
            box_window=box_window,
            box_width_threshold=box_width_threshold,
            smooth_window=None,  # Apply smoothing at the end
            smooth_threshold=None,
        )
        self.mean_reversion_detector = MeanReversionBoxDetector(
            box_window=box_window,
            max_total_return=max_total_return or 0.05,
            max_relative_box_height=max_relative_box_height,
            smooth_window=None,
            smooth_threshold=None,
        )
        self.mean_reversion_v2_detector = MeanReversionBoxDetectorV2(
            box_window=box_window,
            max_total_return=max_total_return,
            max_relative_box_height=max_relative_box_height,
            volatility_ratio=volatility_ratio,
            smooth_window=None,
            smooth_threshold=None,
        )

    def _calculate_score(self, is_candidate: pl.Series) -> pl.Series:
        """
        Calculate score from boolean candidate series.

        Args:
            is_candidate: Boolean series indicating candidates.

        Returns:
            Score series (0.0 or 1.0).
        """
        return is_candidate.cast(pl.Float64)

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using hybrid scoring method (V2).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Apply all three detectors
        df_simple = self.simple_detector.detect_box(df)
        df_mean_rev = self.mean_reversion_detector.detect_box(df)
        df_mean_rev_v2 = self.mean_reversion_v2_detector.detect_box(df)

        # Calculate scores from each detector
        score_simple = self._calculate_score(df_simple["is_box_candidate"])
        score_mean_rev = self._calculate_score(df_mean_rev["is_box_candidate"])
        score_mean_rev_v2 = self._calculate_score(df_mean_rev_v2["is_box_candidate"])

        # Calculate weighted total score
        total_score = (
            score_simple * self.weight_simple
            + score_mean_rev * self.weight_mean_reversion
            + score_mean_rev_v2 * self.weight_mean_reversion_v2
        )

        # Use V2 detector's box bounds (more robust with quantiles)
        df = df.with_columns(
            [
                df_mean_rev_v2["box_h"].alias("box_h"),
                df_mean_rev_v2["box_l"].alias("box_l"),
                df_mean_rev_v2["box_width"].alias("box_width"),
                (total_score >= self.score_threshold).alias("is_box_candidate"),
                total_score.alias("box_score"),  # Add score for analysis
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class HybridBoxDetector(BoxDetector):
    """
    Hybrid box detector combining simple and mean reversion methods.

    Uses both methods and requires both to agree for a box candidate.
    """

    def __init__(
        self,
        box_window: int = 40,
        box_width_threshold: float = 0.15,
        max_total_return: float = 0.05,
        max_relative_box_height: float = 0.08,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize hybrid box detector.

        Args:
            box_window: Window size for box calculation (default: 40 days).
            box_width_threshold: Maximum box width threshold for simple method (default: 0.15).
            max_total_return: Maximum total return for mean reversion method (default: 0.05).
            max_relative_box_height: Maximum relative box height for mean reversion method
                (default: 0.08).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.simple_detector = SimpleBoxDetector(
            box_window=box_window,
            box_width_threshold=box_width_threshold,
            smooth_window=None,  # Apply smoothing at the end
            smooth_threshold=None,
        )
        self.mean_reversion_detector = MeanReversionBoxDetector(
            box_window=box_window,
            max_total_return=max_total_return,
            max_relative_box_height=max_relative_box_height,
            smooth_window=None,  # Apply smoothing at the end
            smooth_threshold=None,
        )

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using hybrid method (both simple and mean reversion).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Apply both detectors
        df_simple = self.simple_detector.detect_box(df)
        df_mean_rev = self.mean_reversion_detector.detect_box(df)

        # Merge results - both must agree
        df = df.with_columns(
            [
                df_simple["box_h"].alias("box_h"),
                df_simple["box_l"].alias("box_l"),
                df_simple["box_width"].alias("box_width"),
                (
                    df_simple["is_box_candidate"] & df_mean_rev["is_box_candidate"]
                ).alias("is_box_candidate"),
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class MeanReversionBoxDetectorV2(BoxDetector):
    """
    Improved mean reversion box detector (V2) for better coarse recall.

    Improvements:
    1. Uses rolling_quantile(0.95/0.05) instead of rolling_max/min to filter extreme outliers
    2. Calculates slope relative to volatility (unit-time return vs volatility)
    3. More flexible parameters for coarse recall

    Recommended parameters for coarse recall:
    - box_window: 30-40
    - max_total_return: 0.07 (allows 7% center displacement)
    - max_relative_box_height: 0.12 (allows 12% total fluctuation space)
    - smooth_window: 10
    - smooth_threshold: 6
    """

    def __init__(
        self,
        box_window: int = 40,
        max_total_return: Optional[float] = None,
        max_relative_box_height: float = 0.12,
        volatility_ratio: float = 0.25,
        quantile_high: float = 0.95,
        quantile_low: float = 0.05,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize improved mean reversion box detector (V2).

        Args:
            box_window: Window size for box calculation (default: 40 days).
            max_total_return: Maximum total return over window. If None, uses volatility-based
                calculation (slope < volatility * volatility_ratio).
            max_relative_box_height: Maximum relative box height (default: 0.12, i.e., 12%).
            volatility_ratio: Ratio for volatility-based slope threshold (default: 0.25).
                If slope < volatility * volatility_ratio, considered no trend.
            quantile_high: High quantile for box upper bound (default: 0.95).
            quantile_low: Low quantile for box lower bound (default: 0.05).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height
        self.volatility_ratio = volatility_ratio
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using improved mean reversion method (V2).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Group by stock code
        df = df.with_columns(
            [
                # 1. Calculate volatility (std of returns)
                pl.col("close")
                .pct_change()
                .rolling_std(window_size=self.box_window)
                .over("ts_code")
                .alias("volatility"),
                # 2. Calculate unit-time return (slope per day)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                    / self.box_window
                )
                .over("ts_code")
                .alias("unit_time_return"),
                # 3. Calculate total return over window (for reference)
                (
                    (pl.col("close") - pl.col("close").shift(self.box_window))
                    / pl.col("close").shift(self.box_window)
                )
                .over("ts_code")
                .alias("total_return"),
                # 4. Calculate average deviation from mean (mean reversion strength)
                (
                    (pl.col("close") - pl.col("close").rolling_mean(self.box_window))
                    .abs()
                    .rolling_mean(self.box_window)
                    .over("ts_code")
                ).alias("avg_deviation"),
            ]
        )

        # Calculate box bounds using quantiles (filters extreme outliers)
        df = df.with_columns(
            [
                # Box upper bound (95th percentile, filters 5% extreme highs)
                pl.col("high")
                .rolling_quantile(
                    quantile=self.quantile_high, window_size=self.box_window
                )
                .shift(1)
                .over("ts_code")
                .alias("box_h"),
                # Box lower bound (5th percentile, filters 5% extreme lows)
                pl.col("low")
                .rolling_quantile(
                    quantile=self.quantile_low, window_size=self.box_window
                )
                .shift(1)
                .over("ts_code")
                .alias("box_l"),
            ]
        )

        # Calculate relative box height using quantile-based bounds
        df = df.with_columns(
            [
                (
                    (pl.col("box_h") - pl.col("box_l"))
                    / pl.col("close").rolling_mean(self.box_window)
                )
                .over("ts_code")
                .alias("relative_box_height"),
            ]
        )

        # Compute box width (for compatibility)
        df = df.with_columns(
            [
                ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias(
                    "box_width"
                ),
            ]
        )

        # Determine slope threshold: use volatility-based if max_total_return is None
        if self.max_total_return is None:
            # Slope should be less than volatility * volatility_ratio
            # Convert to total return: unit_time_return * box_window < volatility * volatility_ratio
            slope_ok = (
                pl.col("unit_time_return").abs() * self.box_window
                < pl.col("volatility") * self.volatility_ratio
            )
        else:
            # Use fixed threshold
            slope_ok = pl.col("total_return").abs() < self.max_total_return

        # Define box candidate using vectorized operations
        df = df.with_columns(
            [
                (
                    slope_ok
                    & (pl.col("relative_box_height") < self.max_relative_box_height)
                    & pl.col("total_return").is_not_null()
                    & pl.col("relative_box_height").is_not_null()
                    & pl.col("avg_deviation").is_not_null()
                    & pl.col("volatility").is_not_null()
                ).alias("is_box_candidate")
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class HybridBoxDetectorV2(BoxDetector):
    """
    Improved hybrid box detector (V2) using scoring system for coarse recall.

    Instead of requiring all detectors to agree (AND logic), this version:
    1. Calculates scores from multiple dimensions
    2. Weighted sum of scores from three detectors
    3. Recalls if total score > threshold

    This approach is better for coarse recall as it's more inclusive.
    """

    def __init__(
        self,
        box_window: int = 40,
        box_width_threshold: float = 0.15,
        max_total_return: Optional[float] = None,
        max_relative_box_height: float = 0.12,
        volatility_ratio: float = 0.25,
        score_threshold: float = 0.5,
        weight_simple: float = 0.3,
        weight_mean_reversion: float = 0.4,
        weight_mean_reversion_v2: float = 0.3,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize improved hybrid box detector (V2).

        Args:
            box_window: Window size for box calculation (default: 40 days).
            box_width_threshold: Maximum box width threshold for simple method (default: 0.15).
            max_total_return: Maximum total return for mean reversion method.
                If None, uses volatility-based calculation.
            max_relative_box_height: Maximum relative box height (default: 0.12).
            volatility_ratio: Ratio for volatility-based slope threshold (default: 0.25).
            score_threshold: Minimum total score to be considered a box candidate (default: 0.5).
            weight_simple: Weight for simple detector score (default: 0.3).
            weight_mean_reversion: Weight for mean reversion detector score (default: 0.4).
            weight_mean_reversion_v2: Weight for mean reversion V2 detector score (default: 0.3).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.box_width_threshold = box_width_threshold
        self.max_total_return = max_total_return
        self.max_relative_box_height = max_relative_box_height
        self.volatility_ratio = volatility_ratio
        self.score_threshold = score_threshold
        self.weight_simple = weight_simple
        self.weight_mean_reversion = weight_mean_reversion
        self.weight_mean_reversion_v2 = weight_mean_reversion_v2

        # Normalize weights to sum to 1.0
        total_weight = weight_simple + weight_mean_reversion + weight_mean_reversion_v2
        if total_weight > 0:
            self.weight_simple = weight_simple / total_weight
            self.weight_mean_reversion = weight_mean_reversion / total_weight
            self.weight_mean_reversion_v2 = weight_mean_reversion_v2 / total_weight

        # Initialize detectors
        self.simple_detector = SimpleBoxDetector(
            box_window=box_window,
            box_width_threshold=box_width_threshold,
            smooth_window=None,  # Apply smoothing at the end
            smooth_threshold=None,
        )
        self.mean_reversion_detector = MeanReversionBoxDetector(
            box_window=box_window,
            max_total_return=max_total_return or 0.05,
            max_relative_box_height=max_relative_box_height,
            smooth_window=None,
            smooth_threshold=None,
        )
        self.mean_reversion_v2_detector = MeanReversionBoxDetectorV2(
            box_window=box_window,
            max_total_return=max_total_return,
            max_relative_box_height=max_relative_box_height,
            volatility_ratio=volatility_ratio,
            smooth_window=None,
            smooth_threshold=None,
        )

    def _calculate_score(self, is_candidate: pl.Series) -> pl.Series:
        """
        Calculate score from boolean candidate series.

        Args:
            is_candidate: Boolean series indicating candidates.

        Returns:
            Score series (0.0 or 1.0).
        """
        return is_candidate.cast(pl.Float64)

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using hybrid scoring method (V2).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # Apply all three detectors
        df_simple = self.simple_detector.detect_box(df)
        df_mean_rev = self.mean_reversion_detector.detect_box(df)
        df_mean_rev_v2 = self.mean_reversion_v2_detector.detect_box(df)

        # Calculate scores from each detector
        score_simple = self._calculate_score(df_simple["is_box_candidate"])
        score_mean_rev = self._calculate_score(df_mean_rev["is_box_candidate"])
        score_mean_rev_v2 = self._calculate_score(df_mean_rev_v2["is_box_candidate"])

        # Calculate weighted total score
        total_score = (
            score_simple * self.weight_simple
            + score_mean_rev * self.weight_mean_reversion
            + score_mean_rev_v2 * self.weight_mean_reversion_v2
        )

        # Use V2 detector's box bounds (more robust with quantiles)
        df = df.with_columns(
            [
                df_mean_rev_v2["box_h"].alias("box_h"),
                df_mean_rev_v2["box_l"].alias("box_l"),
                df_mean_rev_v2["box_width"].alias("box_width"),
                (total_score >= self.score_threshold).alias("is_box_candidate"),
                total_score.alias("box_score"),  # Add score for analysis
            ]
        )

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df
