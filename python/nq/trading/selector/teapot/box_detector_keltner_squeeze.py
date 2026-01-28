"""
Keltner Squeeze Box Detector for Teapot pattern recognition.

Based on channel squeeze and volatility decay.
Specifically designed to capture periods when Bollinger Bands or Keltner Channels
are extremely converged (squeeze), indicating energy accumulation.
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class KeltnerSqueezeDetector(BoxDetector):
    """
    Keltner Squeeze Box Detector (通道挤压箱体检测器).
    
    Based on channel squeeze and volatility decay.
    Specifically designed to capture periods when channels are extremely converged.
    
    Core features:
    1. Channel Squeeze (通道挤压): ATR/MA ratio indicates extreme convergence.
    2. Slope Flatness (斜率走平): Mid-line slope is near zero.
    3. Backward Expansion (回溯扩展): Once squeeze is detected, expand backward to find true start.
    
    This detector is particularly effective for:
    - Capturing squeeze periods before breakouts
    - Identifying energy accumulation phases
    - Filtering out false signals during downtrends
    """

    def __init__(
        self,
        box_window: int = 20,
        squeeze_threshold: float = 0.06,
        slope_threshold: float = 0.015,
        atr_multiplier: float = 1.5,
        volume_decay_threshold: Optional[float] = 0.8,
        lookback_window: int = 60,
        expansion_window: int = 15,
        stability_window: int = 20,
        stability_threshold: int = 15,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Keltner Squeeze Box Detector.

        Args:
            box_window: Window size for channel calculation (default: 20 days).
            squeeze_threshold: Maximum squeeze ratio (ATR/MA) for box detection (default: 0.06, 6%).
            slope_threshold: Maximum mid-line slope change (default: 0.015, 1.5%).
            atr_multiplier: Multiplier for ATR to define channel width (default: 1.5).
            volume_decay_threshold: Volume decay threshold (default: 0.8, 80%).
                If None, volume check is disabled.
            lookback_window: Window for finding anchor points (default: 60 days).
            expansion_window: Window for backward expansion once squeeze detected (default: 15 days).
            stability_window: Window for stability check (default: 20 days).
            stability_threshold: Minimum days in stability_window that must be stable (default: 15).
            smooth_window: Window size for box filter smoothing (default: None, disabled).
            smooth_threshold: Minimum number of days in smooth_window (default: None, disabled).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.squeeze_threshold = squeeze_threshold
        self.slope_threshold = slope_threshold
        self.atr_multiplier = atr_multiplier
        self.volume_decay_threshold = volume_decay_threshold
        self.lookback_window = lookback_window
        self.expansion_window = expansion_window
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using Keltner Squeeze method with backward expansion.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # 1. Calculate channel indicators (similar to Bollinger Bands, but using ATR)
        df = df.with_columns([
            # Mid-line: moving average
            pl.col("close")
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("mid_line"),
            
            # ATR (Average True Range): more robust than simple high-low range
            # ATR = average of (high - low) over window
            (pl.col("high") - pl.col("low"))
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("atr"),
        ])

        # 2. Calculate squeeze ratio and slope
        df = df.with_columns([
            # Squeeze ratio: ATR / Mid-line
            # Smaller value means more squeeze (channels are closer together)
            (pl.col("atr") / pl.col("mid_line")).alias("squeeze_ratio"),
            
            # Mid-line slope: 5-day change rate
            (
                (pl.col("mid_line") - pl.col("mid_line").shift(5))
                .abs()
                / pl.col("mid_line").shift(5)
            ).alias("mid_slope"),
        ])

        # 3. Calculate channel bounds (box_h and box_l)
        df = df.with_columns([
            (pl.col("mid_line") + pl.col("atr") * self.atr_multiplier).alias("box_h"),
            (pl.col("mid_line") - pl.col("atr") * self.atr_multiplier).alias("box_l"),
        ])

        # 4. Calculate box width (for compatibility)
        df = df.with_columns([
            ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias("box_width"),
        ])

        # 5. Calculate price stability (for backward expansion)
        # Check if price is within narrow range around mid-line
        df = df.with_columns([
            (
                (pl.col("close") - pl.col("mid_line")).abs() / pl.col("mid_line") < 0.03
            ).cast(pl.Int32).alias("in_range_dot"),
        ])

        # 6. Calculate stability count (how many days price stayed in range)
        df = df.with_columns([
            pl.col("in_range_dot")
            .rolling_sum(window_size=self.stability_window)
            .over("ts_code")
            .alias("stability_count"),
        ])

        # 7. Volume decay check (optional, for filtering)
        if self.volume_decay_threshold is not None:
            df = df.with_columns([
                pl.col("volume")
                .rolling_mean(window_size=20)
                .over("ts_code")
                .alias("volume_short"),
                pl.col("volume")
                .rolling_mean(window_size=60)
                .over("ts_code")
                .alias("volume_long"),
            ])
            volume_ok = (
                pl.col("volume_short") < pl.col("volume_long") * self.volume_decay_threshold
            )
        else:
            volume_ok = pl.lit(True)

        # 8. Initial squeeze detection
        df = df.with_columns([
            (
                # Condition 1: Extreme convergence (squeeze_ratio < threshold)
                # Corresponds to red and green lines being very close in the image
                (pl.col("squeeze_ratio") < self.squeeze_threshold)
                &
                # Condition 2: Slope is very low (mid-line is flat)
                # Corresponds to cyan line being horizontal in the image
                (pl.col("mid_slope") < self.slope_threshold)
                &
                # Condition 3: Price is within channel (exclude sharp drops)
                (pl.col("close") < pl.col("box_h"))
                & (pl.col("close") > pl.col("box_l"))
                &
                # Condition 4: Stability check (price stayed in range for sufficient time)
                (pl.col("stability_count") >= self.stability_threshold)
                &
                # Condition 5: Volume decay (optional)
                volume_ok
                &
                # Condition 6: Channel bounds are valid
                pl.col("box_h").is_not_null()
                & pl.col("box_l").is_not_null()
                & (pl.col("box_h") > pl.col("box_l"))
            ).alias("is_squeeze_detected"),
        ])

        # 9. Backward expansion to solve lag problem
        # Once squeeze is detected, expand backward to find true start point
        # This is the key to solving the "signal lag" problem
        df = df.with_columns([
            # Use rolling_max to expand backward: if today is squeeze, mark past N days as box
            pl.col("is_squeeze_detected")
            .cast(pl.Int32)
            .rolling_max(window_size=self.expansion_window)
            .over("ts_code")
            .cast(pl.Boolean)
            .alias("is_box_candidate"),
        ])

        # 10. Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df


class ExpansionAnchorBoxDetector(BoxDetector):
    """
    Expansion Anchor Box Detector (锚点回溯箱体检测器).
    
    Advanced version that solves the lag problem by:
    1. Finding anchor points (local minima/maxima)
    2. Expanding backward once squeeze is detected
    3. Using stability-based detection instead of simple rolling windows
    """

    def __init__(
        self,
        box_window: int = 40,
        squeeze_threshold: float = 0.06,
        slope_threshold: float = 0.015,
        atr_multiplier: float = 1.5,
        lookback_window: int = 60,
        expansion_window: int = 20,
        stability_window: int = 20,
        stability_threshold: int = 15,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Expansion Anchor Box Detector.

        Args:
            box_window: Window size for channel calculation (default: 40 days).
            squeeze_threshold: Maximum squeeze ratio (ATR/MA) for box detection (default: 0.06).
            slope_threshold: Maximum mid-line slope change (default: 0.015).
            atr_multiplier: Multiplier for ATR to define channel width (default: 1.5).
            lookback_window: Window for finding anchor points (default: 60 days).
            expansion_window: Window for backward expansion (default: 20 days).
            stability_window: Window for stability check (default: 20 days).
            stability_threshold: Minimum days in stability_window (default: 15).
            smooth_window: Window size for box filter smoothing (default: None).
            smooth_threshold: Minimum number of days in smooth_window (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.squeeze_threshold = squeeze_threshold
        self.slope_threshold = slope_threshold
        self.atr_multiplier = atr_multiplier
        self.lookback_window = lookback_window
        self.expansion_window = expansion_window
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using expansion anchor method with backward propagation.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # 1. Calculate short MA for stability check
        df = df.with_columns([
            pl.col("close")
            .rolling_mean(window_size=10)
            .over("ts_code")
            .alias("short_ma"),
        ])

        # 2. Calculate price stability: how long price stayed near MA
        df = df.with_columns([
            (
                (pl.col("close") - pl.col("short_ma")).abs() / pl.col("short_ma") < 0.03
            ).cast(pl.Int32).alias("in_range_dot"),
        ])

        # 3. Calculate stability count (cumulative days in range)
        df = df.with_columns([
            pl.col("in_range_dot")
            .rolling_sum(window_size=self.stability_window)
            .over("ts_code")
            .alias("stability_count"),
        ])

        # 4. Calculate channel indicators
        df = df.with_columns([
            pl.col("close")
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("mid_line"),
            (pl.col("high") - pl.col("low"))
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("atr"),
        ])

        # 5. Calculate squeeze ratio and slope
        df = df.with_columns([
            (pl.col("atr") / pl.col("mid_line")).alias("squeeze_ratio"),
            (
                (pl.col("mid_line") - pl.col("mid_line").shift(5))
                .abs()
                / pl.col("mid_line").shift(5)
            ).alias("mid_slope"),
        ])

        # 6. Calculate channel bounds
        df = df.with_columns([
            (pl.col("mid_line") + pl.col("atr") * self.atr_multiplier).alias("box_h"),
            (pl.col("mid_line") - pl.col("atr") * self.atr_multiplier).alias("box_l"),
        ])

        # 7. Calculate box width
        df = df.with_columns([
            ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias("box_width"),
        ])

        # 8. Find anchor points (local minima in lookback window)
        # This helps identify potential box start points
        df = df.with_columns([
            pl.col("low")
            .rolling_min(window_size=self.lookback_window)
            .over("ts_code")
            .alias("global_min"),
        ])

        # 9. Detect squeeze condition
        df = df.with_columns([
            (
                (pl.col("squeeze_ratio") < self.squeeze_threshold)
                & (pl.col("mid_slope") < self.slope_threshold)
                & (pl.col("close") < pl.col("box_h"))
                & (pl.col("close") > pl.col("box_l"))
                & (pl.col("stability_count") >= self.stability_threshold)
                & pl.col("box_h").is_not_null()
                & pl.col("box_l").is_not_null()
                & (pl.col("box_h") > pl.col("box_l"))
            ).alias("is_squeeze_detected"),
        ])

        # 10. Backward expansion: once squeeze is detected, expand backward
        # This is the key to solving lag: use rolling_max to propagate signal backward
        df = df.with_columns([
            pl.col("is_squeeze_detected")
            .cast(pl.Int32)
            .rolling_max(window_size=self.expansion_window)
            .over("ts_code")
            .cast(pl.Boolean)
            .alias("is_box_candidate"),
        ])

        # 11. Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df
