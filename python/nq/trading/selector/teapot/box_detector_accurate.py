"""
Accurate Box Detector for Teapot pattern recognition.

Based on price pivot stability and cross-over frequency.
Specifically designed to capture "clear platform periods" with accurate start points.
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class AccurateBoxDetector(BoxDetector):
    """
    Accurate Box Detector (精准起始点平台检测器).
    
    Based on price pivot stability and cross-over frequency.
    Specifically designed to capture "clear platform periods" with accurate start points.
    
    Core features:
    1. Pivot Line (价格中轴): Uses rolling mean as price pivot.
    2. Cross-over Count (穿越次数): Counts how many times price crosses the pivot.
    3. Symmetric Boundaries (对称边界): Requires upper and lower distances to be balanced.
    4. Back-propagation (回填): Once platform is confirmed, marks past N days as box.
    
    This detector is particularly effective for:
    - Capturing clear platform periods with accurate start points
    - Filtering out false platforms during downtrends
    - Identifying "value area" where price oscillates around a pivot
    """

    def __init__(
        self,
        box_window: int = 30,
        price_tol: float = 0.03,
        min_cross_count: int = 3,
        min_touch_count: Optional[int] = None,
        touch_tolerance: float = 0.02,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Accurate Box Detector.

        Args:
            box_window: Window size for pivot calculation and back-propagation (default: 30 days).
            price_tol: Price tolerance from pivot line (default: 0.03, 3%).
            min_cross_count: Minimum number of cross-overs required (default: 3).
            min_touch_count: Minimum number of touches on each boundary (default: None, auto-calculated).
                If None, uses max(1, min_cross_count // 3). Set to 0 to disable touch count check.
            touch_tolerance: Tolerance for "touching" boundary (default: 0.02, 2%).
            smooth_window: Window size for box filter smoothing (default: None, disabled).
            smooth_threshold: Minimum number of days in smooth_window (default: None, disabled).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.price_tol = price_tol
        self.min_cross_count = min_cross_count
        self.min_touch_count = min_touch_count
        self.touch_tolerance = touch_tolerance

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using pivot stability and cross-over frequency.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # 1. Calculate pivot line (price anchor): rolling mean as price pivot
        df = df.with_columns([
            pl.col("close")
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("pivot_line"),
            pl.col("close")
            .rolling_std(window_size=self.box_window)
            .over("ts_code")
            .alias("pivot_std"),
        ])

        # 2. Key feature: Calculate cross-over count (价格穿越中轴的次数)
        # A clear platform period requires price to oscillate around the pivot
        # We count how many times price crosses the pivot line
        df = df.with_columns([
            (
                # Upward cross: price was below pivot, now above
                (
                    (pl.col("close") > pl.col("pivot_line"))
                    & (pl.col("close").shift(1) <= pl.col("pivot_line").shift(1))
                )
                |
                # Downward cross: price was above pivot, now below
                (
                    (pl.col("close") < pl.col("pivot_line"))
                    & (pl.col("close").shift(1) >= pl.col("pivot_line").shift(1))
                )
            )
            .cast(pl.Int32)
            .alias("cross_event"),
        ])

        # Count cross-overs in the rolling window
        df = df.with_columns([
            pl.col("cross_event")
            .rolling_sum(window_size=self.box_window)
            .over("ts_code")
            .alias("cross_count"),
        ])

        # 3. Calculate platform boundary purity (平台边界纯净度)
        # Only when high/low prices are within price_tol of pivot_line
        df = df.with_columns([
            # Upper distance: (max_high - pivot_line) / pivot_line
            (
                (pl.col("high").rolling_max(window_size=self.box_window) - pl.col("pivot_line"))
                / pl.col("pivot_line")
            ).alias("upper_dist"),
            # Lower distance: (pivot_line - min_low) / pivot_line
            (
                (pl.col("pivot_line") - pl.col("low").rolling_min(window_size=self.box_window))
                / pl.col("pivot_line")
            ).alias("lower_dist"),
        ])
        
        # 3.5. Calculate box boundaries for touch detection
        df = df.with_columns([
            (pl.col("pivot_line") * (1 + self.price_tol)).alias("box_h_temp"),
            (pl.col("pivot_line") * (1 - self.price_tol)).alias("box_l_temp"),
        ])
        
        # Count how many times high touches or approaches upper boundary
        # A "touch" is defined as high being within a tolerance of box_h
        
        # Step 1: Define touch events
        df = df.with_columns([
            (
                (pl.col("high") >= pl.col("box_h_temp") * (1 - self.touch_tolerance))
                & (pl.col("high") <= pl.col("box_h_temp") * (1 + self.touch_tolerance))
            )
            .cast(pl.Int32)
            .alias("high_touch_event"),
            (
                (pl.col("low") >= pl.col("box_l_temp") * (1 - self.touch_tolerance))
                & (pl.col("low") <= pl.col("box_l_temp") * (1 + self.touch_tolerance))
            )
            .cast(pl.Int32)
            .alias("low_touch_event"),
        ])
        
        # Step 2: Count touches in the rolling window
        df = df.with_columns([
            pl.col("high_touch_event")
            .rolling_sum(window_size=self.box_window)
            .over("ts_code")
            .alias("high_touch_count"),
            pl.col("low_touch_event")
            .rolling_sum(window_size=self.box_window)
            .over("ts_code")
            .alias("low_touch_count"),
        ])

        # 4. Calculate box boundaries first (for strict boundary checking)
        df = df.with_columns([
            (pl.col("pivot_line") * (1 + self.price_tol)).alias("box_h"),
            (pl.col("pivot_line") * (1 - self.price_tol)).alias("box_l"),
        ])
        
        # 5. Detection logic (Candidate) - STRICT: No candlesticks can exceed box boundaries
        # Clear platform characteristics:
        # - High cross-over count (price oscillates around pivot)
        # - High touches upper boundary multiple times
        # - Low touches lower boundary multiple times
        # - Short boundary distances (price stays within tolerance)
        # - Symmetric boundaries (upper and lower distances balanced)
        # - CRITICAL: Current K-line's high/low must be within box boundaries
        
        # Minimum touch count required
        # If not specified, use auto-calculated value; if 0, disable touch count check
        if self.min_touch_count is None:
            min_touch_count = max(1, self.min_cross_count // 3)  # At least 1 touch, or one-third of cross_count
        else:
            min_touch_count = self.min_touch_count
        
        df = df.with_columns([
            (
                # Condition 1: At least min_cross_count cross-overs
                # This excludes one-way trends (e.g., continuous downtrend)
                (pl.col("cross_count") >= self.min_cross_count)
                &
                # Condition 2: High must touch upper boundary multiple times (if enabled)
                # This ensures price actually reaches the upper boundary, not just stays below it
                (pl.when(min_touch_count > 0)
                 .then(pl.col("high_touch_count") >= min_touch_count)
                 .otherwise(True))
                &
                # Condition 3: Low must touch lower boundary multiple times (if enabled)
                # This ensures price actually reaches the lower boundary, not just stays above it
                (pl.when(min_touch_count > 0)
                 .then(pl.col("low_touch_count") >= min_touch_count)
                 .otherwise(True))
                &
                # Condition 4: Upper boundary within tolerance (rolling window check)
                (pl.col("upper_dist") < self.price_tol)
                &
                # Condition 5: Lower boundary within tolerance (rolling window check)
                (pl.col("lower_dist") < self.price_tol)
                &
                # Condition 6: Pivot line is valid
                pl.col("pivot_line").is_not_null()
                & (pl.col("pivot_line") > 0)
                &
                # Condition 7: STRICT - Current K-line's high must not exceed box_h
                (pl.col("high") <= pl.col("box_h"))
                &
                # Condition 8: STRICT - Current K-line's low must not be below box_l
                (pl.col("low") >= pl.col("box_l"))
            ).alias("is_confirmed"),
        ])
        
        # Clean up temporary columns
        df = df.drop(["box_h_temp", "box_l_temp", "high_touch_event", "low_touch_event"])
        
        # 6. Core logic to solve "start time lag" problem: Back-propagation (回填)
        # Once today is confirmed as a mature platform, we mark past N days as box
        # This is because platforms have persistence: if it's confirmed today,
        # it likely existed for the past N days
        df = df.with_columns([
            # Use rolling_max to propagate signal backward (to the past)
            pl.col("is_confirmed")
            .cast(pl.Int32)
            .rolling_max(window_size=self.box_window)
            .over("ts_code")
            .cast(pl.Boolean)
            .alias("is_box_candidate_temp"),
        ])
        
        # 7. STRICT CHECK: Verify entire box periods have NO candlesticks exceeding boundaries
        # If ANY candlestick in a box period exceeds the boundaries, the entire box period is invalid
        # We need to use unified box boundaries for each box period (based on confirmation point)
        
        # For each box period, use the pivot_line at the confirmation point to calculate unified boundaries
        # Use rolling_max to propagate the confirmed pivot_line backward (matching back-propagation)
        df = df.with_columns([
            # When is_confirmed=True, capture the pivot_line; otherwise None
            # Then use rolling_max to propagate it backward to past days in the same box period
            pl.when(pl.col("is_confirmed"))
            .then(pl.col("pivot_line"))
            .otherwise(None)
            .rolling_max(window_size=self.box_window)
            .over("ts_code")
            .alias("unified_pivot_line"),
        ])
        
        # Calculate unified box boundaries based on unified_pivot_line
        df = df.with_columns([
            (pl.col("unified_pivot_line") * (1 + self.price_tol)).alias("box_h_unified"),
            (pl.col("unified_pivot_line") * (1 - self.price_tol)).alias("box_l_unified"),
        ])
        
        # For box candidate periods, use unified boundaries; otherwise use individual boundaries
        df = df.with_columns([
            pl.when(pl.col("is_box_candidate_temp") & pl.col("unified_pivot_line").is_not_null())
            .then(pl.col("box_h_unified"))
            .otherwise(pl.col("box_h"))
            .alias("box_h_check"),
            pl.when(pl.col("is_box_candidate_temp") & pl.col("unified_pivot_line").is_not_null())
            .then(pl.col("box_l_unified"))
            .otherwise(pl.col("box_l"))
            .alias("box_l_check"),
        ])
        
        # Mark days that violate unified boundaries
        df = df.with_columns([
            (
                pl.col("is_box_candidate_temp")
                & (
                    (pl.col("high") > pl.col("box_h_check"))
                    | (pl.col("low") < pl.col("box_l_check"))
                )
            ).alias("violates_boundary"),
        ])
        
        # For each stock, find consecutive box periods and check if any day violates boundaries
        df = df.sort(["ts_code", "trade_date"])
        
        # Create groups for consecutive box periods per stock
        df = df.with_columns([
            (
                (pl.col("is_box_candidate_temp") != pl.col("is_box_candidate_temp").shift(1).fill_null(False))
                | (pl.col("ts_code") != pl.col("ts_code").shift(1).fill_null(False))
            ).cast(pl.Int32).cum_sum().alias("period_group"),
        ])
        
        # For each period group, check if it's a box period and if any day violates boundaries
        period_stats = (
            df.filter(pl.col("is_box_candidate_temp"))
            .group_by(["ts_code", "period_group"])
            .agg([
                pl.col("violates_boundary").max().alias("has_violation"),
            ])
        )
        
        # Create a mapping: period_group -> is_valid_box (no violations)
        valid_periods = period_stats.filter(~pl.col("has_violation")).select(["ts_code", "period_group"])
        valid_periods = valid_periods.with_columns([
            pl.lit(True).alias("is_valid_box_period"),
        ])
        
        # Join back to mark valid box periods
        df = df.join(
            valid_periods,
            on=["ts_code", "period_group"],
            how="left",
        )
        
        # Only mark as box candidate if it's a valid box period (no violations)
        df = df.with_columns([
            (
                pl.col("is_box_candidate_temp")
                & pl.col("is_valid_box_period").fill_null(False)
            ).alias("is_box_candidate"),
        ])
        
        # Use unified box boundaries for visualization when in valid box periods
        df = df.with_columns([
            pl.when(pl.col("is_box_candidate") & pl.col("unified_pivot_line").is_not_null())
            .then(pl.col("box_h_unified"))
            .otherwise(pl.col("box_h"))
            .alias("box_h"),
            pl.when(pl.col("is_box_candidate") & pl.col("unified_pivot_line").is_not_null())
            .then(pl.col("box_l_unified"))
            .otherwise(pl.col("box_l"))
            .alias("box_l"),
        ])
        
        # Clean up temporary columns
        df = df.drop([
            "violates_boundary", 
            "period_group", 
            "is_valid_box_period",
            "unified_pivot_line",
            "box_h_unified",
            "box_l_unified",
            "box_h_check",
            "box_l_check",
            "is_box_candidate_temp",
        ])

        # 8. Calculate box width (for compatibility)
        df = df.with_columns([
            ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias("box_width"),
        ])

        # 8. Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df
