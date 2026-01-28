"""
Moving Average Convergence Box Detector for Teapot pattern recognition.

Specifically designed to capture "step-like" consolidation patterns after downtrends.
"""

import logging
from typing import List, Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class MovingAverageConvergenceBoxDetector(BoxDetector):
    """
    Moving Average Convergence Box Detector (均线纠缠箱体检测器).
    
    Specifically designed to capture "step-like" consolidation patterns after downtrends.
    
    Core features:
    1. MA Convergence (均线纠缠度): Multiple moving averages converge (low std dev).
    2. Volatility Squeeze (波动收缩): Current volatility is much lower than previous downtrend.
    3. Price Position (价格定位): Price stays within box bounds with convergence after spikes.
    
    This detector is particularly effective for:
    - Downside consolidation steps after sharp declines
    - Teapot pattern "handle" or "body" sections
    - Periods where multiple MAs converge, indicating energy exhaustion and oscillation
    """

    def __init__(
        self,
        box_window: int = 20,
        ma_periods: Optional[List[int]] = None,
        convergence_threshold: float = 0.02,
        max_relative_height: float = 0.10,
        quantile_high: float = 0.90,
        quantile_low: float = 0.10,
        ma_slope_window: int = 5,
        ma_slope_threshold: float = 0.02,
        price_trend_threshold: float = 0.05,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Moving Average Convergence Box Detector.

        Args:
            box_window: Window size for box calculation (default: 20 days, shorter for step patterns).
            ma_periods: List of MA periods to use (default: [5, 10, 20, 30]).
            convergence_threshold: MA convergence threshold (default: 0.02, 2%).
                Lower values mean tighter convergence required.
            max_relative_height: Maximum relative box height (default: 0.10, 10%).
            quantile_high: High quantile for box upper bound (default: 0.90).
            quantile_low: Low quantile for box lower bound (default: 0.10).
            ma_slope_window: Window for MA slope calculation (default: 5 days).
            ma_slope_threshold: Maximum MA slope change (default: 0.02, 2%).
            price_trend_threshold: Maximum price trend change over box_window (default: 0.05, 5%).
            smooth_window: Window size for box filter smoothing (default: None, disabled).
            smooth_threshold: Minimum number of days in smooth_window that must be box candidates
                (default: None, disabled).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        
        if ma_periods is None:
            ma_periods = [5, 10, 20, 30]
        
        self.ma_periods = ma_periods
        self.convergence_threshold = convergence_threshold
        self.max_relative_height = max_relative_height
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low
        self.ma_slope_window = ma_slope_window
        self.ma_slope_threshold = ma_slope_threshold
        self.price_trend_threshold = price_trend_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using moving average convergence method.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.

        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # 1. Calculate multiple moving averages
        ma_cols = []
        for period in self.ma_periods:
            col_name = f"ma_{period}"
            df = df.with_columns([
                pl.col("close")
                .rolling_mean(window_size=period)
                .over("ts_code")
                .alias(col_name)
            ])
            ma_cols.append(col_name)

        # 2. Calculate core indicators
        # MA convergence: standard deviation of MAs / mean of MAs
        # Use a reference MA (typically the middle one) for normalization
        reference_ma = f"ma_{self.ma_periods[len(self.ma_periods) // 2]}"
        
        # Calculate MA convergence using horizontal std
        # Since polars doesn't have horizontal_std, we calculate it manually
        # MA convergence = std(ma_values) / mean(ma_values)
        ma_mean_expr = pl.sum_horizontal([pl.col(col) for col in ma_cols]) / len(ma_cols)
        
        # Calculate variance: sum((x - mean)^2) / n
        ma_variance_expr = (
            pl.sum_horizontal([
                (pl.col(col) - ma_mean_expr).pow(2) for col in ma_cols
            ]) / len(ma_cols)
        )
        
        # Standard deviation = sqrt(variance)
        ma_std_expr = ma_variance_expr.sqrt()
        
        # MA convergence = std / mean (normalized by reference MA)
        df = df.with_columns([
            (ma_std_expr / pl.col(reference_ma)).alias("ma_convergence"),
        ])

        # 3. Calculate entity range (open/close) instead of high/low
        # Entity high = max(open, close), Entity low = min(open, close)
        df = df.with_columns([
            pl.max_horizontal(["open", "close"]).alias("entity_high"),
            pl.min_horizontal(["open", "close"]).alias("entity_low"),
        ])

        # 4. Calculate price tightness using entity quantiles (filters spikes/shadow lines)
        df = df.with_columns([
            (
                (
                    pl.col("entity_high").rolling_quantile(
                        quantile=self.quantile_high, window_size=self.box_window
                    )
                    - pl.col("entity_low").rolling_quantile(
                        quantile=self.quantile_low, window_size=self.box_window
                    )
                )
                / pl.col("close").rolling_mean(self.box_window)
            )
            .over("ts_code")
            .alias("tightness"),
        ])

        # 5. Calculate box bounds using entity quantiles (based on open/close, not high/low)
        df = df.with_columns([
            pl.col("entity_high")
            .rolling_quantile(
                quantile=self.quantile_high, window_size=self.box_window
            )
            .shift(1)
            .over("ts_code")
            .alias("box_h"),
            pl.col("entity_low")
            .rolling_quantile(
                quantile=self.quantile_low, window_size=self.box_window
            )
            .shift(1)
            .over("ts_code")
            .alias("box_l"),
        ])

        # 6. Calculate box width (for compatibility)
        df = df.with_columns([
            ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias("box_width"),
        ])

        # 7. Calculate MA slope (price center stability)
        # Use the reference MA to measure slope
        df = df.with_columns([
            (
                (pl.col(reference_ma) - pl.col(reference_ma).shift(self.ma_slope_window))
                .abs()
                / pl.col(reference_ma).shift(self.ma_slope_window)
            ).alias("ma_slope"),
        ])

        # 8. Calculate price trend within box window (to filter downtrend oscillations)
        # Check if price is actually horizontal, not declining
        df = df.with_columns([
            # Price change over box_window (should be close to 0 for true box)
            (
                (pl.col("close") - pl.col("close").shift(self.box_window))
                .abs()
                / pl.col("close").shift(self.box_window)
            ).over("ts_code").alias("price_trend"),
            
            # Price position within box (should be within box bounds)
            (
                (pl.col("close") >= pl.col("box_l"))
                & (pl.col("close") <= pl.col("box_h"))
            ).alias("price_in_box"),
        ])

        # 9. Define box candidate conditions
        df = df.with_columns([
            (
                # Condition A: MA convergence (most important signal)
                # Indicates price entering directionless oscillation
                (pl.col("ma_convergence") < self.convergence_threshold)
                &
                # Condition B: Overall range is not too wide
                (pl.col("tightness") < self.max_relative_height)
                &
                # Condition C: Price center is relatively stable (MA slope < threshold)
                (pl.col("ma_slope") < self.ma_slope_threshold)
                &
                # Condition D: Price trend check (价格趋势应该接近水平，不是下降)
                # Price change over box_window should be small (true box is horizontal)
                (pl.col("price_trend") < self.price_trend_threshold)
                &
                # Condition E: Price should be within box bounds (价格应该在箱体内)
                pl.col("price_in_box")
                &
                # Condition F: Box bounds are valid
                pl.col("box_h").is_not_null()
                & pl.col("box_l").is_not_null()
                & (pl.col("box_h") > pl.col("box_l"))
            ).alias("is_box_candidate"),
        ])

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df
