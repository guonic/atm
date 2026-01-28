"""
Dynamic Convergence Box Detector for Teapot pattern recognition.

Specifically designed to capture "true convergence" steps after wide oscillations.
Improves upon MovingAverageConvergenceBoxDetector by adding volatility decay detection.
"""

import logging
from typing import List, Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class DynamicConvergenceDetector(BoxDetector):
    """
    Dynamic Convergence Box Detector (动态收敛箱体检测器).
    
    Specifically improved to identify "true convergence" steps after wide oscillations.
    
    Core improvements over MovingAverageConvergenceBoxDetector:
    1. Inter-Quantile Range (分位差): Uses quantiles instead of Max-Min.
    2. Volatility Decay (波动率衰减): Detects when current volatility is much lower than background.
    3. MA Convergence Transition (均线收敛转换): Detects transition from divergence to convergence.
    
    This detector is particularly effective for:
    - True convergence boxes after wide oscillations (like the second yellow box in the image)
    - Filtering out false platforms with high volatility
    - Capturing energy accumulation periods before breakouts
    """

    def __init__(
        self,
        box_window: int = 30,
        ma_periods: Optional[List[int]] = None,
        max_tightness: float = 0.08,
        volness_threshold: float = 0.6,
        quantile_high: float = 0.85,
        quantile_low: float = 0.15,
        convergence_threshold: float = 0.02,
        ma_slope_window: int = 5,
        ma_slope_threshold: float = 0.015,
        price_trend_threshold: float = 0.05,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Dynamic Convergence Box Detector.

        Args:
            box_window: Window size for box calculation (default: 30 days).
            ma_periods: List of MA periods to use (default: [5, 10, 20]).
            max_tightness: Maximum box width (default: 0.08, 8%).
            volness_threshold: Volatility decay threshold (default: 0.6).
                Current volatility must be < background_vol * this threshold.
            quantile_high: High quantile for box upper bound (default: 0.85).
            quantile_low: Low quantile for box lower bound (default: 0.15).
            convergence_threshold: MA convergence threshold (default: 0.02, 2%).
            ma_slope_window: Window for MA slope calculation (default: 5 days).
            ma_slope_threshold: Maximum MA slope change (default: 0.015, 1.5%).
            price_trend_threshold: Maximum price trend change over box_window (default: 0.05, 5%).
            smooth_window: Window size for box filter smoothing (default: None, disabled).
            smooth_threshold: Minimum number of days in smooth_window (default: None, disabled).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        
        if ma_periods is None:
            ma_periods = [5, 10, 20]
        
        self.ma_periods = ma_periods
        self.max_tightness = max_tightness
        self.volness_threshold = volness_threshold
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low
        self.convergence_threshold = convergence_threshold
        self.ma_slope_window = ma_slope_window
        self.ma_slope_threshold = ma_slope_threshold
        self.price_trend_threshold = price_trend_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect box using dynamic convergence method with volatility decay.

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

        # 2. Calculate MA convergence
        # Use a reference MA (typically the middle one) for normalization
        reference_ma = f"ma_{self.ma_periods[len(self.ma_periods) // 2]}"
        
        # Calculate MA convergence: std(ma_values) / mean(ma_values)
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

        # 4. Calculate volatility decay (核心新逻辑)
        # Compare current window volatility with longer background window
        # Use entity range (open/close) instead of high/low for volatility
        df = df.with_columns([
            # Current small window volatility (using entity range)
            (pl.col("entity_high") - pl.col("entity_low"))
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("current_vol"),
            
            # Longer background window volatility (2x box_window)
            (pl.col("entity_high") - pl.col("entity_low"))
            .rolling_mean(window_size=self.box_window * 2)
            .over("ts_code")
            .alias("background_vol"),
        ])

        # 5. Calculate box bounds using entity quantiles (based on open/close, not high/low)
        # Using 85%/15% quantiles to handle spikes like the -10.26 dip in the image
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
                # Condition 1: MA convergence (识别纠缠)
                # Requires MAs to be tightly clustered (energy accumulation)
                (pl.col("ma_convergence") < self.convergence_threshold)
                &
                # Condition 2: Box width contraction (过滤宽幅震荡)
                # Filters out wide oscillations, only keeps tight boxes
                (pl.col("box_width") < self.max_tightness)
                &
                # Condition 3: Volatility decay (当前波动明显小于背景波动)
                # Current volatility must be significantly lower than background
                # This filters out false platforms with high volatility (like the left box in image)
                (pl.col("current_vol") < pl.col("background_vol") * self.volness_threshold)
                &
                # Condition 4: MA slope check (MA20 近似水平)
                # Ensures price center is relatively stable, not a declining slope
                (pl.col("ma_slope") < self.ma_slope_threshold)
                &
                # Condition 5: Price trend check (价格趋势应该接近水平，不是下降)
                # Price change over box_window should be small (true box is horizontal)
                (pl.col("price_trend") < self.price_trend_threshold)
                &
                # Condition 6: Price should be within box bounds (价格应该在箱体内)
                pl.col("price_in_box")
                &
                # Condition 7: Box bounds are valid
                pl.col("box_h").is_not_null()
                & pl.col("box_l").is_not_null()
                & (pl.col("box_h") > pl.col("box_l"))
                &
                # Condition 8: Volatility values are valid
                pl.col("current_vol").is_not_null()
                & pl.col("background_vol").is_not_null()
            ).alias("is_box_candidate"),
        ])

        # Apply smoothing if enabled
        df = self._apply_smoothing(df)

        return df
