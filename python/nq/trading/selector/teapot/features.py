"""
Feature computation for Teapot pattern recognition.

Computes box features, regression R², and volume ratios.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import polars as pl

from nq.trading.selector.teapot.box_detector import (
    BoxDetector,
    SimpleBoxDetector,
)

logger = logging.getLogger(__name__)


class TeapotFeatures:
    """
    Feature calculator for Teapot pattern recognition.

    Computes box statistics, regression R², and volume ratios.
    """

    def __init__(
        self,
        box_window: int = 40,
        box_detector: Optional[BoxDetector] = None,
    ):
        """
        Initialize feature calculator.

        Args:
            box_window: Window size for box calculation (default: 40 days).
            box_detector: Optional box detector instance. If None, uses SimpleBoxDetector.
        """
        self.box_window = box_window
        self.box_detector = box_detector or SimpleBoxDetector(box_window=box_window)

    def compute_box_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute box features using configured box detector.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with added columns:
            - box_h: Box upper bound
            - box_l: Box lower bound
            - box_width: Box width (relative value)
            - is_box_candidate: Boolean indicating if current row is a box candidate
            - box_volatility: Box volatility (std of returns)
        """
        # Use box detector to compute box features
        df = self.box_detector.detect_box(df)

        # Add box volatility (std of returns within box)
        df = df.with_columns(
            [
                pl.col("close")
                .pct_change()
                .rolling_std(window_size=self.box_window)
                .shift(1)
                .over("ts_code")
                .alias("box_volatility"),
            ]
        )

        return df

    def compute_regression_r2(
        self, df: pl.DataFrame, window: int = 40
    ) -> pl.Series:
        """
        Compute linear regression R².

        Used to determine if price oscillates around regression line.

        Args:
            df: Input DataFrame with columns: trade_date, close.
            window: Window size for regression.

        Returns:
            Series with R² values.
        """
        # This is a simplified version - in production, you might want to use
        # a more efficient implementation or UDF
        dates = df["trade_date"].to_list()
        closes = df["close"].to_list()

        r2_values = []
        for i in range(len(df)):
            if i < window:
                r2_values.append(None)
            else:
                window_dates = dates[i - window : i]
                window_closes = closes[i - window : i]

                # Convert dates to numeric (days since start)
                start_date = window_dates[0]
                x = [
                    (d - start_date).days if isinstance(d, pl.Date) else 0
                    for d in window_dates
                ]
                y = window_closes

                # Compute R²
                if len(x) > 1 and len(set(x)) > 1:
                    try:
                        coeffs = np.polyfit(x, y, 1)
                        y_pred = np.polyval(coeffs, x)
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        r2_values.append(r2)
                    except Exception:
                        r2_values.append(None)
                else:
                    r2_values.append(None)

        return pl.Series("regression_r2", r2_values)

    def compute_volume_ratio(
        self, df: pl.DataFrame, window: int = 20
    ) -> pl.Series:
        """
        Compute volume ratio (current volume / average volume).

        Args:
            df: Input DataFrame with columns: volume.
            window: Window size for average volume calculation.

        Returns:
            Series with volume ratios.
        """
        return (
            pl.col("volume")
            / pl.col("volume")
            .rolling_mean(window_size=window)
            .shift(1)
            .over("ts_code")
            ).alias("vol_ratio")

    def compute_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with all computed features.
        """
        # Compute box features
        df = self.compute_box_features(df)

        # Compute regression R² (grouped by stock)
        # Note: This is simplified - in production, apply per stock group
        df = df.with_columns(
            [
                self.compute_regression_r2(df, window=self.box_window).alias(
                    "regression_r2"
                ),
                self.compute_volume_ratio(df, window=20).alias("vol_ratio"),
            ]
        )

        return df
