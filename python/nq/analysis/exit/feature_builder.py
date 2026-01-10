"""
Feature builder for exit model.

Constructs features for predicting when to exit positions based on:
1. Momentum exhaustion indicators
2. Risk asymmetry / position management
3. Time-based features
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExitFeatureBuilder:
    """
    Feature builder for exit prediction model.
    
    Constructs features that capture:
    - Momentum exhaustion (bias, close position, volume ratio)
    - Risk asymmetry (current return, drawdown)
    - Time-based features (days held)
    """

    def __init__(self, ma_period: int = 5, volume_ma_period: int = 5):
        """
        Initialize feature builder.
        
        Args:
            ma_period: Period for moving average calculation (default: 5).
            volume_ma_period: Period for volume moving average (default: 5).
        """
        self.ma_period = ma_period
        self.volume_ma_period = volume_ma_period

    def build_features(
        self,
        daily_df: pd.DataFrame,
        entry_price: Optional[float] = None,
        highest_price_since_entry: Optional[float] = None,
        days_held: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build exit prediction features from daily data.
        
        Args:
            daily_df: DataFrame with columns:
                - close: Close price
                - high: High price
                - low: Low price
                - volume: Volume
            entry_price: Entry price for the position (if None, uses 'entry_price' column).
            highest_price_since_entry: Highest price since entry (if None, uses 'highest_price_since_entry' column).
            days_held: Days held (if None, uses 'days_held' column).
        
        Returns:
            DataFrame with features:
                - bias_5: Deviation from 5-day MA
                - close_pos: Close position in daily range (0-1)
                - vol_ratio: Volume ratio to MA
                - curr_ret: Current return
                - drawdown: Drawdown from highest price
                - days_held: Days held
        """
        if daily_df.empty:
            logger.warning("Empty daily_df provided, returning empty features")
            return pd.DataFrame()

        # Ensure required columns exist
        required_cols = ["close", "high", "low", "volume"]
        missing_cols = [col for col in required_cols if col not in daily_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        feat = pd.DataFrame(index=daily_df.index)

        # --- 1. Momentum Exhaustion Indicators ---

        # Bias from 5-day MA: deviation from moving average
        ma5 = daily_df["close"].rolling(self.ma_period, min_periods=1).mean()
        feat["bias_5"] = (daily_df["close"] - ma5) / (ma5 + 1e-6)

        # Close position in daily range (0-1)
        # If close is near low (close_pos ~ 0), it's a bearish signal
        # If close is near high (close_pos ~ 1), it's a bullish signal
        price_range = daily_df["high"] - daily_df["low"]
        feat["close_pos"] = (daily_df["close"] - daily_df["low"]) / (
            price_range + 1e-6
        )

        # Volume ratio: today's volume vs 5-day average
        vol_ma = daily_df["volume"].rolling(
            self.volume_ma_period, min_periods=1
        ).mean()
        feat["vol_ratio"] = daily_df["volume"] / (vol_ma + 1e-6)

        # --- 2. Risk Asymmetry / Position Management ---

        # Get entry price
        if entry_price is None:
            if "entry_price" in daily_df.columns:
                entry_price = daily_df["entry_price"]
            else:
                logger.warning(
                    "No entry_price provided and not in daily_df, setting to 0"
                )
                entry_price = 0.0

        if isinstance(entry_price, pd.Series):
            feat["curr_ret"] = (daily_df["close"] - entry_price) / (
                entry_price + 1e-6
            )
        else:
            feat["curr_ret"] = (daily_df["close"] - entry_price) / (
                entry_price + 1e-6
            )

        # Drawdown from highest price since entry
        if highest_price_since_entry is None:
            if "highest_price_since_entry" in daily_df.columns:
                highest_price_since_entry = daily_df["highest_price_since_entry"]
            else:
                # Use current close as fallback
                highest_price_since_entry = daily_df["close"]

        if isinstance(highest_price_since_entry, pd.Series):
            feat["drawdown"] = (
                highest_price_since_entry - daily_df["close"]
            ) / (highest_price_since_entry + 1e-6)
        else:
            feat["drawdown"] = (highest_price_since_entry - daily_df["close"]) / (
                highest_price_since_entry + 1e-6
            )

        # Days held
        if days_held is None:
            if "days_held" in daily_df.columns:
                feat["days_held"] = daily_df["days_held"]
            else:
                logger.warning(
                    "No days_held provided and not in daily_df, setting to 0"
                )
                feat["days_held"] = 0
        else:
            if isinstance(days_held, (int, float)):
                feat["days_held"] = days_held
            else:
                feat["days_held"] = days_held

        # Drop rows with NaN (shouldn't happen after min_periods=1, but safe)
        feat = feat.dropna()

        return feat

    def build_label(
        self,
        daily_df: pd.DataFrame,
        future_days: int = 3,
        loss_threshold: float = -0.03,
    ) -> pd.Series:
        """
        Build label for exit prediction.
        
        Label = 1 if future max loss exceeds threshold or return becomes negative.
        Label = 0 otherwise.
        
        Args:
            daily_df: DataFrame with future return data.
            future_days: Number of future days to look ahead (default: 3).
            loss_threshold: Loss threshold for labeling (default: -0.03, i.e., -3%).
        
        Returns:
            Series with labels (0 or 1).
        """
        if "next_3d_max_loss" in daily_df.columns:
            # Use pre-calculated future loss
            label = (daily_df["next_3d_max_loss"] < loss_threshold).astype(int)
        elif "future_max_loss" in daily_df.columns:
            label = (daily_df["future_max_loss"] < loss_threshold).astype(int)
        else:
            logger.warning(
                "No future loss column found, cannot build label. "
                "Please provide 'next_3d_max_loss' or 'future_max_loss' column."
            )
            return pd.Series(dtype=int, index=daily_df.index)

        return label

    def build_features_with_label(
        self,
        daily_df: pd.DataFrame,
        entry_price: Optional[float] = None,
        highest_price_since_entry: Optional[float] = None,
        days_held: Optional[int] = None,
        future_days: int = 3,
        loss_threshold: float = -0.03,
    ) -> pd.DataFrame:
        """
        Build features and label together.
        
        Args:
            daily_df: DataFrame with daily data and future return data.
            entry_price: Entry price for the position.
            highest_price_since_entry: Highest price since entry.
            days_held: Days held.
            future_days: Number of future days to look ahead.
            loss_threshold: Loss threshold for labeling.
        
        Returns:
            DataFrame with features and 'label' column.
        """
        # Build features
        feat = self.build_features(
            daily_df=daily_df,
            entry_price=entry_price,
            highest_price_since_entry=highest_price_since_entry,
            days_held=days_held,
        )

        # Build label
        label = self.build_label(
            daily_df=daily_df, future_days=future_days, loss_threshold=loss_threshold
        )

        # Align indices
        feat = feat.join(label.to_frame("label"), how="left")

        return feat.dropna()
