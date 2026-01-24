"""
State machine for Teapot pattern recognition.

Implements four-state detection logic: Box -> Trap -> Reverse -> Breakout.
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class TeapotStateMachine:
    """
    State machine for Teapot pattern recognition.

    Detects four sequential states:
    1. State A (Box): Price oscillation within narrow range
    2. State B (Trap): Price breaks below box lower bound
    3. State C (Reverse): Price recovers back into box
    4. State D (Breakout): Price breaks above box upper bound with volume
    """

    def __init__(
        self,
        box_window: int = 40,
        box_volatility_threshold: float = 0.90,
        box_r2_threshold: float = 0.5,
        trap_max_depth: float = 0.20,
        trap_max_days: int = 5,
        reverse_max_days: int = 10,
        reverse_recover_ratio: float = 0.8,
        breakout_vol_ratio: float = 1.5,
    ):
        """
        Initialize state machine.

        Args:
            box_window: Window size for box calculation.
            box_volatility_threshold: Maximum box width (15%).
            box_r2_threshold: Minimum R² for box quality (0.7).
            trap_max_depth: Maximum trap depth (20% of box height).
            trap_max_days: Maximum days for trap state.
            reverse_max_days: Maximum days for reverse state.
            reverse_recover_ratio: Minimum recovery ratio (80%).
            breakout_vol_ratio: Minimum volume ratio for breakout (1.5x).
        """
        self.box_window = box_window
        self.box_volatility_threshold = box_volatility_threshold
        self.box_r2_threshold = box_r2_threshold
        self.trap_max_depth = trap_max_depth
        self.trap_max_days = trap_max_days
        self.reverse_max_days = reverse_max_days
        self.reverse_recover_ratio = reverse_recover_ratio
        self.breakout_vol_ratio = breakout_vol_ratio

    def detect_state_a_box(self, df: pl.DataFrame) -> pl.Series:
        """
        Detect State A: Box oscillation.

        Conditions:
        - Box width < threshold (15%)
        - R² > threshold (price oscillates around regression line)

        Args:
            df: DataFrame with box features.

        Returns:
            Boolean Series indicating State A.
        """
        return (
            (pl.col("box_width") < 0.9)
            & (pl.col("box_width") > 0)
            & (pl.col("regression_r2") > 0.1)
            & (pl.col("regression_r2").is_not_null())
        )
        return (
            (pl.col("box_width") < self.box_volatility_threshold)
            & (pl.col("box_width") > 0)
            & (pl.col("regression_r2") > self.box_r2_threshold)
            & (pl.col("regression_r2").is_not_null())
        )

    def detect_state_b_trap(self, df: pl.DataFrame) -> pl.Series:
        """
        Detect State B: Trap (price breaks below box).

        Conditions:
        - Close price < box lower bound
        - Trap depth < max_depth (20% of box height)
        - Duration <= max_days (1-5 days)

        Args:
            df: DataFrame with box features.

        Returns:
            Boolean Series indicating State B.
        """
        # Check if close is below box lower bound
        below_box = pl.col("close") < pl.col("box_l")

        # Calculate trap depth
        trap_depth = (pl.col("box_l") - pl.col("close")) / pl.col("box_l")
        depth_ok = trap_depth <= self.trap_max_depth

        return below_box & depth_ok & pl.col("box_l").is_not_null()

    def detect_state_c_reverse(self, df: pl.DataFrame) -> pl.Series:
        """
        Detect State C: Reverse (price recovers into box).

        Conditions:
        - Price recovers back into box (close > box_l)
        - Recovery happens within max_days after trap
        - Recovery ratio >= threshold (80%)

        Args:
            df: DataFrame with box features.

        Returns:
            Boolean Series indicating State C.
        """
        # Check if close is back above box lower bound
        recovered = pl.col("close") > pl.col("box_l")

        # Calculate recovery ratio
        recovery_ratio = (pl.col("close") - pl.col("box_l")) / (
            pl.col("box_h") - pl.col("box_l")
        )
        recovery_ok = recovery_ratio >= self.reverse_recover_ratio

        return (
            recovered
            & recovery_ok
            & pl.col("box_l").is_not_null()
            & pl.col("box_h").is_not_null()
        )

    def detect_state_d_breakout(self, df: pl.DataFrame) -> pl.Series:
        """
        Detect State D: Breakout (price breaks above box with volume).

        Conditions:
        - Close price > box upper bound
        - Volume ratio > threshold (1.5x)
        - This is the signal trigger point

        Args:
            df: DataFrame with box features and volume ratio.

        Returns:
            Boolean Series indicating State D (signals).
        """
        # Check if close breaks above box upper bound
        breakout = pl.col("close") > pl.col("box_h")

        # Check volume ratio
        vol_ok = pl.col("vol_ratio") >= self.breakout_vol_ratio

        return (
            breakout
            & vol_ok
            & pl.col("box_h").is_not_null()
            & pl.col("vol_ratio").is_not_null()
        )

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate complete signals by detecting sequential states.

        States must occur in order: A -> B -> C -> D

        Args:
            df: DataFrame with all features computed.

        Returns:
            DataFrame with signals, containing columns:
            - ts_code: Stock code
            - signal_date: Signal date
            - box_h: Box upper bound
            - box_l: Box lower bound
            - box_width: Box width
            - trap_depth: Trap depth
            - breakout_vol_ratio: Breakout volume ratio
            - signal_score: Signal score (optional)
        """
        # Detect each state
        state_a = self.detect_state_a_box(df)
        state_b = self.detect_state_b_trap(df)
        state_c = self.detect_state_c_reverse(df)
        state_d = self.detect_state_d_breakout(df)

        # State sequence: A -> B -> C -> D
        # B must occur after A, C after B, D after C
        # Use shift to check previous states

        # Mark states
        df = df.with_columns(
            [
                state_a.alias("state_a"),
                state_b.alias("state_b"),
                state_c.alias("state_c"),
                state_d.alias("state_d"),
            ]
        )

        # Track state sequence per stock
        # For each stock, check if states occurred in sequence
        signals = []

        # Group by stock and process sequentially
        for ts_code in df["ts_code"].unique():
            stock_df = df.filter(pl.col("ts_code") == ts_code).sort("trade_date")

            # Track state progression
            in_box = False
            in_trap = False
            in_reverse = False

            for row in stock_df.iter_rows(named=True):
                if row["state_a"]:
                    in_box = True
                    in_trap = False
                    in_reverse = False
                elif row["state_b"] and in_box:
                    in_trap = True
                    in_reverse = False
                elif row["state_c"] and in_trap:
                    in_reverse = True
                elif row["state_d"] and in_reverse:
                    # Signal triggered!
                    trap_depth = (
                        (row["box_l"] - row["close"]) / row["box_l"]
                        if row["box_l"] > 0
                        else 0
                    )
                    signals.append(
                        {
                            "ts_code": row["ts_code"],
                            "signal_date": row["trade_date"],
                            "box_h": row["box_h"],
                            "box_l": row["box_l"],
                            "box_width": row["box_width"],
                            "trap_depth": trap_depth,
                            "breakout_vol_ratio": row["vol_ratio"],
                            "signal_score": self._calculate_signal_score(row),
                        }
                    )
                    # Reset states after signal
                    in_box = False
                    in_trap = False
                    in_reverse = False

        if not signals:
            return pl.DataFrame(
                schema={
                    "ts_code": pl.Utf8,
                    "signal_date": pl.Utf8,
                    "box_h": pl.Float64,
                    "box_l": pl.Float64,
                    "box_width": pl.Float64,
                    "trap_depth": pl.Float64,
                    "breakout_vol_ratio": pl.Float64,
                    "signal_score": pl.Float64,
                }
            )

        return pl.DataFrame(signals)

    def _calculate_signal_score(self, row: dict) -> float:
        """
        Calculate signal score (0-1).

        Args:
            row: Signal row dictionary.

        Returns:
            Signal score.
        """
        score = 0.5  # Base score

        # Higher volume ratio -> higher score
        if row["vol_ratio"] > self.breakout_vol_ratio:
            score += 0.2

        # Smaller trap depth -> higher score
        if row.get("trap_depth", 0) < self.trap_max_depth * 0.5:
            score += 0.2

        # Box width in optimal range
        if 0.05 < row["box_width"] < 0.12:
            score += 0.1

        return min(score, 1.0)
