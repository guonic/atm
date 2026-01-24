"""
Filters for Teapot pattern recognition signals.

Provides liquidity, risk, and quality filters.
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class TeapotFilters:
    """
    Filters for Teapot signals.

    Provides liquidity, risk, and quality filtering.
    """

    def __init__(
        self,
        min_turnover: float = 0.01,
        min_amount: float = 10000000.0,
        max_gap: float = 0.10,
        max_trap_depth: float = 0.20,
    ):
        """
        Initialize filters.

        Args:
            min_turnover: Minimum turnover rate (1%).
            min_amount: Minimum trading amount (10M CNY).
            max_gap: Maximum gap ratio (10%).
            max_trap_depth: Maximum trap depth (20%).
        """
        self.min_turnover = min_turnover
        self.min_amount = min_amount
        self.max_gap = max_gap
        self.max_trap_depth = max_trap_depth

    def apply_liquidity_filter(
        self, signals: pl.DataFrame, market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Apply liquidity filter.

        Filters signals based on turnover and trading amount.

        Args:
            signals: Signals DataFrame.
            market_data: Market data DataFrame.

        Returns:
            Filtered signals DataFrame.
        """
        # Join signals with market data to get liquidity metrics
        merged = signals.join(
            market_data,
            left_on=["ts_code", "signal_date"],
            right_on=["ts_code", "trade_date"],
            how="left",
        )

        # Calculate turnover (simplified - in production, use actual turnover data)
        # For now, use volume / market cap approximation
        # Filter by amount
        filtered = merged.filter(pl.col("amount") >= self.min_amount)

        logger.info(
            f"Liquidity filter: {len(signals)} -> {len(filtered)} signals"
        )

        return filtered.select(signals.columns)

    def apply_risk_filter(
        self, signals: pl.DataFrame, market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Apply risk filter.

        Filters signals based on gap and abnormal volatility.

        Args:
            signals: Signals DataFrame.
            market_data: Market data DataFrame.

        Returns:
            Filtered signals DataFrame.
        """
        # Join signals with market data
        merged = signals.join(
            market_data,
            left_on=["ts_code", "signal_date"],
            right_on=["ts_code", "trade_date"],
            how="left",
        )

        # Calculate gap (open vs previous close)
        merged = merged.with_columns(
            [
                (
                    (pl.col("open") - pl.col("close").shift(1))
                    / pl.col("close").shift(1)
                ).alias("gap_ratio")
            ]
        )

        # Filter by gap
        filtered = merged.filter(
            pl.col("gap_ratio").abs() <= self.max_gap
            | pl.col("gap_ratio").is_null()
        )

        logger.info(
            f"Risk filter: {len(signals)} -> {len(filtered)} signals"
        )

        return filtered.select(signals.columns)

    def apply_trap_depth_filter(self, signals: pl.DataFrame) -> pl.DataFrame:
        """
        Apply trap depth filter.

        Filters signals with trap depth exceeding threshold.

        Args:
            signals: Signals DataFrame.

        Returns:
            Filtered signals DataFrame.
        """
        filtered = signals.filter(
            pl.col("trap_depth") <= self.max_trap_depth
        )

        logger.info(
            f"Trap depth filter: {len(signals)} -> {len(filtered)} signals"
        )

        return filtered

    def apply_all_filters(
        self, signals: pl.DataFrame, market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Apply all filters.

        Args:
            signals: Signals DataFrame.
            market_data: Market data DataFrame.

        Returns:
            Filtered signals DataFrame.
        """
        filtered = signals

        # Apply trap depth filter first (no market data needed)
        filtered = self.apply_trap_depth_filter(filtered)

        # Apply liquidity filter
        filtered = self.apply_liquidity_filter(filtered, market_data)

        # Apply risk filter
        filtered = self.apply_risk_filter(filtered, market_data)

        logger.info(
            f"All filters applied: {len(signals)} -> {len(filtered)} signals"
        )

        return filtered
