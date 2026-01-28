"""
Base selector for Teapot pattern recognition.

Main selector class that coordinates all modules.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import polars as pl

from nq.config import DatabaseConfig
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.base import BaseSelector, SelectionResult
from nq.trading.selector.teapot.box_detector import (
    BoxDetector,
    HybridBoxDetector,
    HybridBoxDetectorV2,
    MeanReversionBoxDetector,
    MeanReversionBoxDetectorV2,
    SimpleBoxDetector,
)
from nq.trading.selector.teapot.box_detector_accurate import (
    AccurateBoxDetector,
)
from nq.trading.selector.teapot.box_detector_dynamic_convergence import (
    DynamicConvergenceDetector,
)
from nq.trading.selector.teapot.box_detector_keltner_squeeze import (
    ExpansionAnchorBoxDetector,
    KeltnerSqueezeDetector,
)
from nq.trading.selector.teapot.box_detector_ma_convergence import (
    MovingAverageConvergenceBoxDetector,
)
from nq.trading.selector.teapot.features import TeapotFeatures
from nq.trading.selector.teapot.filters import TeapotFilters
from nq.trading.selector.teapot.state_machine import TeapotStateMachine

logger = logging.getLogger(__name__)


class TeapotSelector(BaseSelector):
    """
    Teapot pattern recognition selector.

    Coordinates data loading, feature computation, state machine, and filtering.
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        schema: str = "quant",
        config: Optional[dict] = None,
    ):
        """
        Initialize Teapot selector.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            config: Optional configuration dictionary.
        """
        super().__init__(db_config, schema)

        # Load configuration
        self.config = config or {}
        box_window = self.config.get("box_window", 40)
        use_cache = self.config.get("use_cache", False)
        strict_cache = self.config.get("strict_cache", False)
        cache_dir = self.config.get("cache_dir")

        # Initialize box detector
        box_detector = self._create_box_detector(box_window)

        # Initialize components
        self.data_loader = TeapotDataLoader(
            db_config=db_config,
            schema=schema,
            use_cache=use_cache,
            cache_dir=cache_dir,
            strict_cache=strict_cache,
        )

        self.features = TeapotFeatures(box_window=box_window, box_detector=box_detector)
        self.state_machine = TeapotStateMachine(
            box_window=box_window,
            box_volatility_threshold=self.config.get(
                "box_volatility_threshold", 0.15
            ),
            box_r2_threshold=self.config.get("box_r2_threshold", 0.7),
            trap_max_depth=self.config.get("trap_max_depth", 0.20),
            trap_max_days=self.config.get("trap_max_days", 5),
            reverse_max_days=self.config.get("reverse_max_days", 10),
            reverse_recover_ratio=self.config.get(
                "reverse_recover_ratio", 0.8
            ),
            breakout_vol_ratio=self.config.get("breakout_vol_ratio", 1.5),
        )

        self.filters = TeapotFilters(
            min_turnover=self.config.get("min_turnover", 0.01),
            min_amount=self.config.get("min_amount", 10000000.0),
            max_gap=self.config.get("max_gap", 0.10),
            max_trap_depth=self.config.get("max_trap_depth", 0.20),
        )

    def _create_box_detector(self, box_window: int) -> BoxDetector:
        """
        Create box detector based on configuration.

        Args:
            box_window: Window size for box calculation.

        Returns:
            BoxDetector instance.
        """
        detector_type = self.config.get("box_detector_type", "simple")
        smooth_window = self.config.get("box_smooth_window")
        smooth_threshold = self.config.get("box_smooth_threshold")

        if detector_type == "simple":
            return SimpleBoxDetector(
                box_window=box_window,
                box_width_threshold=self.config.get("box_width_threshold", 0.15),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "mean_reversion":
            return MeanReversionBoxDetector(
                box_window=box_window,
                max_total_return=self.config.get("max_total_return", 0.05),
                max_relative_box_height=self.config.get(
                    "max_relative_box_height", 0.08
                ),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "hybrid":
            return HybridBoxDetector(
                box_window=box_window,
                box_width_threshold=self.config.get("box_width_threshold", 0.15),
                max_total_return=self.config.get("max_total_return", 0.05),
                max_relative_box_height=self.config.get(
                    "max_relative_box_height", 0.08
                ),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "mean_reversion_v2":
            return MeanReversionBoxDetectorV2(
                box_window=box_window,
                max_total_return=self.config.get("max_total_return"),  # None for volatility-based
                max_relative_box_height=self.config.get(
                    "max_relative_box_height", 0.12
                ),
                volatility_ratio=self.config.get("volatility_ratio", 0.25),
                quantile_high=self.config.get("quantile_high", 0.95),
                quantile_low=self.config.get("quantile_low", 0.05),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "hybrid_v2":
            return HybridBoxDetectorV2(
                box_window=box_window,
                box_width_threshold=self.config.get("box_width_threshold", 0.15),
                max_total_return=self.config.get("max_total_return"),  # None for volatility-based
                max_relative_box_height=self.config.get(
                    "max_relative_box_height", 0.12
                ),
                volatility_ratio=self.config.get("volatility_ratio", 0.25),
                score_threshold=self.config.get("score_threshold", 0.5),
                weight_simple=self.config.get("weight_simple", 0.3),
                weight_mean_reversion=self.config.get("weight_mean_reversion", 0.4),
                weight_mean_reversion_v2=self.config.get("weight_mean_reversion_v2", 0.3),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "ma_convergence":
            return MovingAverageConvergenceBoxDetector(
                box_window=box_window,
                ma_periods=self.config.get("ma_periods", [5, 10, 20, 30]),
                convergence_threshold=self.config.get("convergence_threshold", 0.02),
                max_relative_height=self.config.get("max_relative_height", 0.10),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "dynamic_convergence":
            return DynamicConvergenceDetector(
                box_window=box_window,
                ma_periods=self.config.get("ma_periods", [5, 10, 20]),
                max_tightness=self.config.get("max_tightness", 0.08),
                volness_threshold=self.config.get("volness_threshold", 0.6),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "keltner_squeeze":
            return KeltnerSqueezeDetector(
                box_window=box_window,
                squeeze_threshold=self.config.get("squeeze_threshold", 0.06),
                slope_threshold=self.config.get("slope_threshold", 0.015),
                atr_multiplier=self.config.get("atr_multiplier", 1.5),
                volume_decay_threshold=self.config.get("volume_decay_threshold", 0.8),
                expansion_window=self.config.get("expansion_window", 15),
                stability_window=self.config.get("stability_window", 20),
                stability_threshold=self.config.get("stability_threshold", 15),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "expansion_anchor":
            return ExpansionAnchorBoxDetector(
                box_window=box_window,
                squeeze_threshold=self.config.get("squeeze_threshold", 0.06),
                slope_threshold=self.config.get("slope_threshold", 0.015),
                atr_multiplier=self.config.get("atr_multiplier", 1.5),
                lookback_window=self.config.get("lookback_window", 60),
                expansion_window=self.config.get("expansion_window", 20),
                stability_window=self.config.get("stability_window", 20),
                stability_threshold=self.config.get("stability_threshold", 15),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        elif detector_type == "accurate":
            return AccurateBoxDetector(
                box_window=box_window,
                price_tol=self.config.get("price_tol", 0.03),
                min_cross_count=self.config.get("min_cross_count", 3),
                smooth_window=smooth_window,
                smooth_threshold=smooth_threshold,
            )
        else:
            logger.warning(
                f"Unknown box_detector_type: {detector_type}, using simple"
            )
            return SimpleBoxDetector(box_window=box_window)

    def select(
        self,
        exchange: Optional[str] = None,
        market: Optional[str] = None,
        min_list_date: Optional[datetime] = None,
        max_list_date: Optional[datetime] = None,
        **kwargs,
    ) -> SelectionResult:
        """
        Select stocks based on Teapot pattern.

        Args:
            exchange: Exchange code (not used in Teapot).
            market: Market type (not used in Teapot).
            min_list_date: Minimum listing date (not used in Teapot).
            max_list_date: Maximum listing date (not used in Teapot).
            **kwargs: Additional parameters:
                - selection_date: Selection date (required).
                - start_date: Start date for scanning (optional).
                - end_date: End date for scanning (optional).
                - symbols: List of stock codes to scan (optional).

        Returns:
            SelectionResult with selected stocks.
        """
        selection_date = kwargs.get("selection_date")
        if not selection_date:
            raise ValueError("selection_date is required")

        # Use selection_date as end_date, scan last N days
        lookback_days = kwargs.get("lookback_days", 365)
        start_date = kwargs.get(
            "start_date",
            (selection_date - timedelta(days=lookback_days)).strftime(
                "%Y-%m-%d"
            ),
        )
        end_date = kwargs.get(
            "end_date", selection_date.strftime("%Y-%m-%d")
        )

        symbols = kwargs.get("symbols")

        # Scan market
        signals = self.scan_market(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

        # Filter signals up to selection_date
        signals = signals.filter(
            pl.col("signal_date") <= selection_date.strftime("%Y-%m-%d")
        )

        # Get unique stock codes
        selected_stocks = signals["ts_code"].unique().to_list()

        return SelectionResult(
            selected_stocks=selected_stocks,
            selection_date=selection_date,
            total_candidates=len(signals),
            selection_criteria={
                "pattern": "teapot",
                "start_date": start_date,
                "end_date": end_date,
            },
            metadata={
                "total_signals": len(signals),
                "signals": signals.to_dicts() if not signals.is_empty() else [],
            },
        )

    def scan_market(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Scan market for Teapot signals.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes.

        Returns:
            DataFrame with detected signals.
        """
        logger.info(f"Scanning market: {start_date} to {end_date}")

        # 1. Load data
        market_data = self.data_loader.load_daily_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

        if market_data.is_empty():
            logger.warning("No market data loaded")
            return pl.DataFrame()

        # 2. Compute features
        logger.info("Computing features...")
        market_data = self.features.compute_all_features(market_data)

        # 3. Detect states and generate signals
        logger.info("Detecting states and generating signals...")
        signals = self.state_machine.generate_signals(market_data)

        if signals.is_empty():
            logger.info("No signals detected")
            return signals

        # 4. Apply filters
        logger.info("Applying filters...")
        signals = self.filters.apply_all_filters(signals, market_data)

        logger.info(f"Scan complete: {len(signals)} signals detected")

        return signals
