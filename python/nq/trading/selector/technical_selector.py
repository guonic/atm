"""
Technical indicator-based stock selector.

Selects stocks based on technical indicators like price, volume, moving averages, etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from nq.repo.kline_repo import StockKlineDayRepo
from nq.repo.stock_repo import StockBasicRepo
from .base import BaseSelector, SelectionResult

logger = logging.getLogger(__name__)


class TechnicalSelector(BaseSelector):
    """
    Technical indicator-based stock selector.

    Selects stocks based on technical indicators such as:
    - Price levels (current price, price change)
    - Volume (trading volume, volume ratio)
    - Moving averages (SMA, EMA crossovers)
    - Technical indicators (RSI, MACD, etc.)
    """

    def __init__(
        self,
        db_config,
        schema: str = "quant",
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_volume: Optional[int] = None,
        min_price_change_pct: Optional[float] = None,
        max_price_change_pct: Optional[float] = None,
        min_days_listed: int = 60,  # Minimum days since listing
    ):
        """
        Initialize technical selector.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            min_price: Minimum current price.
            max_price: Maximum current price.
            min_volume: Minimum daily trading volume.
            min_price_change_pct: Minimum price change percentage (e.g., -10.0 for -10%).
            max_price_change_pct: Maximum price change percentage (e.g., 10.0 for +10%).
            min_days_listed: Minimum days since listing (default: 60).
        """
        super().__init__(db_config, schema)
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_price_change_pct = min_price_change_pct
        self.max_price_change_pct = max_price_change_pct
        self.min_days_listed = min_days_listed

    def select(
        self,
        exchange: Optional[str] = None,
        market: Optional[str] = None,
        min_list_date: Optional[datetime] = None,
        max_list_date: Optional[datetime] = None,
        reference_date: Optional[datetime] = None,
        **kwargs,
    ) -> SelectionResult:
        """
        Select stocks based on technical indicators.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE). If None, select from all exchanges.
            market: Market type (沪A/深A/北交所). If None, select from all markets.
            min_list_date: Minimum listing date.
            max_list_date: Maximum listing date.
            reference_date: Reference date for technical data (default: latest available).
            **kwargs: Additional selection criteria.

        Returns:
            SelectionResult containing selected stock codes.
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Get candidate stocks
        stock_repo = StockBasicRepo(self.db_config, self.schema)
        candidates = stock_repo.get_by_exchange(exchange) if exchange else stock_repo.get_all()

        # Filter by market if specified
        if market:
            candidates = [s for s in candidates if s.market == market]

        # Filter by listing date
        if min_list_date:
            candidates = [s for s in candidates if s.list_date >= min_list_date.date()]
        if max_list_date:
            candidates = [s for s in candidates if s.list_date <= max_list_date.date()]

        # Filter by minimum days listed
        if self.min_days_listed:
            min_date = (reference_date - timedelta(days=self.min_days_listed)).date()
            candidates = [s for s in candidates if s.list_date <= min_date]

        # Filter by listing status
        candidates = [s for s in candidates if s.is_listed]

        logger.info(f"Found {len(candidates)} candidate stocks")

        # Load K-line data and apply technical filters
        kline_repo = StockKlineDayRepo(self.db_config, self.schema)
        selected_stocks = []

        for stock in candidates:
            try:
                # Get latest K-line data
                klines = kline_repo.get_by_ts_code(
                    ts_code=stock.ts_code,
                    end_time=reference_date,
                    limit=1,  # Only need the latest day
                )

                if not klines:
                    continue

                latest_kline = klines[0]

                # Apply technical filters
                if self.min_price and latest_kline.close and latest_kline.close < self.min_price:
                    continue
                if self.max_price and latest_kline.close and latest_kline.close > self.max_price:
                    continue
                if self.min_volume and latest_kline.volume and latest_kline.volume < self.min_volume:
                    continue

                # Calculate price change percentage
                if latest_kline.close and latest_kline.pre_close:
                    price_change_pct = (
                        (latest_kline.close - latest_kline.pre_close) / latest_kline.pre_close * 100
                    )
                    if self.min_price_change_pct is not None and price_change_pct < self.min_price_change_pct:
                        continue
                    if self.max_price_change_pct is not None and price_change_pct > self.max_price_change_pct:
                        continue

                selected_stocks.append(stock.ts_code)

            except Exception as e:
                logger.debug(f"Error processing {stock.ts_code}: {e}")
                continue

        logger.info(f"Selected {len(selected_stocks)} stocks based on technical criteria")

        return SelectionResult(
            selected_stocks=selected_stocks,
            selection_date=reference_date,
            total_candidates=len(candidates),
            selection_criteria={
                "min_price": self.min_price,
                "max_price": self.max_price,
                "min_volume": self.min_volume,
                "min_price_change_pct": self.min_price_change_pct,
                "max_price_change_pct": self.max_price_change_pct,
                "min_days_listed": self.min_days_listed,
                "exchange": exchange,
                "market": market,
            },
            metadata={
                "selector_type": "technical",
            },
        )

