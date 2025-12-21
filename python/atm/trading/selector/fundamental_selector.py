"""
Fundamental analysis-based stock selector.

Selects stocks based on fundamental indicators like market cap, PE ratio, PB ratio, etc.
"""

import logging
from datetime import datetime
from typing import Optional

from atm.repo.stock_repo import StockBasicRepo, StockFinanceBasicRepo
from .base import BaseSelector, SelectionResult

logger = logging.getLogger(__name__)


class FundamentalSelector(BaseSelector):
    """
    Fundamental analysis-based stock selector.

    Selects stocks based on fundamental indicators such as:
    - Market capitalization (total_mv, float_mv)
    - Valuation ratios (PE, PB)
    - Share structure (total_share, float_share)
    """

    def __init__(
        self,
        db_config,
        schema: str = "quant",
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_pe: Optional[float] = None,
        max_pe: Optional[float] = None,
        min_pb: Optional[float] = None,
        max_pb: Optional[float] = None,
        use_latest_finance: bool = True,
    ):
        """
        Initialize fundamental selector.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            min_market_cap: Minimum market capitalization (in 10K CNY).
            max_market_cap: Maximum market capitalization (in 10K CNY).
            min_pe: Minimum PE ratio (TTM).
            max_pe: Maximum PE ratio (TTM).
            min_pb: Minimum PB ratio.
            max_pb: Maximum PB ratio.
            use_latest_finance: Whether to use latest finance data (default: True).
        """
        super().__init__(db_config, schema)
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_pe = min_pe
        self.max_pe = max_pe
        self.min_pb = min_pb
        self.max_pb = max_pb
        self.use_latest_finance = use_latest_finance

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
        Select stocks based on fundamental indicators.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE). If None, select from all exchanges.
            market: Market type (沪A/深A/北交所). If None, select from all markets.
            min_list_date: Minimum listing date.
            max_list_date: Maximum listing date.
            reference_date: Reference date for finance data (default: latest available).
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

        # Filter by listing status
        candidates = [s for s in candidates if s.is_listed]

        logger.info(f"Found {len(candidates)} candidate stocks")

        # Load finance data and apply fundamental filters
        finance_repo = StockFinanceBasicRepo(self.db_config, self.schema)
        selected_stocks = []

        for stock in candidates:
            try:
                # Get latest finance data
                finance_data_list = finance_repo.get_by_ts_code(
                    ts_code=stock.ts_code,
                    limit=1,
                )

                if not finance_data_list:
                    continue

                finance_data = finance_data_list[0]

                # Apply fundamental filters
                if self.min_market_cap and finance_data.total_mv:
                    if float(finance_data.total_mv) < self.min_market_cap:
                        continue
                if self.max_market_cap and finance_data.total_mv:
                    if float(finance_data.total_mv) > self.max_market_cap:
                        continue

                if self.min_pe and finance_data.pe_ttm:
                    if float(finance_data.pe_ttm) < self.min_pe:
                        continue
                if self.max_pe and finance_data.pe_ttm:
                    if float(finance_data.pe_ttm) > self.max_pe:
                        continue

                if self.min_pb and finance_data.pb:
                    if float(finance_data.pb) < self.min_pb:
                        continue
                if self.max_pb and finance_data.pb:
                    if float(finance_data.pb) > self.max_pb:
                        continue

                selected_stocks.append(stock.ts_code)

            except Exception as e:
                logger.debug(f"Error processing {stock.ts_code}: {e}")
                continue

        logger.info(f"Selected {len(selected_stocks)} stocks based on fundamental criteria")

        return SelectionResult(
            selected_stocks=selected_stocks,
            selection_date=reference_date,
            total_candidates=len(candidates),
            selection_criteria={
                "min_market_cap": self.min_market_cap,
                "max_market_cap": self.max_market_cap,
                "min_pe": self.min_pe,
                "max_pe": self.max_pe,
                "min_pb": self.min_pb,
                "max_pb": self.max_pb,
                "exchange": exchange,
                "market": market,
            },
            metadata={
                "selector_type": "fundamental",
            },
        )

