"""
Composite stock selector.

Combines multiple selectors using logical operations (AND, OR, NOT).
"""

import logging
from datetime import datetime
from typing import List, Optional

from .base import BaseSelector, SelectionResult

logger = logging.getLogger(__name__)


class CompositeSelector(BaseSelector):
    """
    Composite stock selector.

    Combines multiple selectors using logical operations:
    - AND: Stocks must pass all selectors
    - OR: Stocks must pass at least one selector
    - NOT: Stocks must not pass the selector
    """

    def __init__(
        self,
        selectors: List[BaseSelector],
        operation: str = "AND",
        db_config=None,
        schema: str = "quant",
    ):
        """
        Initialize composite selector.

        Args:
            selectors: List of selectors to combine.
            operation: Logical operation ("AND", "OR", "NOT"). Default: "AND".
            db_config: Database configuration (optional, can be None if all selectors have it).
            schema: Database schema name.
        """
        # Use db_config from first selector if not provided
        if db_config is None and selectors:
            db_config = selectors[0].db_config
            schema = selectors[0].schema

        super().__init__(db_config, schema)
        self.selectors = selectors
        self.operation = operation.upper()

        if self.operation not in ["AND", "OR", "NOT"]:
            raise ValueError(f"Invalid operation: {operation}. Must be 'AND', 'OR', or 'NOT'")

        if self.operation == "NOT" and len(selectors) != 1:
            raise ValueError("NOT operation requires exactly one selector")

    def select(
        self,
        exchange: Optional[str] = None,
        market: Optional[str] = None,
        min_list_date: Optional[datetime] = None,
        max_list_date: Optional[datetime] = None,
        **kwargs,
    ) -> SelectionResult:
        """
        Select stocks by combining multiple selectors.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE). If None, select from all exchanges.
            market: Market type (沪A/深A/北交所). If None, select from all markets.
            min_list_date: Minimum listing date.
            max_list_date: Maximum listing date.
            **kwargs: Additional selection criteria passed to all selectors.

        Returns:
            SelectionResult containing selected stock codes.
        """
        if not self.selectors:
            raise ValueError("No selectors provided")

        # Run all selectors
        results = []
        for selector in self.selectors:
            result = selector.select(
                exchange=exchange,
                market=market,
                min_list_date=min_list_date,
                max_list_date=max_list_date,
                **kwargs,
            )
            results.append(result)

        # Combine results based on operation
        if self.operation == "AND":
            # Stocks must be in all results
            if not results:
                selected_stocks = []
            else:
                selected_stocks = set(results[0].selected_stocks)
                for result in results[1:]:
                    selected_stocks &= set(result.selected_stocks)
                selected_stocks = list(selected_stocks)

        elif self.operation == "OR":
            # Stocks must be in at least one result
            selected_stocks = set()
            for result in results:
                selected_stocks |= set(result.selected_stocks)
            selected_stocks = list(selected_stocks)

        elif self.operation == "NOT":
            # Stocks must not be in the result
            # Get all candidate stocks first
            from atm.repo.stock_repo import StockBasicRepo

            stock_repo = StockBasicRepo(self.db_config, self.schema)
            all_stocks = stock_repo.get_by_exchange(exchange) if exchange else stock_repo.get_all()

            if market:
                all_stocks = [s for s in all_stocks if s.market == market]
            if min_list_date:
                all_stocks = [s for s in all_stocks if s.list_date >= min_list_date.date()]
            if max_list_date:
                all_stocks = [s for s in all_stocks if s.list_date <= max_list_date.date()]

            all_ts_codes = {s.ts_code for s in all_stocks if s.is_listed}
            excluded_ts_codes = set(results[0].selected_stocks)
            selected_stocks = list(all_ts_codes - excluded_ts_codes)

        logger.info(
            f"Composite selector ({self.operation}) selected {len(selected_stocks)} stocks "
            f"from {len(results)} selector(s)"
        )

        # Combine selection criteria
        combined_criteria = {
            "operation": self.operation,
            "selectors": [s.get_info() for s in self.selectors],
        }
        for i, result in enumerate(results):
            combined_criteria[f"selector_{i}_criteria"] = result.selection_criteria

        return SelectionResult(
            selected_stocks=selected_stocks,
            selection_date=results[0].selection_date if results else datetime.now(),
            total_candidates=results[0].total_candidates if results else 0,
            selection_criteria=combined_criteria,
            metadata={
                "selector_type": "composite",
                "num_selectors": len(self.selectors),
                "operation": self.operation,
            },
        )

