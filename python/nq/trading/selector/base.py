"""
Base selector interface.

Provides abstract base class for all stock selectors.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SelectionResult(BaseModel):
    """Selection result model."""

    selected_stocks: List[str] = Field(..., description="List of selected stock codes (ts_code)")
    selection_date: datetime = Field(..., description="Date when selection was performed")
    total_candidates: int = Field(..., description="Total number of candidate stocks")
    selection_criteria: dict = Field(default_factory=dict, description="Selection criteria used")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class BaseSelector(ABC):
    """
    Abstract base class for all stock selectors.

    All selectors should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        db_config,
        schema: str = "quant",
    ):
        """
        Initialize selector.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
        """
        self.db_config = db_config
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def select(
        self,
        exchange: Optional[str] = None,
        market: Optional[str] = None,
        min_list_date: Optional[datetime] = None,
        max_list_date: Optional[datetime] = None,
        **kwargs,
    ) -> SelectionResult:
        """
        Select stocks based on criteria.

        Args:
            exchange: Exchange code (SSE/SZSE/BSE). If None, select from all exchanges.
            market: Market type (沪A/深A/北交所). If None, select from all markets.
            min_list_date: Minimum listing date. If None, no minimum date filter.
            max_list_date: Maximum listing date. If None, no maximum date filter.
            **kwargs: Additional selection criteria.

        Returns:
            SelectionResult containing selected stock codes and metadata.
        """
        pass

    def get_info(self) -> dict:
        """
        Get selector information.

        Returns:
            Dictionary containing selector information.
        """
        return {
            "class": self.__class__.__name__,
            "schema": self.schema,
        }

