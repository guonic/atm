"""
Trading calendar models.

Defines data models for trading calendar information.
"""

from datetime import date
from typing import Optional

from pydantic import Field

from nq.models.base import BaseModel


class TradingCalendar(BaseModel):
    """Trading calendar model."""

    exchange: str = Field(..., description="Exchange code (SSE=上交所, SZSE=深交所, BSE=北交所)")
    cal_date: date = Field(..., description="Calendar date")
    is_open: bool = Field(..., description="Whether trading day (True=交易, False=休市)")
    pretrade_date: Optional[date] = Field(None, description="Previous trading day")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            date: lambda v: v.isoformat() if v else None,
        }

