"""
K-line data models.

Defines data models for stock K-line data at different time intervals:
quarter, month, week, day, hour, 30min, 15min, 5min, 1min.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import Field

from nq.models.base import BaseModel


# =============================================
# Base K-line Model
# =============================================


class BaseKline(BaseModel):
    """Base K-line model with common fields."""

    ts_code: str = Field(..., description="Stock code (e.g., 000001.SZ)")
    open: Optional[Decimal] = Field(None, description="Open price")
    high: Optional[Decimal] = Field(None, description="High price")
    low: Optional[Decimal] = Field(None, description="Low price")
    close: Optional[Decimal] = Field(None, description="Close price")
    volume: Optional[int] = Field(None, description="Volume (lots)")
    amount: Optional[Decimal] = Field(None, description="Amount (10K CNY)")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


# =============================================
# K-line Models by Time Interval
# =============================================


class StockKlineQuarter(BaseKline):
    """Quarterly K-line model."""

    quarter_date: datetime = Field(..., description="Quarter date (YYYY-MM-01, e.g., 2024-01-01 for Q1 2024)")
    pct_chg: Optional[Decimal] = Field(None, description="Quarter price change percentage (%)")


class StockKlineMonth(BaseKline):
    """Monthly K-line model."""

    month_date: datetime = Field(..., description="Month date (YYYY-MM-01)")
    pct_chg: Optional[Decimal] = Field(None, description="Month price change percentage (%)")


class StockKlineWeek(BaseKline):
    """Weekly K-line model."""

    week_date: datetime = Field(..., description="Week date (Monday, e.g., 2024-01-01)")
    pct_chg: Optional[Decimal] = Field(None, description="Week price change percentage (%)")


class StockKlineDay(BaseKline):
    """Daily K-line model."""

    trade_date: datetime = Field(..., description="Trading date (day precision)")
    pre_close: Optional[Decimal] = Field(None, description="Previous close price (daily specific)")
    turnover: Optional[Decimal] = Field(None, description="Turnover rate (%, daily specific)")
    pct_chg: Optional[Decimal] = Field(None, description="Daily price change percentage (%)")


class StockKlineHour(BaseKline):
    """Hourly K-line model."""

    trade_time: datetime = Field(..., description="Trading time (millisecond precision)")


class StockKline30Min(BaseKline):
    """30-minute K-line model."""

    trade_time: datetime = Field(..., description="Trading time (millisecond precision)")


class StockKline15Min(BaseKline):
    """15-minute K-line model."""

    trade_time: datetime = Field(..., description="Trading time (millisecond precision)")


class StockKline5Min(BaseKline):
    """5-minute K-line model."""

    trade_time: datetime = Field(..., description="Trading time (millisecond precision)")


class StockKline1Min(BaseKline):
    """1-minute K-line model."""

    trade_time: datetime = Field(..., description="Trading time (millisecond precision)")
    is_end: bool = Field(False, description="Whether this is the final data for this minute (prevents duplicates)")


