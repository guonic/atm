"""
Stock information models.

Defines data models for stock basic information, classification, trade rules, finance, and quotes.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import Field

from atm.models.base import BaseModel


# =============================================
# Stock Basic Information Models
# =============================================


class StockBasic(BaseModel):
    """Stock basic information model."""

    ts_code: str = Field(..., description="Stock code (e.g., 000001.SZ, globally unique)")
    symbol: str = Field(..., description="Stock symbol (e.g., 平安银行)")
    full_name: str = Field(..., description="Full company name")
    exchange: str = Field(..., description="Exchange (SH/SE/SZ/创业板/科创板)")
    market: str = Field(..., description="Market type (沪A/深A/北交所/港股/美股)")
    list_date: date = Field(..., description="Listing date")
    delist_date: Optional[date] = Field(None, description="Delisting date (NULL if not delisted)")
    is_listed: bool = Field(True, description="Whether listed (TRUE=listed, FALSE=delisted)")
    currency: str = Field("CNY", description="Trading currency (CNY/USD/HKD)")
    create_time: Optional[datetime] = Field(None, description="Data insertion time")
    update_time: Optional[datetime] = Field(None, description="Last update time")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            date: lambda v: v.isoformat() if v else None,
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class StockClassify(BaseModel):
    """Stock classification model."""

    id: Optional[int] = Field(None, description="Primary key")
    ts_code: str = Field(..., description="Stock code (references stock_basic)")
    classify_type: str = Field(..., description="Classification type (industry/concept/board)")
    classify_value: str = Field(..., description="Classification value (e.g., 银行/人工智能/创业板)")
    is_main: bool = Field(False, description="Whether main classification (e.g., main industry)")
    update_time: Optional[datetime] = Field(None, description="Classification update time")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


class StockTradeRule(BaseModel):
    """Stock trading rule model."""

    ts_code: str = Field(..., description="Stock code (references stock_basic)")
    price_tick: Decimal = Field(..., description="Minimum price change unit (e.g., 0.01)")
    lot_size: int = Field(100, description="Trading unit (shares per lot, A-share default 100)")
    max_limit: Decimal = Field(10.00, description="Price limit percentage (e.g., 10=±10%)")
    is_tplus1: bool = Field(True, description="Whether T+1 trading (A-share default TRUE)")
    is_st: bool = Field(False, description="Whether ST stock (5% price limit)")
    is_suspend: bool = Field(False, description="Whether suspended")
    suspend_start: Optional[date] = Field(None, description="Suspension start date")
    suspend_end: Optional[date] = Field(None, description="Suspension end date")
    update_time: Optional[datetime] = Field(None, description="Update time")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            date: lambda v: v.isoformat() if v else None,
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class StockFinanceBasic(BaseModel):
    """Stock finance basic information model."""

    id: Optional[int] = Field(None, description="Primary key")
    ts_code: str = Field(..., description="Stock code (references stock_basic)")
    report_date: date = Field(..., description="Report date (e.g., 2024-03-31)")
    total_share: Optional[Decimal] = Field(None, description="Total shares (10K shares)")
    float_share: Optional[Decimal] = Field(None, description="Float shares (10K shares)")
    total_mv: Optional[Decimal] = Field(None, description="Total market value (10K CNY)")
    float_mv: Optional[Decimal] = Field(None, description="Float market value (10K CNY)")
    pb: Optional[Decimal] = Field(None, description="Price-to-book ratio")
    pe_ttm: Optional[Decimal] = Field(None, description="Price-to-earnings ratio (TTM)")
    update_time: Optional[datetime] = Field(None, description="Update time")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            date: lambda v: v.isoformat() if v else None,
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


class StockQuoteSnapshot(BaseModel):
    """Stock quote snapshot model."""

    ts_code: str = Field(..., description="Stock code (references stock_basic)")
    last_price: Optional[Decimal] = Field(None, description="Latest price")
    open_price: Optional[Decimal] = Field(None, description="Today's open price")
    high_price: Optional[Decimal] = Field(None, description="Today's high price")
    low_price: Optional[Decimal] = Field(None, description="Today's low price")
    pre_close: Optional[Decimal] = Field(None, description="Previous close price")
    pct_chg: Optional[Decimal] = Field(None, description="Today's price change percentage (%)")
    volume: Optional[int] = Field(None, description="Today's volume (lots)")
    amount: Optional[Decimal] = Field(None, description="Today's amount (10K CNY)")
    update_time: Optional[datetime] = Field(None, description="Snapshot update time (millisecond precision)")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None,
        }


