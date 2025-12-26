"""Data repository module for ATM project."""

from atm.repo.base import BaseRepo, ConnectionError, RepoError, SaveError
from atm.repo.database_repo import DatabaseRepo
from atm.repo.backtest_repo import BacktestRepo
from atm.repo.kline_repo import (
    StockKline15MinRepo,
    StockKline1MinRepo,
    StockKline30MinRepo,
    StockKline5MinRepo,
    StockKlineDayRepo,
    StockKlineHourRepo,
    StockKlineMonthRepo,
    StockKlineQuarterRepo,
    StockKlineWeekRepo,
)
from atm.repo.state_repo import (
    BaseStateRepo,
    DatabaseStateRepo,
    FileStateRepo,
    IngestionState,
)
from atm.repo.task_lock import (
    BaseTaskLock,
    DatabaseTaskLock,
    FileTaskLock,
    TaskLockError,
)
from atm.repo.stock_repo import (
    StockBasicRepo,
    StockClassifyRepo,
    StockFinanceBasicRepo,
    StockIndustryClassifyRepo,
    StockIndustryMemberRepo,
    StockKlineSyncStateRepo,
    StockPremarketRepo,
    StockQuoteSnapshotRepo,
    StockTradeRuleRepo,
)
from atm.repo.trading_calendar_repo import TradingCalendarRepo

__all__ = [
    # Base classes
    "BaseRepo",
    "DatabaseRepo",
    "RepoError",
    "ConnectionError",
    "SaveError",
    # Backtest
    "BacktestRepo",
    # Stock repositories
    "StockBasicRepo",
    "StockClassifyRepo",
    "StockTradeRuleRepo",
    "StockFinanceBasicRepo",
    "StockIndustryClassifyRepo",
    "StockIndustryMemberRepo",
    "StockPremarketRepo",
    "StockQuoteSnapshotRepo",
    "StockKlineSyncStateRepo",
    # K-line repositories
    "StockKlineQuarterRepo",
    "StockKlineMonthRepo",
    "StockKlineWeekRepo",
    "StockKlineDayRepo",
    "StockKlineHourRepo",
    "StockKline30MinRepo",
    "StockKline15MinRepo",
    "StockKline5MinRepo",
    "StockKline1MinRepo",
    # State repositories
    "BaseStateRepo",
    "FileStateRepo",
    "DatabaseStateRepo",
    "IngestionState",
    # Task locks
    "BaseTaskLock",
    "FileTaskLock",
    "DatabaseTaskLock",
    "TaskLockError",
    # Trading calendar
    "TradingCalendarRepo",
]

