"""Data source module for ATM project."""

from atm.data.source.base import BaseSource
from atm.data.source.http_source import HttpSource
from atm.data.source.tushare_source import TushareSource, TushareSourceConfig

try:
    from atm.data.source.akshare_source import AkshareSource, AkshareSourceConfig

    __all__ = [
        "BaseSource",
        "HttpSource",
        "TushareSource",
        "TushareSourceConfig",
        "AkshareSource",
        "AkshareSourceConfig",
    ]
except ImportError:
    # AkShare not installed, exclude from exports
    __all__ = [
        "BaseSource",
        "HttpSource",
        "TushareSource",
        "TushareSourceConfig",
    ]

