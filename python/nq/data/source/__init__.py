"""Data source module for ATM project."""

from nq.data.source.base import BaseSource
from nq.data.source.http_source import HttpSource
from nq.data.source.tushare_source import TushareSource, TushareSourceConfig

try:
    from nq.data.source.akshare_source import AkshareSource, AkshareSourceConfig

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

