"""Utility modules for ATM project."""

from nq.utils.logger import setup_logger
from nq.utils.data_normalize import (
    normalize_stock_code,
    normalize_stock_codes,
    normalize_ts_code,
    convert_ts_code_to_qlib_format,
    extract_code_and_market,
    is_valid_stock_code,
    normalize_index_code,
    normalize_date,
    normalize_date_for_storage,
    normalize_industry_code,
    normalize_qlib_directory_name,
)
from nq.utils.industry import (
    load_industry_label_map,
    load_industry_map,
)
from nq.utils.model import save_embeddings

# OpenTelemetry utilities (optional)
try:
    from nq.utils.otel import (
        get_config,
        get_meter,
        get_tracer,
        initialize_otel,
        shutdown_otel,
    )
    from nq.utils.otel_helpers import (
        MetricsCounter,
        MetricsHistogram,
        record_duration,
        trace_function,
        trace_span,
    )
    from nq.utils.otel_logger import setup_logger as setup_logger_with_otel

    __all__ = [
        "setup_logger",
        "setup_logger_with_otel",
        "initialize_otel",
        "shutdown_otel",
        "get_config",
        "get_tracer",
        "get_meter",
        "trace_function",
        "trace_span",
        "record_duration",
        "MetricsCounter",
        "MetricsHistogram",
        "normalize_stock_code",
        "normalize_stock_codes",
        "normalize_ts_code",
        "convert_ts_code_to_qlib_format",
        "extract_code_and_market",
        "is_valid_stock_code",
        "normalize_index_code",
        "normalize_date",
        "normalize_date_for_storage",
        "normalize_industry_code",
        "normalize_qlib_directory_name",
        "load_industry_map",
        "load_industry_label_map",
        "save_embeddings",
    ]
except ImportError:
    __all__ = [
        "setup_logger",
        "normalize_stock_code",
        "normalize_stock_codes",
        "normalize_ts_code",
        "convert_ts_code_to_qlib_format",
        "extract_code_and_market",
        "is_valid_stock_code",
        "normalize_index_code",
        "normalize_date",
        "normalize_date_for_storage",
        "normalize_industry_code",
        "normalize_qlib_directory_name",
        "load_industry_map",
        "load_industry_label_map",
        "save_embeddings",
    ]


