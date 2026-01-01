"""
Industry utility functions for ATM project.

This module provides utility functions for loading and processing industry data,
including industry mappings and labels.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from nq.config import DatabaseConfig
from nq.repo.stock_repo import StockIndustryMemberRepo
from nq.utils.data_normalize import normalize_stock_code

logger = logging.getLogger(__name__)


def load_industry_map(
    db_config: DatabaseConfig, target_date: Optional[datetime] = None, schema: str = "quant"
) -> Dict[str, str]:
    """
    Load industry mapping from database.

    This function loads the mapping between stock codes and industry codes (L3 level)
    from the stock_industry_member table. The mapping is normalized to Qlib format and includes
    multiple format variants for robust matching.

    Args:
        db_config: Database configuration.
        target_date: Target date for industry membership (default: current date).
        schema: Database schema name.

    Returns:
        Dictionary mapping stock codes (in Qlib format) to industry codes (L3 level).
        Format: {stock_code: l3_code}
        The mapping includes multiple format variants (uppercase, lowercase, original)
        for robust matching with different code formats.

    Examples:
        >>> from nq.config import load_config
        >>> config = load_config("config/config.yaml")
        >>> industry_map = load_industry_map(config.database)
        >>> print(industry_map.get("000001.SZ"))
        "801010"  # L3 industry code for this stock
    """
    repo = StockIndustryMemberRepo(db_config, schema=schema)

    if target_date is None:
        target_date = datetime.now()

    # Use repo method to get industry mapping data
    rows = repo.get_industry_mapping(target_date=target_date.date())

    # Create stock_code -> l3_code mapping
    # Store multiple format variants for robust matching (Qlib may use lowercase)
    industry_map = {}
    industry_codes = set()
    for ts_code, l3_code in rows:
        normalized_code = normalize_stock_code(ts_code)
        industry_codes.add(l3_code)
        # Store standard format (uppercase)
        industry_map[normalized_code] = l3_code
        # Also store lowercase variant for Qlib compatibility
        industry_map[normalized_code.lower()] = l3_code

    logger.info(f"Loaded industry mapping: {len(industry_map)} stocks, {len(industry_codes)} industries")
    return industry_map


def load_industry_label_map(
    db_config: DatabaseConfig, target_date: Optional[datetime] = None, schema: str = "quant"
) -> Dict[str, str]:
    """
    Load industry label mapping from database.

    This function loads the mapping between stock codes and industry names (L3 level)
    from the stock_industry_member table. The mapping is normalized to Qlib format
    and includes multiple format variants for robust matching.

    Args:
        db_config: Database configuration.
        target_date: Target date for industry membership (default: current date).
        schema: Database schema name.

    Returns:
        Dictionary mapping stock codes to industry names.
        Format: {stock_code: "行业名称"}
        The mapping includes multiple format variants (uppercase, lowercase, original)
        for robust matching with different code formats.

    Examples:
        >>> from nq.config import load_config
        >>> config = load_config("config/config.yaml")
        >>> label_map = load_industry_label_map(config.database)
        >>> print(label_map.get("000001.SZ"))
        "银行"  # Industry name for this stock
    """
    repo = StockIndustryMemberRepo(db_config, schema=schema)

    if target_date is None:
        target_date = datetime.now()

    # Use repo method to get industry label mapping data
    try:
        rows = repo.get_industry_label_mapping(target_date=target_date.date())

        # Convert to standard format (already normalized)
        label_map = {}
        for ts_code, l3_name in rows:
            normalized_code = normalize_stock_code(ts_code)
            industry_name = l3_name if l3_name else "Unknown"
            # Store normalized code (standard format is uppercase)
            label_map[normalized_code] = industry_name
            # Also store lowercase variant for Qlib compatibility (Qlib uses lowercase in some contexts)
            if normalized_code:
                label_map[normalized_code.lower()] = industry_name
                # Also store original format variants for robust matching
                label_map[ts_code] = industry_name
                label_map[ts_code.upper()] = industry_name
                label_map[ts_code.lower()] = industry_name

        logger.info(f"Loaded industry labels: {len(rows)} stocks (with case variants: {len(label_map)} entries)")
        if len(label_map) == 0:
            logger.warning("No industry labels loaded! Check database connection and data availability.")
        return label_map
    except Exception as e:
        logger.warning(f"Failed to load industry labels: {e}")
        return {}

