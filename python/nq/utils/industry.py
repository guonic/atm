"""
Industry utility functions for ATM project.

This module provides utility functions for loading and processing industry data,
including industry mappings and labels.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import text

from nq.config import DatabaseConfig
from nq.repo.stock_repo import StockIndustryMemberRepo
from nq.utils.data_normalize import normalize_stock_code

logger = logging.getLogger(__name__)


def load_industry_map(
    db_config: DatabaseConfig, target_date: Optional[datetime] = None, schema: str = "quant"
) -> Dict[str, int]:
    """
    Load industry mapping from database.

    This function loads the mapping between stock codes and industry IDs from the
    stock_industry_member table. The mapping is normalized to Qlib format and includes
    multiple format variants for robust matching.

    Args:
        db_config: Database configuration.
        target_date: Target date for industry membership (default: current date).
        schema: Database schema name.

    Returns:
        Dictionary mapping stock codes (in Qlib format) to industry IDs.
        Format: {stock_code: industry_id}
        The mapping includes multiple format variants (uppercase, lowercase, original)
        for robust matching with different code formats.

    Examples:
        >>> from nq.config import load_config
        >>> config = load_config("config/config.yaml")
        >>> industry_map = load_industry_map(config.database)
        >>> print(industry_map.get("000001.SZ"))
        42  # Industry ID for this stock
    """
    repo = StockIndustryMemberRepo(db_config, schema=schema)

    if target_date is None:
        target_date = datetime.now()

    # Get all current industry members
    engine = repo._get_engine()
    table_name = repo._get_full_table_name()

    sql = f"""
    SELECT DISTINCT ts_code, l3_code
    FROM {table_name}
    WHERE (out_date IS NULL OR out_date > :target_date)
      AND in_date <= :target_date
    ORDER BY ts_code
    """

    with engine.connect() as conn:
        result = conn.execute(
            text(sql),
            {"target_date": target_date.date()},
        )
        rows = result.fetchall()

    # Convert to Qlib format and create mapping
    industry_codes = sorted(set(row[1] for row in rows))
    industry_id_map = {code: idx for idx, code in enumerate(industry_codes)}

    # Create stock_code -> industry_id mapping
    # Store multiple format variants for robust matching (Qlib may use lowercase)
    industry_map = {}
    for ts_code, l3_code in rows:
        normalized_code = normalize_stock_code(ts_code)
        industry_id = industry_id_map[l3_code]
        # Store standard format (uppercase)
        if normalized_code:
            industry_map[normalized_code] = industry_id
            # Also store lowercase variant for Qlib compatibility
            industry_map[normalized_code.lower()] = industry_id
        # Also store original format variants for robust matching
        industry_map[ts_code] = industry_id
        industry_map[ts_code.upper()] = industry_id
        industry_map[ts_code.lower()] = industry_id

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

    engine = repo._get_engine()
    table_name = repo._get_full_table_name()

    # Query stock codes with L3 industry names
    # Note: stock_industry_member table already has l3_name field, no need to JOIN
    # Get the most recent record for each stock that is valid at target_date
    sql = f"""
    SELECT DISTINCT ON (sim.ts_code)
        sim.ts_code,
        sim.l3_name
    FROM {table_name} sim
    WHERE sim.in_date <= :target_date
      AND (sim.out_date IS NULL OR sim.out_date > :target_date)
    ORDER BY sim.ts_code, sim.in_date DESC
    """

    # If no results, try without out_date filter (use latest in_date for each stock)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(sql),
                {"target_date": target_date.date()},
            )
            rows = result.fetchall()

        # If no results with date filter, try getting latest record for each stock
        if not rows:
            logger.warning(f"No industry labels found with date filter, trying latest records...")
            sql_latest = f"""
            SELECT DISTINCT ON (sim.ts_code)
                sim.ts_code,
                sim.l3_name
            FROM {table_name} sim
            ORDER BY sim.ts_code, sim.in_date DESC
            """
            with engine.connect() as conn:
                result = conn.execute(text(sql_latest))
                rows = result.fetchall()
            if rows:
                logger.info(f"Found {len(rows)} industry labels using latest records")

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
            logger.warning(f"Query was: SELECT DISTINCT ts_code, l3_name FROM {table_name} WHERE ...")
        return label_map
    except Exception as e:
        logger.warning(f"Failed to load industry labels: {e}")
        return {}

