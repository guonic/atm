# -*- coding: utf-8 -*-
"""
Debug script to analyze why box detectors are not detecting boxes.

This script helps diagnose why certain detectors (e.g., AccurateBoxDetector, 
ExpansionAnchorBoxDetector) are not finding any boxes by:
1. Checking intermediate conditions
2. Showing statistics for each condition
3. Identifying which conditions are too strict
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.box_detector_accurate import AccurateBoxDetector
from nq.trading.selector.teapot.box_detector_keltner_squeeze import (
    ExpansionAnchorBoxDetector,
    KeltnerSqueezeDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def debug_accurate_detector(
    df: pl.DataFrame,
    box_window: int = 30,
    price_tol: float = 0.1372,  # Updated to match evaluation script
    min_cross_count: int = 3,
) -> Dict:
    """Debug AccurateBoxDetector to see why no boxes are detected."""
    logger.info("=" * 80)
    logger.info("Debugging AccurateBoxDetector")
    logger.info("=" * 80)
    
    detector = AccurateBoxDetector(
        box_window=box_window,
        price_tol=price_tol,
        min_cross_count=min_cross_count,
    )
    
    # Run detection
    df_result = detector.detect_box(df)
    
    # Check intermediate conditions
    total_rows = len(df_result)
    
    # Condition 1: Cross count
    cross_count_ok = df_result.filter(pl.col("cross_count") >= min_cross_count)
    logger.info(f"Condition 1: Cross count >= {min_cross_count}")
    logger.info(f"  Satisfied: {len(cross_count_ok)} / {total_rows} ({len(cross_count_ok)/total_rows*100:.2f}%)")
    if len(cross_count_ok) > 0:
        logger.info(f"  Cross count stats: min={cross_count_ok['cross_count'].min()}, "
                   f"max={cross_count_ok['cross_count'].max()}, "
                   f"mean={cross_count_ok['cross_count'].mean():.2f}")
    
    # Condition 2: Upper distance
    upper_dist_ok = df_result.filter(pl.col("upper_dist") < price_tol)
    logger.info(f"Condition 2: Upper distance < {price_tol}")
    logger.info(f"  Satisfied: {len(upper_dist_ok)} / {total_rows} ({len(upper_dist_ok)/total_rows*100:.2f}%)")
    if len(upper_dist_ok) > 0:
        logger.info(f"  Upper dist stats: min={upper_dist_ok['upper_dist'].min():.4f}, "
                   f"max={upper_dist_ok['upper_dist'].max():.4f}, "
                   f"mean={upper_dist_ok['upper_dist'].mean():.4f}")
    
    # Condition 3: Lower distance
    lower_dist_ok = df_result.filter(pl.col("lower_dist") < price_tol)
    logger.info(f"Condition 3: Lower distance < {price_tol}")
    logger.info(f"  Satisfied: {len(lower_dist_ok)} / {total_rows} ({len(lower_dist_ok)/total_rows*100:.2f}%)")
    if len(lower_dist_ok) > 0:
        logger.info(f"  Lower dist stats: min={lower_dist_ok['lower_dist'].min():.4f}, "
                   f"max={lower_dist_ok['lower_dist'].max():.4f}, "
                   f"mean={lower_dist_ok['lower_dist'].mean():.4f}")
    
    # Combined conditions
    all_conditions = df_result.filter(
        (pl.col("cross_count") >= min_cross_count)
        & (pl.col("upper_dist") < price_tol)
        & (pl.col("lower_dist") < price_tol)
        & pl.col("pivot_line").is_not_null()
        & (pl.col("pivot_line") > 0)
    )
    logger.info(f"All conditions satisfied (is_confirmed):")
    logger.info(f"  Count: {len(all_conditions)} / {total_rows} ({len(all_conditions)/total_rows*100:.2f}%)")
    
    # Final result (after back-propagation)
    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final boxes (after back-propagation):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")
    
    # Show distribution of intermediate values
    logger.info(f"Intermediate value distributions:")
    logger.info(f"  Cross count: min={df_result['cross_count'].min()}, "
               f"max={df_result['cross_count'].max()}, "
               f"mean={df_result['cross_count'].mean():.2f}, "
               f"median={df_result['cross_count'].median():.2f}")
    logger.info(f"  Upper dist: min={df_result['upper_dist'].min():.4f}, "
               f"max={df_result['upper_dist'].max():.4f}, "
               f"mean={df_result['upper_dist'].mean():.4f}, "
               f"median={df_result['upper_dist'].median():.4f}")
    logger.info(f"  Lower dist: min={df_result['lower_dist'].min():.4f}, "
               f"max={df_result['lower_dist'].max():.4f}, "
               f"mean={df_result['lower_dist'].mean():.4f}, "
               f"median={df_result['lower_dist'].median():.4f}")
    
    # Suggest parameter adjustments
    logger.info(f"" + "=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    
    # Check percentiles
    cross_p50 = df_result['cross_count'].quantile(0.5)
    cross_p75 = df_result['cross_count'].quantile(0.75)
    upper_p50 = df_result['upper_dist'].quantile(0.5)
    upper_p75 = df_result['upper_dist'].quantile(0.75)
    lower_p50 = df_result['lower_dist'].quantile(0.5)
    lower_p75 = df_result['lower_dist'].quantile(0.75)
    
    logger.info(f"If you want to capture ~50% of data:")
    logger.info(f"  min_cross_count: {int(cross_p50)} (current: {min_cross_count})")
    logger.info(f"  price_tol: {upper_p50:.4f} (current: {price_tol})")
    
    logger.info(f"If you want to capture ~75% of data:")
    logger.info(f"  min_cross_count: {int(cross_p75)} (current: {min_cross_count})")
    logger.info(f"  price_tol: {max(upper_p75, lower_p75):.4f} (current: {price_tol})")
    
    return {
        "total_rows": total_rows,
        "cross_count_ok": len(cross_count_ok),
        "upper_dist_ok": len(upper_dist_ok),
        "lower_dist_ok": len(lower_dist_ok),
        "all_conditions_ok": len(all_conditions),
        "final_boxes": len(final_boxes),
    }


def debug_expansion_anchor_detector(
    df: pl.DataFrame,
    box_window: int = 40,
    squeeze_threshold: float = 0.06,
    slope_threshold: float = 0.015,
    stability_threshold: int = 15,
    stability_window: int = 20,
) -> Dict:
    """Debug ExpansionAnchorBoxDetector to see why no boxes are detected."""
    logger.info("=" * 80)
    logger.info("Debugging ExpansionAnchorBoxDetector")
    logger.info("=" * 80)
    
    detector = ExpansionAnchorBoxDetector(
        box_window=box_window,
        squeeze_threshold=squeeze_threshold,
        slope_threshold=slope_threshold,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
    )
    
    # Run detection
    df_result = detector.detect_box(df)
    
    total_rows = len(df_result)
    
    # Condition 1: Squeeze ratio
    squeeze_ok = df_result.filter(pl.col("squeeze_ratio") < squeeze_threshold)
    logger.info(f"Condition 1: Squeeze ratio < {squeeze_threshold}")
    logger.info(f"  Satisfied: {len(squeeze_ok)} / {total_rows} ({len(squeeze_ok)/total_rows*100:.2f}%)")
    if len(squeeze_ok) > 0:
        logger.info(f"  Squeeze ratio stats: min={squeeze_ok['squeeze_ratio'].min():.4f}, "
                   f"max={squeeze_ok['squeeze_ratio'].max():.4f}, "
                   f"mean={squeeze_ok['squeeze_ratio'].mean():.4f}")
    
    # Condition 2: Slope
    slope_ok = df_result.filter(pl.col("mid_slope") < slope_threshold)
    logger.info(f"Condition 2: Mid slope < {slope_threshold}")
    logger.info(f"  Satisfied: {len(slope_ok)} / {total_rows} ({len(slope_ok)/total_rows*100:.2f}%)")
    if len(slope_ok) > 0:
        logger.info(f"  Mid slope stats: min={slope_ok['mid_slope'].min():.4f}, "
                   f"max={slope_ok['mid_slope'].max():.4f}, "
                   f"mean={slope_ok['mid_slope'].mean():.4f}")
    
    # Condition 3: Price in channel
    price_in_channel = df_result.filter(
        (pl.col("close") < pl.col("box_h")) & (pl.col("close") > pl.col("box_l"))
    )
    logger.info(f"Condition 3: Price in channel")
    logger.info(f"  Satisfied: {len(price_in_channel)} / {total_rows} ({len(price_in_channel)/total_rows*100:.2f}%)")
    
    # Condition 4: Stability count
    stability_ok = df_result.filter(pl.col("stability_count") >= stability_threshold)
    logger.info(f"Condition 4: Stability count >= {stability_threshold}")
    logger.info(f"  Satisfied: {len(stability_ok)} / {total_rows} ({len(stability_ok)/total_rows*100:.2f}%)")
    if len(stability_ok) > 0:
        logger.info(f"  Stability count stats: min={stability_ok['stability_count'].min()}, "
                   f"max={stability_ok['stability_count'].max()}, "
                   f"mean={stability_ok['stability_count'].mean():.2f}")
    
    # Combined conditions
    all_conditions = df_result.filter(
        (pl.col("squeeze_ratio") < squeeze_threshold)
        & (pl.col("mid_slope") < slope_threshold)
        & (pl.col("close") < pl.col("box_h"))
        & (pl.col("close") > pl.col("box_l"))
        & (pl.col("stability_count") >= stability_threshold)
        & pl.col("box_h").is_not_null()
        & pl.col("box_l").is_not_null()
        & (pl.col("box_h") > pl.col("box_l"))
    )
    logger.info(f"All conditions satisfied (is_squeeze_detected):")
    logger.info(f"  Count: {len(all_conditions)} / {total_rows} ({len(all_conditions)/total_rows*100:.2f}%)")
    
    # Final result (after back-propagation)
    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final boxes (after back-propagation):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")
    
    # Show distribution of intermediate values
    logger.info(f"Intermediate value distributions:")
    logger.info(f"  Squeeze ratio: min={df_result['squeeze_ratio'].min():.4f}, "
               f"max={df_result['squeeze_ratio'].max():.4f}, "
               f"mean={df_result['squeeze_ratio'].mean():.4f}, "
               f"median={df_result['squeeze_ratio'].median():.4f}")
    logger.info(f"  Mid slope: min={df_result['mid_slope'].min():.4f}, "
               f"max={df_result['mid_slope'].max():.4f}, "
               f"mean={df_result['mid_slope'].mean():.4f}, "
               f"median={df_result['mid_slope'].median():.4f}")
    logger.info(f"  Stability count: min={df_result['stability_count'].min()}, "
               f"max={df_result['stability_count'].max()}, "
               f"mean={df_result['stability_count'].mean():.2f}, "
               f"median={df_result['stability_count'].median():.2f}")
    
    # Suggest parameter adjustments
    logger.info(f"" + "=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    
    squeeze_p50 = df_result['squeeze_ratio'].quantile(0.5)
    squeeze_p75 = df_result['squeeze_ratio'].quantile(0.75)
    slope_p50 = df_result['mid_slope'].quantile(0.5)
    slope_p75 = df_result['mid_slope'].quantile(0.75)
    stability_p50 = df_result['stability_count'].quantile(0.5)
    stability_p75 = df_result['stability_count'].quantile(0.75)
    
    logger.info(f"If you want to capture ~50% of data:")
    logger.info(f"  squeeze_threshold: {squeeze_p50:.4f} (current: {squeeze_threshold})")
    logger.info(f"  slope_threshold: {slope_p50:.4f} (current: {slope_threshold})")
    logger.info(f"  stability_threshold: {int(stability_p50)} (current: {stability_threshold})")
    
    logger.info(f"If you want to capture ~75% of data:")
    logger.info(f"  squeeze_threshold: {squeeze_p75:.4f} (current: {squeeze_threshold})")
    logger.info(f"  slope_threshold: {slope_p75:.4f} (current: {slope_threshold})")
    logger.info(f"  stability_threshold: {int(stability_p75)} (current: {stability_threshold})")
    
    return {
        "total_rows": total_rows,
        "squeeze_ok": len(squeeze_ok),
        "slope_ok": len(slope_ok),
        "price_in_channel": len(price_in_channel),
        "stability_ok": len(stability_ok),
        "all_conditions_ok": len(all_conditions),
        "final_boxes": len(final_boxes),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Debug box detectors to understand why no boxes are detected"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Optional list of stock codes to debug (default: all stocks)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["accurate", "expansion_anchor", "both"],
        default="both",
        help="Which detector to debug (default: both)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data if available",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        db_config = DatabaseConfig()
    
    # Load data
    data_loader = TeapotDataLoader(
        db_config=db_config,
        schema="quant",
        use_cache=args.use_cache,
    )
    
    logger.info(f"Loading data: {args.start_date} to {args.end_date}")
    df = data_loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
    )
    
    if df.is_empty():
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(df)} rows for {df['ts_code'].n_unique()} stocks")
    
    # Debug detectors
    if args.detector in ["accurate", "both"]:
        debug_accurate_detector(df)
    
    if args.detector in ["expansion_anchor", "both"]:
        debug_expansion_anchor_detector(df)
    
    logger.info("\n" + "=" * 80)
    logger.info("Debug complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
