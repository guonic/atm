# -*- coding: utf-8 -*-
"""
Analyze why detected candidate boxes don't form consecutive periods.
"""

import logging
from pathlib import Path

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.box_detector_accurate import AccurateBoxDetector
from nq.trading.selector.teapot.box_detector_keltner_squeeze import (
    ExpansionAnchorBoxDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_box_periods(
    df: pl.DataFrame,
    detector_name: str,
    detector,
    min_box_days_list: list = [3, 5, 10, 15, 20],
):
    """Analyze consecutive box periods for different min_box_days thresholds."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {detector_name}")
    logger.info(f"{'='*80}")
    
    # Detect boxes
    df_with_boxes = detector.detect_box(df)
    
    # Total candidate boxes
    total_candidates = df_with_boxes.filter(pl.col("is_box_candidate")).height
    logger.info(f"\nTotal candidate boxes: {total_candidates:,}")
    
    # Analyze consecutive periods for different min_box_days
    for min_box_days in min_box_days_list:
        periods = []
        
        for ts_code in df_with_boxes["ts_code"].unique():
            stock_df = (
                df_with_boxes.filter(pl.col("ts_code") == ts_code)
                .sort("trade_date")
                .with_row_index("row_idx")
            )
            
            # Create groups for consecutive box periods
            stock_df = stock_df.with_columns([
                (
                    (pl.col("is_box_candidate") != pl.col("is_box_candidate").shift(1).fill_null(False))
                    | (pl.col("row_idx") == 0)
                ).cast(pl.Int32).cum_sum().alias("group_id")
            ])
            
            # Filter only box candidate groups
            box_groups = stock_df.filter(pl.col("is_box_candidate"))
            
            if box_groups.is_empty():
                continue
            
            # Group by group_id to find consecutive periods
            for group_id in box_groups["group_id"].unique():
                group_df = box_groups.filter(pl.col("group_id") == group_id)
                
                if len(group_df) >= min_box_days:
                    periods.append({
                        "ts_code": ts_code,
                        "group_id": group_id,
                        "duration": len(group_df),
                        "start_date": group_df["trade_date"].min(),
                        "end_date": group_df["trade_date"].max(),
                    })
        
        periods_df = pl.DataFrame(periods) if periods else pl.DataFrame()
        
        if not periods_df.is_empty():
            logger.info(f"\nmin_box_days={min_box_days}:")
            logger.info(f"  Total periods: {len(periods_df):,}")
            logger.info(f"  Unique stocks: {periods_df['ts_code'].n_unique()}")
            logger.info(f"  Avg duration: {periods_df['duration'].mean():.1f} days")
            logger.info(f"  Min duration: {periods_df['duration'].min()} days")
            logger.info(f"  Max duration: {periods_df['duration'].max()} days")
            logger.info(f"  Median duration: {periods_df['duration'].median():.1f} days")
        else:
            logger.info(f"\nmin_box_days={min_box_days}: No periods found")


def main():
    """Main function."""
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
        use_cache=True,
    )
    
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    logger.info(f"Loading data: {start_date} to {end_date}")
    df = data_loader.load_daily_data(
        start_date=start_date,
        end_date=end_date,
        symbols=None,
    )
    
    if df.is_empty():
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(df):,} rows for {df['ts_code'].n_unique()} stocks")
    
    # Analyze AccurateBoxDetector
    accurate_detector = AccurateBoxDetector(
        box_window=30,
        price_tol=0.12,
        min_cross_count=3,
        smooth_window=5,
        smooth_threshold=3,
    )
    analyze_box_periods(df, "AccurateBoxDetector", accurate_detector)
    
    # Analyze ExpansionAnchorBoxDetector
    expansion_detector = ExpansionAnchorBoxDetector(
        box_window=40,
        squeeze_threshold=0.06,
        slope_threshold=0.015,
        atr_multiplier=1.5,
        lookback_window=60,
        expansion_window=20,
        stability_window=20,
        stability_threshold=12,
        smooth_window=5,
        smooth_threshold=3,
    )
    analyze_box_periods(df, "ExpansionAnchorBoxDetector", expansion_detector)
    
    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
