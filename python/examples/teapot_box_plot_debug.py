# -*- coding: utf-8 -*-
"""
teapot_box_plot_debug.py

Debug script for box chart generation issues.
"""

import logging
from datetime import datetime, timedelta

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.box_detector import MeanReversionBoxDetectorV2
from nq.trading.selector.teapot.box_visualizer import BoxVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def debug_box_extraction(
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    min_box_days: int = 5,
    use_cache: bool = False,
):
    """Debug box extraction and visualization."""
    
    # Load config
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        db_config = DatabaseConfig()

    # Initialize data loader
    data_loader = TeapotDataLoader(
        db_config=db_config,
        schema="quant",
        use_cache=use_cache,
    )

    # Load data
    logger.info(f"Loading data: {start_date} to {end_date}")
    df = data_loader.load_daily_data(
        start_date=start_date,
        end_date=end_date,
        symbols=None,
    )

    if df.is_empty():
        logger.error("No data loaded!")
        return

    logger.info(f"✓ Loaded {len(df)} rows for {df['ts_code'].n_unique()} stocks")

    # Test detector
    logger.info("Testing box detector...")
    detector = MeanReversionBoxDetectorV2(
        box_window=40,
        max_total_return=None,
        max_relative_box_height=0.12,
        volatility_ratio=0.25,
    )

    # Detect boxes
    df_with_boxes = detector.detect_box(df)
    
    # Count box candidates
    box_candidates = df_with_boxes.filter(pl.col("is_box_candidate"))
    logger.info(f"✓ Found {len(box_candidates)} box candidate rows")
    
    if box_candidates.is_empty():
        logger.error("No box candidates detected!")
        return

    # Check per stock
    stocks_with_boxes = box_candidates["ts_code"].n_unique()
    logger.info(f"✓ Stocks with boxes: {stocks_with_boxes}")

    # Test extraction with different min_box_days
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from teapot_box_detector_evaluation import BoxDetectorEvaluator
    
    evaluator = BoxDetectorEvaluator(
        db_config=db_config,
        schema="quant",
        use_cache=use_cache,
    )

    for test_min_days in [1, 3, 5, 10]:
        logger.info(f"\nTesting with min_box_days={test_min_days}...")
        boxes_df = evaluator.extract_boxes_with_dates(
            df=df,
            detector=detector,
            detector_name="mean_reversion_v2",
            min_box_days=test_min_days,
        )
        
        if boxes_df.is_empty():
            logger.warning(f"  ✗ No boxes with min_box_days={test_min_days}")
        else:
            logger.info(f"  ✓ Found {len(boxes_df)} boxes")
            if "box_duration_days" in boxes_df.columns:
                avg_duration = boxes_df["box_duration_days"].mean()
                min_duration = boxes_df["box_duration_days"].min()
                max_duration = boxes_df["box_duration_days"].max()
                logger.info(f"    Duration: min={min_duration}, avg={avg_duration:.1f}, max={max_duration}")

    # Test visualization
    logger.info("\nTesting visualization...")
    try:
        import plotly.graph_objects as go
        logger.info("  ✓ Plotly is available")
    except ImportError:
        logger.error("  ✗ Plotly is not installed")
        logger.error("  Install with: pip install plotly")
        return

    # Use boxes with min_box_days=5
    boxes_df = evaluator.extract_boxes_with_dates(
        df=df,
        detector=detector,
        detector_name="mean_reversion_v2",
        min_box_days=5,
    )

    if boxes_df.is_empty():
        logger.warning("No boxes to visualize (try reducing min_box_days)")
        return

    logger.info(f"Generating charts for {len(boxes_df)} boxes...")
    visualizer = BoxVisualizer(output_dir="outputs/teapot/debug_charts")
    
    chart_results = visualizer.batch_plot_boxes(
        boxes_df=boxes_df.head(5),  # Test with first 5 boxes
        kline_df=df,
        context_days=30,
        max_boxes=5,
    )

    success_count = chart_results.filter(pl.col("status") == "success").height
    logger.info(f"✓ Generated {success_count}/5 test charts")
    
    if success_count > 0:
        logger.info(f"Charts saved to: outputs/teapot/debug_charts")
    else:
        logger.error("No charts were generated. Check errors above.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug box chart generation")
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--end-date", type=str, default="2024-01-01")
    parser.add_argument("--min-box-days", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true")
    
    args = parser.parse_args()
    
    debug_box_extraction(
        start_date=args.start_date,
        end_date=args.end_date,
        min_box_days=args.min_box_days,
        use_cache=args.use_cache,
    )
