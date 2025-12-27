#!/usr/bin/env python3
"""
Training script for Structure Expert GNN model.

This script integrates the Structure Expert GNN model with the system's data:
- Loads industry mapping from database
- Uses Qlib DatasetH to load features
- Implements rolling training over date range
- Saves model checkpoints

Usage:
    python python/tools/qlib/train/train_structure_expert.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --qlib-dir ~/.qlib/qlib_data/cn_data
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import qlib
import torch
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.repo.stock_repo import StockIndustryMemberRepo

# Import structure expert model
# Add tools/qlib/train to path for import
tools_train_dir = Path(__file__).parent
if str(tools_train_dir) not in sys.path:
    sys.path.insert(0, str(tools_train_dir))

from structure_expert import (
    GraphDataBuilder,
    StructureExpertGNN,
    StructureTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_industry_map(
    db_config: DatabaseConfig, target_date: Optional[datetime] = None, schema: str = "quant"
) -> Dict[str, int]:
    """
    Load industry mapping from database.

    Args:
        db_config: Database configuration.
        target_date: Target date for industry membership (default: current date).
        schema: Database schema name.

    Returns:
        Dictionary mapping stock codes (in Qlib format) to industry IDs.
        Format: {stock_code: industry_id}
    """
    repo = StockIndustryMemberRepo(db_config, schema=schema)

    if target_date is None:
        target_date = datetime.now()

    # Get all current industry members
    # Query all stocks and their L3 industry codes
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
    # Map L3 codes to integer IDs
    industry_codes = sorted(set(row[1] for row in rows))
    industry_id_map = {code: idx for idx, code in enumerate(industry_codes)}

    # Create stock_code -> industry_id mapping
    industry_map = {}
    for ts_code, l3_code in rows:
        # Convert ts_code to Qlib format (e.g., 000001.SZ -> 000001.SZ)
        qlib_code = convert_ts_code_to_qlib_format(ts_code)
        industry_map[qlib_code] = industry_id_map[l3_code]

    logger.info(f"Loaded industry mapping: {len(industry_map)} stocks, {len(industry_codes)} industries")
    return industry_map


def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    """
    Convert ts_code to Qlib format.

    Args:
        ts_code: Stock code in format like '000001.SZ' or '600000.SH'.

    Returns:
        Qlib format code (same format for most cases).
    """
    # Qlib uses same format as Tushare for most cases
    return ts_code


def get_feature_names() -> List[str]:
    """
    Get feature names for Alpha158.

    Returns:
        List of feature names.
    """
    # Alpha158 has 158 features
    # For simplicity, we'll use the handler to get feature names
    # Create a temporary handler to get feature names
    handler = Alpha158(start_time="2020-01-01", end_time="2020-01-02")
    handler.setup_data()
    return handler.get_feature_names()


def train_rolling(
    model: StructureExpertGNN,
    trainer: StructureTrainer,
    builder: GraphDataBuilder,
    date_range: List[datetime],
    qlib_dir: str,
) -> None:
    """
    Perform rolling training over date range.

    Args:
        model: Structure Expert GNN model.
        trainer: Structure trainer instance.
        builder: Graph data builder instance.
        date_range: List of dates to train on.
        qlib_dir: Path to Qlib data directory.
    """
    logger.info(f"Starting rolling training for {len(date_range)} days")

    total_loss = 0.0
    trained_days = 0

    for idx, train_date in enumerate(date_range):
        try:
            # Get data for this date
            # DatasetH provides data in segments, we need to extract daily data
            # For rolling training, we'll use the dataset's prepare method
            train_date_str = train_date.strftime("%Y-%m-%d")

            # Load features for this date using Alpha158 handler
            # DatasetH is designed for segment-based training, for daily rolling we use handler directly
            handler = Alpha158(
                start_time=train_date_str,
                end_time=train_date_str,
            )
            handler.setup_data()
            
            # Get features - use handler.data or fetch without data_key
            # For single date, we can use handler.data directly
            try:
                # Try using handler.data property
                df_x = handler.data
                if df_x is None or df_x.empty:
                    # Fallback: try fetch method
                    df_x = handler.fetch(col_set="feature")
            except Exception:
                # If that fails, try fetch without data_key
                df_x = handler.fetch(col_set="feature")

            # Load labels (future returns) for this date
            # For simplicity, use next day return as label
            if idx < len(date_range) - 1:
                next_date = date_range[idx + 1]
                next_date_str = next_date.strftime("%Y-%m-%d")

                # Get all instruments
                instruments = D.instruments()
                if not isinstance(instruments, list):
                    instruments = list(instruments)

                # Calculate returns
                df_y = D.features(
                    instruments,
                    ["Ref($close, -1) / $close - 1"],  # Next day return
                    start_time=train_date_str,
                    end_time=next_date_str,
                )
            else:
                # Last date, use current close as placeholder
                df_y = None

            # Check if we have valid data
            if df_x.empty:
                logger.warning(f"No data for {train_date_str}, skipping")
                continue

            # Align df_x and df_y indices if df_y is provided
            if df_y is not None and not df_y.empty:
                # Find common indices between df_x and df_y
                common_idx = df_x.index.intersection(df_y.index)
                if len(common_idx) == 0:
                    logger.warning(
                        f"No common indices between features and labels for {train_date_str}, skipping"
                    )
                    continue
                
                # Filter to common indices
                df_x_aligned = df_x.loc[common_idx]
                df_y_aligned = df_y.loc[common_idx]
                
                if df_x_aligned.empty:
                    logger.warning(f"No aligned data for {train_date_str}, skipping")
                    continue
            else:
                # No labels available, use df_x only
                df_x_aligned = df_x
                df_y_aligned = None
                logger.debug(f"No labels available for {train_date_str}, using features only")

            # Build graph for this date
            daily_graph = builder.get_daily_graph(df_x_aligned, df_y_aligned)

            # Skip if no edges (no industry connections)
            if daily_graph.edge_index.shape[1] == 0:
                logger.warning(f"No edges for {train_date_str}, skipping")
                continue

            # Train on this day's graph
            loss = trainer.train_step(daily_graph)
            total_loss += loss
            trained_days += 1

            if (idx + 1) % 10 == 0:
                avg_loss = total_loss / trained_days if trained_days > 0 else 0.0
                logger.info(
                    f"Processed {idx + 1}/{len(date_range)} days | "
                    f"Avg Loss: {avg_loss:.6f} | Last Loss: {loss:.6f}"
                )

        except Exception as e:
            logger.warning(f"Failed to train on {train_date_str}: {e}")
            continue

    if trained_days > 0:
        avg_loss = total_loss / trained_days
        logger.info(f"Rolling training completed: {trained_days} days, Avg Loss: {avg_loss:.6f}")
    else:
        logger.warning("No days were successfully trained")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Structure Expert GNN model with system data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on date range
  python python/tools/qlib/train/train_structure_expert.py \\
    --start-date 2024-01-01 --end-date 2024-12-31

  # Train with custom Qlib directory
  python python/tools/qlib/train/train_structure_expert.py \\
    --start-date 2024-01-01 --end-date 2024-12-31 \\
    --qlib-dir ~/.qlib/qlib_data/cn_data

  # Train with custom model parameters
  python python/tools/qlib/train/train_structure_expert.py \\
    --start-date 2024-01-01 --end-date 2024-12-31 \\
    --n-hidden 128 --n-heads 8
        """,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for training (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for training (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--qlib-dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Path to Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )

    parser.add_argument(
        "--n-feat",
        type=int,
        default=158,
        help="Number of input features (default: 158 for Alpha158)",
    )

    parser.add_argument(
        "--n-hidden",
        type=int,
        default=64,
        help="Hidden dimension size (default: 64)",
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema (default: quant)",
    )

    parser.add_argument(
        "--save-model",
        type=str,
        help="Path to save trained model (optional)",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Get date range (trading days only)
    qlib_dir = str(Path(args.qlib_dir).expanduser())
    qlib.init(provider_uri=qlib_dir, region="cn")
    
    # Get full calendar first to check available range
    full_calendar = D.calendar()
    if len(full_calendar) == 0:
        logger.error("No trading days found in Qlib data. Please check data export.")
        return 1
    
    available_start = pd.Timestamp(full_calendar[0])
    available_end = pd.Timestamp(full_calendar[-1])
    logger.info(f"Available calendar range: {available_start.date()} to {available_end.date()}")
    
    # Adjust date range to match available data
    if start_date < available_start:
        logger.warning(f"Start date {start_date.date()} is before available data, adjusting to {available_start.date()}")
        start_date = available_start
    if end_date > available_end:
        logger.warning(f"End date {end_date.date()} is after available data, adjusting to {available_end.date()}")
        end_date = available_end
    
    # Get calendar for adjusted date range
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    date_range = [pd.Timestamp(dt) for dt in calendar]

    if len(date_range) == 0:
        logger.error(f"No trading days in date range {start_date.date()} to {end_date.date()}")
        logger.error(f"Available calendar range: {available_start.date()} to {available_end.date()}")
        logger.error("Please adjust date range to match available data.")
        return 1

    logger.info(f"Training on {len(date_range)} trading days from {start_date.date()} to {end_date.date()}")

    # Load industry mapping
    logger.info("Loading industry mapping from database...")
    industry_map = load_industry_map(db_config, target_date=start_date, schema=args.schema)

    if not industry_map:
        logger.error("No industry mapping found. Please sync industry data first.")
        return 1

    # Initialize model and components
    logger.info("Initializing model and components...")
    model = StructureExpertGNN(n_feat=args.n_feat, n_hidden=args.n_hidden, n_heads=args.n_heads)
    trainer = StructureTrainer(model, lr=args.lr, device=args.device)
    builder = GraphDataBuilder(industry_map)

    # Perform rolling training
    logger.info("Starting rolling training...")
    train_rolling(model, trainer, builder, date_range, qlib_dir)

    # Save model if requested
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    logger.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

