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
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.contrib.data.handler import Alpha158
from qlib.data import D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.config import load_config
from nq.utils.industry import load_industry_map

# Import structure expert model using standard package import
from tools.qlib.train.structure_expert import (
    DirectionalStockGNN,
    GraphDataBuilder,
    StructureExpertGNN,
    StructureTrainer,
)
from tools.qlib.utils import (
    align_and_clean_features_labels,
    clean_dataframe,
    get_handler_data,
    is_valid_number,
    load_next_day_returns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    model: torch.nn.Module,
    trainer: StructureTrainer,
    builder: GraphDataBuilder,
    date_range: List[datetime],
    qlib_dir: str,
    use_edge_attr: bool = False,
) -> None:
    """
    Perform rolling training over date range.

    Args:
        model: GNN model instance (StructureExpertGNN or DirectionalStockGNN).
        trainer: Structure trainer instance.
        builder: Graph data builder instance.
        date_range: List of dates to train on.
        qlib_dir: Path to Qlib data directory.
        use_edge_attr: Whether to include edge attributes in graph data.
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
            
            # Get features using unified data retrieval function
            df_x = get_handler_data(handler, col_set="feature")

            # Load labels (future returns) for this date using unified function
            df_y = load_next_day_returns(
                current_date=train_date,
                date_range=date_range,
                current_idx=idx,
            )

            # Check if we have valid data
            if df_x.empty:
                logger.warning(f"No data for {train_date_str}, skipping")
                continue

            # Clean NaN and Inf values in features using unified function
            df_x = clean_dataframe(
                df_x, fill_value=0.0, log_stats=True, context=f"features for {train_date_str}"
            )

            # Align and clean features and labels using unified function
            df_x_aligned, df_y_aligned = align_and_clean_features_labels(
                df_x=df_x,
                df_y=df_y,
                fill_value=0.0,
                log_stats=True,
                context=train_date_str,
            )

            # Build graph for this date
            daily_graph = builder.get_daily_graph(
                df_x_aligned, df_y_aligned, include_edge_attr=use_edge_attr
            )

            # Skip if no edges (no industry connections)
            if daily_graph.edge_index.shape[1] == 0:
                logger.warning(f"No edges for {train_date_str}, skipping")
                continue

            # Train on this day's graph
            loss = trainer.train_step(daily_graph)
            
            # Check if loss is valid using unified function
            if is_valid_number(loss):
                total_loss += loss
                trained_days += 1
                logger.debug(f"Day {train_date_str}: Loss = {loss:.6f}")
            else:
                logger.warning(f"Day {train_date_str}: Loss is invalid (None/NaN/Inf), skipping from average")

            # Print progress every 10 days or when no successful training yet
            if (idx + 1) % 3 == 0 and trained_days != 0:
                avg_loss = total_loss / trained_days
                # Format loss value safely using unified function
                if is_valid_number(loss):
                    loss_str = f"{loss:.6f}"
                else:
                    loss_str = "NaN/Inf"

                logger.info(
                    f"Progress: {idx + 1}/{len(date_range)} days processed | "
                    f"Successfully trained: {trained_days} days | "
                    f"Avg Loss: {avg_loss:.6f} | Last Loss: {loss_str}"
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

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["structure_expert", "directional"],
        default="structure_expert",
        help="Model type to use: 'structure_expert' (default) or 'directional' (DirectionalStockGNN)",
    )

    parser.add_argument(
        "--edge-in-channels",
        type=int,
        default=4,
        help="Number of edge attribute channels for DirectionalStockGNN (default: 4)",
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
    logger.info(f"Initializing {args.model_type} model and components...")
    
    if args.model_type == "directional":
        # Use DirectionalStockGNN model
        use_edge_attr = True
        model = DirectionalStockGNN(
            node_in_channels=args.n_feat,
            edge_in_channels=args.edge_in_channels,
            hidden_channels=args.n_hidden,
        )
        logger.info(
            f"Using DirectionalStockGNN: "
            f"node_in={args.n_feat}, edge_in={args.edge_in_channels}, hidden={args.n_hidden}"
        )
    else:
        # Use StructureExpertGNN model (default)
        use_edge_attr = False
        model = StructureExpertGNN(
            n_feat=args.n_feat, n_hidden=args.n_hidden, n_heads=args.n_heads
        )
        logger.info(
            f"Using StructureExpertGNN: "
            f"n_feat={args.n_feat}, n_hidden={args.n_hidden}, n_heads={args.n_heads}"
        )
    
    trainer = StructureTrainer(model, lr=args.lr, device=args.device)
    builder = GraphDataBuilder(industry_map)

    # Perform rolling training
    logger.info("Starting rolling training...")
    train_rolling(model, trainer, builder, date_range, qlib_dir, use_edge_attr=use_edge_attr)

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

