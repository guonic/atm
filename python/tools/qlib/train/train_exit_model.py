#!/usr/bin/env python3
"""
Train exit prediction model.

This script trains a logistic regression model to predict when to exit positions
based on momentum exhaustion and position management features.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.analysis.exit import ExitFeatureBuilder, ExitModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load training data from CSV.
    
    Expected columns:
    - trade_id: Trade identifier
    - date: Snapshot date
    - symbol: Stock symbol
    - close, high, low, volume: OHLCV data
    - entry_price: Entry price
    - highest_price_since_entry: Highest price since entry
    - days_held: Days held
    - next_3d_max_loss: Future 3-day maximum loss (for labeling)
    
    Args:
        data_path: Path to CSV file.
    
    Returns:
        DataFrame with training data.
    
    Raises:
        FileNotFoundError: If data file does not exist.
    """
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(
            f"Training data file not found: {data_path}\n"
            f"Please run the data extraction script first:\n"
            f"  python python/tools/qlib/extract_exit_training_data.py --output {data_path}\n"
            f"Or generate sample data for testing."
        )
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    logger.info(f"Loaded {len(df)} records from {data_path}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def build_features_and_labels(
    df: pd.DataFrame, feature_builder: ExitFeatureBuilder
) -> pd.DataFrame:
    """
    Build features and labels from training data.
    
    Args:
        df: DataFrame with training data.
        feature_builder: Feature builder instance.
    
    Returns:
        DataFrame with features and labels.
    """
    # Group by trade_id to process each position separately
    feature_list = []

    for trade_id, trade_df in df.groupby("trade_id"):
        trade_df = trade_df.sort_values("date").reset_index(drop=True)

        # Build features for this trade
        features = feature_builder.build_features_with_label(
            daily_df=trade_df,
            entry_price=trade_df["entry_price"].iloc[0]
            if "entry_price" in trade_df.columns
            else None,
            highest_price_since_entry=trade_df["highest_price_since_entry"]
            if "highest_price_since_entry" in trade_df.columns
            else None,
            days_held=trade_df["days_held"] if "days_held" in trade_df.columns else None,
        )

        if not features.empty:
            # Add metadata (optional, for tracking purposes)
            # These columns will be automatically excluded from training
            features["trade_id"] = trade_id
            features["symbol"] = trade_df["symbol"].iloc[0]
            features["date"] = trade_df["date"].values[: len(features)]

            feature_list.append(features)

    if not feature_list:
        logger.warning("No features generated")
        return pd.DataFrame()

    result_df = pd.concat(feature_list, ignore_index=True)

    logger.info(f"Built features: {len(result_df)} samples")
    logger.info(f"Positive labels: {result_df['label'].sum()}")
    logger.info(f"Negative labels: {(result_df['label'] == 0).sum()}")

    return result_df


def train_model(
    feature_df: pd.DataFrame,
    model_path: str,
    scaler_path: Optional[str] = None,
    C: float = 0.1,
    class_weight: str = "balanced",
) -> ExitModel:
    """
    Train exit model.
    
    Args:
        feature_df: DataFrame with features and labels.
        model_path: Path to save model.
        scaler_path: Path to save scaler.
        C: Regularization strength.
        class_weight: Class weight strategy.
    
    Returns:
        Trained ExitModel instance.
    """
    # Initialize model
    model = ExitModel()

    # Train model
    model.fit(feature_df, C=C, class_weight=class_weight)

    # Save model
    model.save(model_path, scaler_path)

    # Print feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        logger.info("\nFeature Importance:")
        logger.info(f"\n{importance.to_string()}")

    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train exit prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with default parameters
  python python/tools/qlib/train/train_exit_model.py \\
    --data outputs/exit_training_data.csv \\
    --model models/exit_model.pkl

  # Train with custom parameters
  python python/tools/qlib/train/train_exit_model.py \\
    --data outputs/exit_training_data.csv \\
    --model models/exit_model.pkl \\
    --C 0.01 --class-weight balanced
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data CSV file. "
        "If file doesn't exist, you can generate sample data with: "
        "python python/tools/qlib/generate_sample_exit_data.py --output <path>",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/exit_model.pkl",
        help="Path to save model (default: models/exit_model.pkl)",
    )

    parser.add_argument(
        "--scaler",
        type=str,
        help="Path to save scaler (default: auto-generated from model path)",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=0.1,
        help="Regularization strength (default: 0.1)",
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "None"],
        help="Class weight strategy (default: balanced)",
    )

    parser.add_argument(
        "--ma-period",
        type=int,
        default=5,
        help="Moving average period for features (default: 5)",
    )

    parser.add_argument(
        "--volume-ma-period",
        type=int,
        default=5,
        help="Volume moving average period (default: 5)",
    )

    args = parser.parse_args()

    # Load training data
    logger.info(f"Loading training data from {args.data}")
    df = load_training_data(args.data)

    if df.empty:
        logger.error("No training data loaded")
        return 1

    # Initialize feature builder
    feature_builder = ExitFeatureBuilder(
        ma_period=args.ma_period, volume_ma_period=args.volume_ma_period
    )

    # Build features and labels
    logger.info("Building features and labels...")
    feature_df = build_features_and_labels(df, feature_builder)

    if feature_df.empty:
        logger.error("No features generated")
        return 1

    # Train model
    logger.info("Training model...")
    class_weight = None if args.class_weight == "None" else args.class_weight
    model = train_model(
        feature_df=feature_df,
        model_path=args.model,
        scaler_path=args.scaler,
        C=args.C,
        class_weight=class_weight,
    )

    logger.info(f"Model trained and saved to {args.model}")
    logger.info("Training completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
