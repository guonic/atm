#!/usr/bin/env python3
"""
Generate sample exit training data for testing.

This script generates synthetic position snapshots that can be used to test
the exit model training pipeline.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_sample_data(
    n_trades: int = 100,
    min_hold_days: int = 1,
    max_hold_days: int = 20,
    start_date: str = "2024-01-01",
    output_path: str = "outputs/exit_training_data.csv",
) -> pd.DataFrame:
    """
    Generate sample exit training data.
    
    Args:
        n_trades: Number of trades to generate.
        min_hold_days: Minimum holding days.
        max_hold_days: Maximum holding days.
        start_date: Start date for generating data.
        output_path: Output CSV file path.
    
    Returns:
        DataFrame with sample training data.
    """
    logger.info(f"Generating {n_trades} sample trades...")
    
    snapshots = []
    np.random.seed(42)  # For reproducibility
    
    start = pd.to_datetime(start_date)
    current_date = start
    
    symbols = [f"{i:06d}.SZ" for i in range(1, 51)]  # 50 different symbols
    
    for trade_id in range(1, n_trades + 1):
        # Random symbol
        symbol = np.random.choice(symbols)
        
        # Random entry price (10-100)
        entry_price = np.random.uniform(10, 100)
        
        # Random holding period
        hold_days = np.random.randint(min_hold_days, max_hold_days + 1)
        
        # Entry date
        entry_date = current_date + timedelta(days=np.random.randint(0, 100))
        
        # Generate daily snapshots
        highest_price = entry_price
        highest_date = entry_date
        
        for day in range(hold_days):
            date = entry_date + timedelta(days=day)
            
            # Simulate price movement
            # Random walk with slight upward bias
            price_change = np.random.normal(0.01, 0.02)  # 1% mean, 2% std
            current_price = entry_price * (1 + price_change) ** day
            
            # Add some noise
            current_price *= np.random.uniform(0.98, 1.02)
            
            # Update highest price
            if current_price > highest_price:
                highest_price = current_price
                highest_date = date
            
            # Generate OHLCV
            daily_range = current_price * np.random.uniform(0.01, 0.05)
            high = current_price + daily_range * np.random.uniform(0.3, 1.0)
            low = current_price - daily_range * np.random.uniform(0.3, 1.0)
            close = current_price
            volume = np.random.uniform(1000000, 10000000)
            
            # Calculate future 3-day max loss (for labeling)
            # Simulate: if price is near high, future loss is likely
            # If price has already dropped, future loss is less likely
            price_from_high = (highest_price - close) / highest_price
            if price_from_high > 0.05:  # Already dropped 5%+
                # More likely to continue dropping
                future_max_loss = np.random.uniform(-0.08, -0.02)
            else:
                # Less likely to drop
                future_max_loss = np.random.uniform(-0.05, 0.02)
            
            snapshot = {
                "trade_id": f"T{trade_id:04d}",
                "date": date,
                "symbol": symbol,
                "close": round(close, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "volume": int(volume),
                "entry_price": round(entry_price, 2),
                "highest_price_since_entry": round(highest_price, 2),
                "days_held": day,
                "next_3d_max_loss": round(future_max_loss, 4),
            }
            
            snapshots.append(snapshot)
        
        # Update current date
        current_date = entry_date + timedelta(days=hold_days)
    
    df = pd.DataFrame(snapshots)
    df = df.sort_values(["trade_id", "date"]).reset_index(drop=True)
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {len(df)} snapshots for {df['trade_id'].nunique()} trades")
    logger.info(f"Saved to {output_path}")
    logger.info(f"Label distribution: {(df['next_3d_max_loss'] < -0.03).sum()} positive, "
                f"{(df['next_3d_max_loss'] >= -0.03).sum()} negative")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate sample exit training data for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default sample data
  python python/tools/qlib/generate_sample_exit_data.py

  # Generate more trades
  python python/tools/qlib/generate_sample_exit_data.py \\
    --n-trades 500 \\
    --output outputs/exit_training_data.csv
        """,
    )

    parser.add_argument(
        "--n-trades",
        type=int,
        default=100,
        help="Number of trades to generate (default: 100)",
    )

    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=1,
        help="Minimum holding days (default: 1)",
    )

    parser.add_argument(
        "--max-hold-days",
        type=int,
        default=20,
        help="Maximum holding days (default: 20)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for generating data (default: 2024-01-01)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exit_training_data.csv",
        help="Output CSV file path (default: outputs/exit_training_data.csv)",
    )

    args = parser.parse_args()

    # Generate sample data
    df = generate_sample_data(
        n_trades=args.n_trades,
        min_hold_days=args.min_hold_days,
        max_hold_days=args.max_hold_days,
        start_date=args.start_date,
        output_path=args.output,
    )

    logger.info("Sample data generation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
