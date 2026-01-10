#!/usr/bin/env python3
"""
Extract position snapshots for exit model training.

This script extracts daily position snapshots from backtest results,
which can be used to train the exit prediction model.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.analysis.backtest.eidos_structure_expert import extract_trades_from_backtest_results
from nq.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PositionSnapshotExtractor:
    """
    Extract daily position snapshots from backtest results.
    
    This class tracks positions over time and extracts snapshots for each day
    a position is held, including entry price, current price, highest price,
    and other relevant information.
    """

    def __init__(self):
        """Initialize position snapshot extractor."""
        self.snapshots: List[Dict] = []
        self.active_positions: Dict[str, Dict] = {}

    def add_buy_order(
        self,
        symbol: str,
        date: datetime,
        price: float,
        amount: int,
        trade_id: Optional[str] = None,
    ) -> None:
        """
        Add a buy order (open position).
        
        Args:
            symbol: Stock symbol.
            date: Trade date.
            price: Entry price.
            amount: Order amount.
            trade_id: Optional trade ID for tracking.
        """
        if trade_id is None:
            trade_id = f"{symbol}_{date.strftime('%Y%m%d')}"

        position_key = f"{trade_id}_{symbol}"

        self.active_positions[position_key] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_date": date,
            "entry_price": price,
            "amount": amount,
            "highest_price_since_entry": price,
            "highest_date": date,
        }

        logger.debug(
            f"Opened position: {symbol} at {price:.2f} on {date.strftime('%Y-%m-%d')}"
        )

    def add_sell_order(
        self,
        symbol: str,
        date: datetime,
        price: float,
        amount: int,
        trade_id: Optional[str] = None,
    ) -> None:
        """
        Add a sell order (close position).
        
        Args:
            symbol: Stock symbol.
            date: Trade date.
            price: Sell price.
            amount: Order amount.
            trade_id: Optional trade ID for tracking.
        """
        if trade_id is None:
            trade_id = f"{symbol}_{date.strftime('%Y%m%d')}"

        position_key = f"{trade_id}_{symbol}"

        if position_key in self.active_positions:
            position = self.active_positions[position_key]
            # Add final snapshot before closing
            self.add_daily_snapshot(
                symbol=symbol,
                date=date,
                close=price,
                high=price,
                low=price,
                volume=0,  # Volume not available from order
            )
            del self.active_positions[position_key]
            logger.debug(
                f"Closed position: {symbol} at {price:.2f} on {date.strftime('%Y-%m-%d')}"
            )

    def add_daily_snapshot(
        self,
        symbol: str,
        date: datetime,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> None:
        """
        Add daily snapshot for active positions.
        
        Args:
            symbol: Stock symbol.
            date: Snapshot date.
            close: Close price.
            high: High price (if None, uses close).
            low: Low price (if None, uses close).
            volume: Volume (if None, uses 0).
        """
        if high is None:
            high = close
        if low is None:
            low = close
        if volume is None:
            volume = 0.0

        # Update all active positions for this symbol
        for position_key, position in list(self.active_positions.items()):
            if position["symbol"] == symbol:
                # Update highest price
                if close > position["highest_price_since_entry"]:
                    position["highest_price_since_entry"] = close
                    position["highest_date"] = date

                # Calculate days held
                days_held = (date - position["entry_date"]).days

                # Add snapshot
                snapshot = {
                    "trade_id": position["trade_id"],
                    "date": date,
                    "symbol": symbol,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "entry_price": position["entry_price"],
                    "highest_price_since_entry": position["highest_price_since_entry"],
                    "days_held": days_held,
                }

                self.snapshots.append(snapshot)

    def get_snapshots_df(self) -> pd.DataFrame:
        """
        Get snapshots as DataFrame.
        
        Returns:
            DataFrame with position snapshots.
        """
        if not self.snapshots:
            logger.warning("No snapshots collected")
            return pd.DataFrame()

        df = pd.DataFrame(self.snapshots)
        df = df.sort_values(["trade_id", "date"]).reset_index(drop=True)

        logger.info(f"Extracted {len(df)} position snapshots")
        logger.info(f"Unique trades: {df['trade_id'].nunique()}")
        logger.info(f"Unique symbols: {df['symbol'].nunique()}")

        return df


def extract_from_executed_orders(
    executed_orders: List[Dict],
    price_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Extract position snapshots from executed orders.
    
    Args:
        executed_orders: List of executed order dictionaries.
        price_data: Optional DataFrame with OHLCV data (index: date, columns: symbols).
    
    Returns:
        DataFrame with position snapshots.
    """
    extractor = PositionSnapshotExtractor()

    # Sort orders by date
    sorted_orders = sorted(executed_orders, key=lambda x: x.get("deal_time", datetime.min))

    for order in sorted_orders:
        symbol = order.get("instrument", "")
        deal_time = order.get("deal_time")
        direction = order.get("direction", 0)
        price = order.get("trade_price", 0.0)
        amount = order.get("amount", 0)

        if not symbol or deal_time is None or amount <= 0:
            continue

        if isinstance(deal_time, str):
            deal_time = pd.to_datetime(deal_time)
        elif hasattr(deal_time, "date"):
            deal_time = pd.to_datetime(deal_time)

        if direction == 1:  # Buy
            extractor.add_buy_order(
                symbol=symbol,
                date=deal_time,
                price=price,
                amount=amount,
            )
        elif direction == -1:  # Sell
            extractor.add_sell_order(
                symbol=symbol,
                date=deal_time,
                price=price,
                amount=amount,
            )

        # Add daily snapshot if price data available
        if price_data is not None and symbol in price_data.columns:
            # Get price data for this symbol and date
            if deal_time in price_data.index:
                row = price_data.loc[deal_time]
                if isinstance(row, pd.Series):
                    close = row.get("close", price)
                    high = row.get("high", close)
                    low = row.get("low", close)
                    volume = row.get("volume", 0.0)
                else:
                    close = price
                    high = close
                    low = close
                    volume = 0.0

                extractor.add_daily_snapshot(
                    symbol=symbol,
                    date=deal_time,
                    close=close,
                    high=high,
                    low=low,
                    volume=volume,
                )

    return extractor.get_snapshots_df()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract position snapshots for exit model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exit_training_data.csv",
        help="Output CSV file path (default: outputs/exit_training_data.csv)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # For now, this is a template script
    # In practice, you would:
    # 1. Load backtest results
    # 2. Extract executed orders
    # 3. Get price data
    # 4. Extract snapshots
    # 5. Save to CSV

    logger.info(
        "This script is a template. "
        "Please integrate it with your backtest workflow to extract position snapshots."
    )
    logger.info(
        "Example usage: "
        "python python/tools/qlib/extract_exit_training_data.py --output outputs/exit_training_data.csv"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
