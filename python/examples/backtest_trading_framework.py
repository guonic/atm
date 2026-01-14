#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_trading_framework.py

使用新的 trading 框架进行回测的示例脚本。

这个脚本展示了如何使用完全独立于 Qlib 状态管理的交易框架：
- 使用自定义的 Account、Position、OrderBook
- 使用 AsymmetricStrategy 协调买入和卖出模型
- 使用自定义的 Executor 执行订单

Usage:
    python backtest_trading_framework.py \
        --model_path models/structure_expert_directional.pth \
        --exit_model_path models/exit_model.pkl \
        --start_date 2025-07-01 \
        --end_date 2025-08-01 \
        --initial_cash 1000000
"""

import argparse
import logging
import sys
from pathlib import Path

import qlib

from nq.config import load_config
from nq.utils.industry import load_industry_map
from tools.qlib.train.structure_expert import (
    GraphDataBuilder,
    load_structure_expert_model,
)
from nq.analysis.exit import ExitModel

# Import trading framework
from nq.trading.strategies import AsymmetricStrategy
from nq.trading.strategies.buy_models import StructureExpertBuyModel
from nq.trading.strategies.sell_models import MLExitSellModel
from nq.trading.state import Account, PositionManager, OrderBook
from nq.trading.logic import RiskManager, PositionAllocator
from nq.trading.storage import MemoryStorage
from nq.trading.backtest import run_custom_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest using new trading framework"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Structure Expert model file (.pth)",
    )
    parser.add_argument(
        "--exit_model_path",
        type=str,
        required=True,
        help="Path to exit model file (.pkl)",
    )
    parser.add_argument(
        "--exit_scaler_path",
        type=str,
        default=None,
        help="Path to exit model scaler file (optional)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=1000000.0,
        help="Initial cash amount (default: 1000000)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of positions to hold (default: 30)",
    )
    parser.add_argument(
        "--exit_threshold",
        type=float,
        default=0.65,
        help="Exit model threshold (default: 0.65)",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--n_feat",
        type=int,
        default=158,
        help="Number of input features (default: 158)",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=128,
        help="Hidden layer size (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
        help="Path to config file (for database access)",
    )
    
    args = parser.parse_args()
    
    # Initialize Qlib
    qlib_dir = str(Path(args.qlib_dir).expanduser())
    qlib.init(provider_uri=qlib_dir, region="cn")
    logger.info(f"Qlib initialized with data directory: {qlib_dir}")
    
    # Load config for database access (optional, for industry mapping)
    db_config = None
    industry_map = {}
    try:
        config = load_config(args.config_path)
        db_config = config.database
        from datetime import datetime
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
        industry_map = load_industry_map(db_config, target_date=end_dt)
        logger.info(f"Loaded industry mapping: {len(industry_map)} stocks")
    except Exception as e:
        logger.warning(f"Failed to load config or industry map: {e}")
    
    # Create graph builder
    builder = GraphDataBuilder(industry_map)
    
    # Load Structure Expert model
    logger.info(f"Loading Structure Expert model from {args.model_path}")
    structure_expert_model = load_structure_expert_model(
        model_path=args.model_path,
        n_feat=args.n_feat,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        device=args.device,
    )
    
    # Create buy model
    logger.info("Creating buy model...")
    buy_model = StructureExpertBuyModel(
        model_path=args.model_path,
        builder=builder,
        device=args.device,
        n_feat=args.n_feat,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
    )
    
    # Load exit model
    logger.info(f"Loading exit model from {args.exit_model_path}")
    exit_model = ExitModel.load(args.exit_model_path)
    if args.exit_scaler_path:
        exit_model.scaler = ExitModel.load_scaler(args.exit_scaler_path)
    
    # Create sell model
    logger.info("Creating sell model...")
    sell_model = MLExitSellModel(
        exit_model=exit_model,
        threshold=args.exit_threshold,
    )
    
    # Initialize storage
    storage = MemoryStorage()
    
    # Initialize account
    account = Account(
        account_id="backtest_001",
        available_cash=args.initial_cash,
        initial_cash=args.initial_cash,
    )
    
    # Initialize position manager
    position_manager = PositionManager(account, storage)
    
    # Set position_manager reference in account (bidirectional relationship)
    account.set_position_manager(position_manager)
    
    # Initialize order book
    order_book = OrderBook(storage)
    
    # Initialize logic layer
    risk_manager = RiskManager(account, position_manager, storage)
    position_allocator = PositionAllocator(target_positions=args.top_k)
    
    # Create strategy
    logger.info("Creating dual-model strategy...")
    strategy = AsymmetricStrategy(
        buy_model=buy_model,
        sell_model=sell_model,
        position_manager=position_manager,
        order_book=order_book,
        risk_manager=risk_manager,
        position_allocator=position_allocator,
        account=account,
    )
    
    # Run backtest
    logger.info(f"Running backtest from {args.start_date} to {args.end_date}...")
    results = run_custom_backtest(
        strategy=strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        instruments=None,  # Use all instruments
        storage_backend=storage,
    )
    
    # Print results
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    
    final_account = results['account']
    final_positions = results['positions']
    all_orders = results['orders']
    snapshots = results['snapshots']
    
    # Calculate final metrics
    final_total_value = final_account.get_total_value(position_manager)
    total_return = (final_total_value - args.initial_cash) / args.initial_cash * 100
    
    logger.info(f"Initial Cash: {args.initial_cash:,.2f}")
    logger.info(f"Final Total Value: {final_total_value:,.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Final Positions: {len(final_positions)}")
    logger.info(f"Total Orders: {len(all_orders)}")
    
    # Count buy/sell orders
    buy_orders = [o for o in all_orders if o.side.value == "BUY"]
    sell_orders = [o for o in all_orders if o.side.value == "SELL"]
    filled_orders = [o for o in all_orders if o.status.value in ["FILLED", "PARTIAL_FILLED"]]
    
    logger.info(f"Buy Orders: {len(buy_orders)}")
    logger.info(f"Sell Orders: {len(sell_orders)}")
    logger.info(f"Filled Orders: {len(filled_orders)}")
    
    # Print position details
    if final_positions:
        logger.info("\nFinal Positions:")
        for symbol, pos in final_positions.items():
            logger.info(
                f"  {symbol}: {pos.amount:.0f} shares @ {pos.entry_price:.2f}, "
                f"high={pos.high_price_since_entry:.2f}"
            )
    
    # Print performance over time
    if snapshots:
        logger.info("\nPerformance Snapshot (first 5 and last 5 days):")
        for i, snapshot in enumerate(snapshots[:5] + snapshots[-5:]):
            logger.info(
                f"  {snapshot['date']}: "
                f"total_value={snapshot['total_value']:,.2f}, "
                f"positions={snapshot['position_count']}"
            )
    
    logger.info("=" * 80)
    logger.info("Backtest completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
