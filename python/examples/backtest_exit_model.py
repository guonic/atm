#!/usr/bin/env python3
"""
完整的买入+卖出模型回测脚本

这个脚本集成了：
1. Structure Expert 模型（买入模型）- 用于生成选股信号
2. Exit 模型（卖出模型）- 用于决定何时卖出

工作流程：
1. 加载训练好的 Structure Expert 模型（买入信号）
2. 加载训练好的 Exit 模型（卖出信号）
3. 使用 Structure Expert 模型生成每日预测分数
4. 使用 AsymmetricStrategy + MLExitSellModel 进行回测（结合买入和卖出逻辑）
5. 对比有/无退出模型的效果

这是集成了买入和卖出模型的完整回测脚本。
使用自定义回测框架，完全独立于 Qlib 的策略系统。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import qlib
import torch

# Note: PYTHONPATH must be set to project root directory
# Example: export PYTHONPATH=/path/to/atm:$PYTHONPATH

from tools.qlib.train.structure_expert import GraphDataBuilder, load_structure_expert_model
from nq.utils.industry import load_industry_map
from nq.analysis.exit import ExitModel
from nq.config import load_config

# Import custom trading framework
from nq.trading.strategies import AsymmetricStrategy
from nq.trading.strategies.buy_models import StructureExpertBuyModel
from nq.trading.strategies.sell_models import MLExitSellModel
from nq.trading.state import Account, PositionManager, OrderBook
from nq.trading.logic import RiskManager, PositionAllocator
from nq.trading.storage import MemoryStorage
from nq.trading.backtest import run_custom_backtest

# Import Eidos integration
from nq.analysis.backtest.eidos_integration import EidosBacktestWriter
from nq.trading.utils.eidos_converter import save_custom_backtest_to_eidos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default constants
DEFAULT_INITIAL_CASH = 1000000.0
DEFAULT_TOP_K = 30


def create_asymmetric_strategy(
    model_path: str,
    exit_model_path: str,
    exit_scaler_path: str = None,
    exit_threshold: float = 0.65,
    top_k: int = DEFAULT_TOP_K,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    builder: GraphDataBuilder = None,
    device: str = "cuda",
    n_feat: int = 158,
    n_hidden: int = 128,
    n_heads: int = 8,
) -> AsymmetricStrategy:
    """
    Create AsymmetricStrategy with Structure Expert buy model and ML Exit sell model.
    
    Args:
        model_path: Path to Structure Expert model file.
        exit_model_path: Path to trained exit model.
        exit_scaler_path: Path to feature scaler (if None, auto-generated).
        exit_threshold: Risk probability threshold for exit signal.
        top_k: Number of top stocks to select.
        initial_cash: Initial cash amount.
        builder: GraphDataBuilder instance.
        device: Device to use (cuda/cpu).
        n_feat: Number of input features.
        n_hidden: Hidden layer size.
        n_heads: Number of attention heads.
    
    Returns:
        AsymmetricStrategy instance with all required components.
    """
    logger.info(f"Creating AsymmetricStrategy with exit model: {exit_model_path}")
    
    # Create buy model
    logger.info("Creating Structure Expert buy model...")
    buy_model = StructureExpertBuyModel(
        model_path=model_path,
        builder=builder,
        device=device,
        n_feat=n_feat,
        n_hidden=n_hidden,
        n_heads=n_heads,
    )
    
    # Load exit model
    logger.info(f"Loading exit model from {exit_model_path}")
    exit_model = ExitModel.load(exit_model_path)
    if exit_scaler_path:
        exit_model.scaler = ExitModel.load_scaler(exit_scaler_path)
    
    # Create sell model
    logger.info("Creating ML Exit sell model...")
    sell_model = MLExitSellModel(
        exit_model=exit_model,
        threshold=exit_threshold,
    )
    
    # Initialize storage
    storage = MemoryStorage()
    
    # Initialize account
    account = Account(
        account_id="backtest_001",
        available_cash=initial_cash,
        initial_cash=initial_cash,
    )
    
    # Initialize position manager
    position_manager = PositionManager(account, storage)
    account.set_position_manager(position_manager)
    
    # Initialize order book
    order_book = OrderBook(storage)
    
    # Initialize logic layer
    risk_manager = RiskManager(account, position_manager, storage)
    position_allocator = PositionAllocator(target_positions=top_k)
    
    # Create strategy
    strategy = AsymmetricStrategy(
        buy_model=buy_model,
        sell_model=sell_model,
        position_manager=position_manager,
        order_book=order_book,
        risk_manager=risk_manager,
        position_allocator=position_allocator,
        account=account,
    )
    
    return strategy


def calculate_metrics_from_results(results: dict, initial_cash: float) -> dict:
    """
    Calculate performance metrics from custom backtest results.
    
    Args:
        results: Results dict from run_custom_backtest.
        initial_cash: Initial cash amount.
    
    Returns:
        Dict with calculated metrics.
    """
    final_account = results['account']
    final_positions = results['positions']
    all_orders = results['orders']
    snapshots = results['snapshots']
    
    # Get position manager from account
    position_manager = final_account.position_manager
    
    # Calculate final total value
    # Note: get_total_value expects current_prices dict (or None to use entry_price)
    # Since we don't have current prices at the end, pass None to use entry_price
    final_total_value = final_account.get_total_value(current_prices=None)
    total_return = (final_total_value - initial_cash) / initial_cash
    
    # Calculate daily returns from snapshots
    daily_returns = []
    if len(snapshots) > 1:
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i-1]['total_value']
            curr_value = snapshots[i]['total_value']
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
    
    # Calculate metrics
    metrics = {
        'total_return': total_return,
        'final_total_value': final_total_value,
        'initial_cash': initial_cash,
        'final_positions': len(final_positions),
        'total_orders': len(all_orders),
        'daily_returns': daily_returns,
    }
    
    # Calculate annualized return
    if len(snapshots) > 1:
        days = len(snapshots)
        years = days / 252.0  # Trading days per year
        if years > 0:
            metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1
        else:
            metrics['annualized_return'] = 0.0
    else:
        metrics['annualized_return'] = 0.0
    
    # Calculate Sharpe ratio
    if len(daily_returns) > 0:
        import numpy as np
        returns_array = np.array(daily_returns)
        if returns_array.std() > 0:
            metrics['sharpe_ratio'] = float(returns_array.mean() / returns_array.std() * (252 ** 0.5))
        else:
            metrics['sharpe_ratio'] = 0.0
    else:
        metrics['sharpe_ratio'] = 0.0
    
    # Calculate max drawdown
    if len(snapshots) > 0:
        values = [s['total_value'] for s in snapshots]
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        metrics['max_drawdown'] = max_dd
    else:
        metrics['max_drawdown'] = 0.0
    
    return metrics


def compare_strategies(
    model_path: str,
    exit_model_path: str,
    start_date: str,
    end_date: str,
    top_k: int = DEFAULT_TOP_K,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    exit_scaler_path: str = None,
    exit_threshold: float = 0.65,
    builder: GraphDataBuilder = None,
    device: str = "cuda",
    n_feat: int = 158,
    n_hidden: int = 128,
    n_heads: int = 8,
):
    """
    Compare backtest results with and without exit model.
    
    Note: Currently only supports comparison using custom backtest framework.
    Baseline strategy (without exit model) is not yet implemented in custom framework.
    
    Args:
        model_path: Path to Structure Expert model.
        exit_model_path: Path to exit model.
        start_date: Start date.
        end_date: End date.
        top_k: Number of top stocks.
        initial_cash: Initial cash.
        exit_scaler_path: Path to exit model scaler.
        exit_threshold: Exit threshold.
        builder: GraphDataBuilder instance.
        device: Device to use.
        n_feat: Number of features.
        n_hidden: Hidden layer size.
        n_heads: Number of attention heads.
    """
    logger.info("=" * 80)
    logger.info("Running backtest WITH ML Exit model (custom framework)")
    logger.info("=" * 80)
    
    # Strategy: With exit model (using custom framework)
    logger.info("\nRunning backtest WITH exit model...")
    strategy = create_asymmetric_strategy(
        model_path=model_path,
        exit_model_path=exit_model_path,
        exit_scaler_path=exit_scaler_path,
        exit_threshold=exit_threshold,
        top_k=top_k,
        initial_cash=initial_cash,
        builder=builder,
        device=device,
        n_feat=n_feat,
        n_hidden=n_hidden,
        n_heads=n_heads,
    )
    
    # Get storage from strategy's position_manager
    storage = strategy.position_manager.storage
    
    ml_exit_results = run_custom_backtest(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        instruments=None,
        storage_backend=storage,
    )
    
    # Calculate metrics
    ml_exit_metrics = calculate_metrics_from_results(ml_exit_results, initial_cash)
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS (ML Exit Strategy)")
    print("=" * 80)
    
    print(f"\n📊 Performance Metrics:")
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Total Return':<25} {ml_exit_metrics['total_return']:>14.2%}")
    print(f"{'Annualized Return':<25} {ml_exit_metrics['annualized_return']:>14.2%}")
    print(f"{'Sharpe Ratio':<25} {ml_exit_metrics['sharpe_ratio']:>14.4f}")
    print(f"{'Max Drawdown':<25} {ml_exit_metrics['max_drawdown']:>14.2%}")
    print(f"{'Final Total Value':<25} {ml_exit_metrics['final_total_value']:>14,.2f}")
    print(f"{'Final Positions':<25} {ml_exit_metrics['final_positions']:>15}")
    print(f"{'Total Orders':<25} {ml_exit_metrics['total_orders']:>15}")
    print("-" * 40)
    
    # Print position details
    final_positions = ml_exit_results['positions']
    if final_positions:
        print("\nFinal Positions:")
        for symbol, pos in list(final_positions.items())[:10]:  # Show first 10
            print(f"  {symbol}: {pos.amount:.0f} shares @ {pos.entry_price:.2f}, high={pos.high_price_since_entry:.2f}")
        if len(final_positions) > 10:
            print(f"  ... and {len(final_positions) - 10} more positions")
    
    # Print performance snapshot
    snapshots = ml_exit_results['snapshots']
    if snapshots:
        print("\nPerformance Snapshot (first 5 and last 5 days):")
        for snapshot in snapshots[:5] + snapshots[-5:]:
            print(f"  {snapshot['date']}: total_value={snapshot['total_value']:,.2f}, positions={snapshot['position_count']}")
    
    print("\n" + "=" * 80)
    
    return {
        "ml_exit": ml_exit_results,
        "metrics": ml_exit_metrics,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest Structure Expert model with ML-based exit strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with exit model
  python python/examples/backtest_exit_model.py \\
    --model_path models/structure_expert.pth \\
    --exit_model_path models/exit_model.pkl \\
    --start_date 2024-01-01 \\
    --end_date 2024-06-30

  # Compare with and without exit model
  python python/examples/backtest_exit_model.py \\
    --model_path models/structure_expert.pth \\
    --exit_model_path models/exit_model.pkl \\
    --start_date 2024-01-01 \\
    --end_date 2024-06-30 \\
    --compare

  # Custom exit threshold
  python python/examples/backtest_exit_model.py \\
    --model_path models/structure_expert.pth \\
    --exit_model_path models/exit_model.pkl \\
    --start_date 2024-01-01 \\
    --end_date 2024-06-30 \\
    --exit_threshold 0.7
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained Structure Expert model file (.pth)",
    )
    parser.add_argument(
        "--exit_model_path",
        type=str,
        required=True,
        help="Path to trained exit model file (.pkl)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date of backtest period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date of backtest period (YYYY-MM-DD)",
    )

    # Optional arguments
    parser.add_argument(
        "--exit_scaler_path",
        type=str,
        default=None,
        help="Path to exit model scaler (if None, auto-generated from model path)",
    )
    parser.add_argument(
        "--exit_threshold",
        type=float,
        default=0.65,
        help="Risk probability threshold for exit signal (default: 0.65)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top stocks to select (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--buffer_ratio",
        type=float,
        default=0.15,
        help="Buffer ratio for reducing turnover (default: 0.15)",
    )
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"Initial cash amount (default: {DEFAULT_INITIAL_CASH})",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark code (e.g., 'SH000300' or '000300.SH')",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results with and without exit model",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Path to Qlib data directory (default: ~/.qlib/qlib_data/cn_data)",
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
        "--enable_eidos",
        action="store_true",
        help="Enable Eidos integration to save backtest results to database",
    )
    parser.add_argument(
        "--eidos_exp_name",
        type=str,
        default=None,
        help="Experiment name for Eidos (if not provided, auto-generated)",
    )

    args = parser.parse_args()

    # Initialize Qlib
    qlib_dir = str(Path(args.qlib_dir).expanduser())
    qlib.init(provider_uri=qlib_dir, region="cn")
    logger.info(f"Qlib initialized with data directory: {qlib_dir}")

    # Load config for database access
    db_config = None
    if args.enable_eidos:
        try:
            config = load_config(args.config)
            db_config = config.database
        except Exception as e:
            logger.warning(f"Failed to load config for Eidos: {e}")
            logger.warning("Eidos integration will be disabled")
            args.enable_eidos = False

    # Load industry mapping for graph builder
    industry_map = {}
    if db_config is not None:
        try:
            from datetime import datetime
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            industry_map = load_industry_map(db_config, target_date=end_dt)
            logger.info(f"Loaded industry mapping: {len(industry_map)} stocks with industry info")
        except Exception as e:
            logger.warning(f"Failed to load industry map: {e}")
    
    # Create graph builder
    builder = GraphDataBuilder(industry_map)

    # Compare strategies if requested
    if args.compare:
        compare_strategies(
            model_path=args.model_path,
            exit_model_path=args.exit_model_path,
            start_date=args.start_date,
            end_date=args.end_date,
            top_k=args.top_k,
            initial_cash=args.initial_cash,
            exit_scaler_path=args.exit_scaler_path,
            exit_threshold=args.exit_threshold,
            builder=builder,
            device=args.device,
            n_feat=158,
            n_hidden=128,
            n_heads=8,
        )
    else:
        # Run backtest with exit model only (using custom framework)
        logger.info("Creating AsymmetricStrategy with ML Exit...")
        strategy = create_asymmetric_strategy(
            model_path=args.model_path,
            exit_model_path=args.exit_model_path,
            exit_scaler_path=args.exit_scaler_path,
            exit_threshold=args.exit_threshold,
            top_k=args.top_k,
            initial_cash=args.initial_cash,
            builder=builder,
            device=args.device,
            n_feat=158,
            n_hidden=128,
            n_heads=8,
        )

        # Get storage from strategy's position_manager
        storage = strategy.position_manager.storage

        logger.info("Running backtest...")
        results = run_custom_backtest(
            strategy=strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.initial_cash,
            instruments=None,
            storage_backend=storage,
        )

        # Calculate and display metrics
        metrics = calculate_metrics_from_results(results, args.initial_cash)
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Performance Metrics:")
        print(f"{'Metric':<25} {'Value':>15}")
        print("-" * 40)
        print(f"{'Total Return':<25} {metrics['total_return']:>14.2%}")
        print(f"{'Annualized Return':<25} {metrics['annualized_return']:>14.2%}")
        print(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>14.4f}")
        print(f"{'Max Drawdown':<25} {metrics['max_drawdown']:>14.2%}")
        print(f"{'Final Total Value':<25} {metrics['final_total_value']:>14,.2f}")
        print(f"{'Final Positions':<25} {metrics['final_positions']:>15}")
        print(f"{'Total Orders':<25} {metrics['total_orders']:>15}")
        print("-" * 40)
        
        # Print position details
        final_positions = results['positions']
        if final_positions:
            print("\nFinal Positions:")
            for symbol, pos in list(final_positions.items())[:10]:  # Show first 10
                print(f"  {symbol}: {pos.amount:.0f} shares @ {pos.entry_price:.2f}, high={pos.high_price_since_entry:.2f}")
            if len(final_positions) > 10:
                print(f"  ... and {len(final_positions) - 10} more positions")
        
        # Print performance snapshot
        snapshots = results['snapshots']
        if snapshots:
            print("\nPerformance Snapshot (first 5 and last 5 days):")
            for snapshot in snapshots[:5] + snapshots[-5:]:
                print(f"  {snapshot['date']}: total_value={snapshot['total_value']:,.2f}, positions={snapshot['position_count']}")
        
        print("\n" + "=" * 80)

        # Save to Eidos if enabled
        if args.enable_eidos:
            try:
                logger.info("Saving backtest results to Eidos...")
                
                # Load database config
                config = load_config(args.config)
                db_config = config.database
                
                # Create Eidos writer
                writer = EidosBacktestWriter(db_config, schema="eidos")
                
                # Create experiment
                exp_name = args.eidos_exp_name or f"Custom Backtest - {args.start_date} to {args.end_date}"
                if args.exit_model_path:
                    exp_name += " (with ML Exit)"
                
                exp_id = writer.create_experiment_from_backtest(
                    name=exp_name,
                    start_date=pd.to_datetime(args.start_date).date(),
                    end_date=pd.to_datetime(args.end_date).date(),
                    config={
                        "model_path": str(args.model_path) if args.model_path else None,
                        "exit_model_path": str(args.exit_model_path) if args.exit_model_path else None,
                        "top_k": args.top_k,
                        "initial_cash": args.initial_cash,
                        "exit_threshold": args.exit_threshold,
                    },
                    model_type="StructureExpert+MLExit" if args.exit_model_path else "StructureExpert",
                    engine_type="Custom",
                    strategy_name="AsymmetricStrategy",
                )
                
                # Get predictions if available (from buy model)
                predictions = None
                if hasattr(strategy, 'buy_model') and hasattr(strategy.buy_model, 'predictions'):
                    predictions = strategy.buy_model.predictions
                
                # Save results to Eidos
                counts = save_custom_backtest_to_eidos(
                    exp_id=exp_id,
                    writer=writer,
                    results=results,
                    initial_cash=args.initial_cash,
                    predictions=predictions,
                )
                
                logger.info(f"✓ Successfully saved to Eidos (exp_id: {exp_id})")
                logger.info(f"  - Ledger records: {counts.get('ledger', 0)}")
                logger.info(f"  - Trade records: {counts.get('trades', 0)}")
                logger.info(f"  - Model output records: {counts.get('model_outputs', 0)}")
                
            except Exception as e:
                logger.error(f"Failed to save to Eidos: {e}", exc_info=True)
                logger.warning("Continuing without Eidos integration...")

    logger.info("\nBacktest completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
