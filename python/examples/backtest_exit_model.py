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
4. 使用 MLExitStrategy 进行回测（结合买入和卖出逻辑）
5. 对比有/无退出模型的效果

这是集成了买入和卖出模型的完整回测脚本。
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

from examples.backtest_structure_expert import (
    DEFAULT_INITIAL_CASH,
    DEFAULT_TOP_K,
    generate_predictions,
    normalize_index_code,
    run_backtest,
    load_model,
    print_results,
    PortfolioMetrics,
    PerformanceMetrics,
)
from tools.qlib.train.structure_expert import GraphDataBuilder
from nq.utils.industry import load_industry_map

# Import Eidos integration
from nq.analysis.backtest.eidos_integration import EidosBacktestWriter
from nq.analysis.backtest.eidos_structure_expert import (
    save_structure_expert_backtest_to_eidos,
)
from nq.config import load_config

from examples.backtest_structure_expert_ml_exit import MLExitStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_ml_exit_strategy(
    predictions: pd.DataFrame,
    exit_model_path: str,
    exit_scaler_path: str = None,
    exit_threshold: float = 0.65,
    top_k: int = DEFAULT_TOP_K,
    buffer_ratio: float = 0.15,
    use_ml_exit: bool = True,
) -> MLExitStrategy:
    """
    Create MLExitStrategy with exit model.
    
    Args:
        predictions: Predictions DataFrame with MultiIndex (datetime, instrument) and 'score' column.
        exit_model_path: Path to trained exit model.
        exit_scaler_path: Path to feature scaler (if None, auto-generated).
        exit_threshold: Risk probability threshold for exit signal.
        top_k: Number of top stocks to select.
        buffer_ratio: Buffer ratio for reducing turnover.
        use_ml_exit: Whether to use ML exit model.
    
    Returns:
        MLExitStrategy instance.
    """
    logger.info(f"Creating MLExitStrategy with exit model: {exit_model_path}")
    
    strategy = MLExitStrategy(
        signal=predictions,
        topk=top_k,
        buffer_ratio=buffer_ratio,
        exit_model_path=exit_model_path,
        exit_scaler_path=exit_scaler_path,
        exit_threshold=exit_threshold,
        use_ml_exit=use_ml_exit,
    )
    
    return strategy


def compare_strategies(
    predictions: pd.DataFrame,
    exit_model_path: str,
    start_date: str,
    end_date: str,
    top_k: int = DEFAULT_TOP_K,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    benchmark: str = None,
    enable_eidos: bool = False,
    eidos_exp_name: str = None,
    db_config = None,
    model = None,
    builder = None,
    device: str = "cuda",
):
    """
    Compare backtest results with and without exit model.
    
    Args:
        predictions: Predictions DataFrame.
        exit_model_path: Path to exit model.
        start_date: Start date.
        end_date: End date.
        top_k: Number of top stocks.
        initial_cash: Initial cash.
        benchmark: Benchmark code.
    """
    from examples.backtest_structure_expert import RefinedTopKStrategy, create_portfolio_strategy
    
    logger.info("=" * 80)
    logger.info("Comparing strategies: RefinedTopKStrategy vs MLExitStrategy")
    logger.info("=" * 80)
    
    # Strategy 1: Without exit model (baseline)
    logger.info("\n1. Running backtest WITHOUT exit model (baseline)...")
    baseline_strategy = create_portfolio_strategy(
        predictions=predictions,
        strategy_class=RefinedTopKStrategy,
        top_k=top_k,
        buffer_ratio=0.15,
    )
    
    baseline_results = run_backtest(
        strategy=baseline_strategy,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        benchmark=benchmark,
    )
    
    # Strategy 2: With exit model
    logger.info("\n2. Running backtest WITH exit model...")
    ml_exit_strategy = create_ml_exit_strategy(
        predictions=predictions,
        exit_model_path=exit_model_path,
        top_k=top_k,
        buffer_ratio=0.15,
        use_ml_exit=True,
    )
    
    ml_exit_results = run_backtest(
        strategy=ml_exit_strategy,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        benchmark=benchmark,
    )
    
    # Compare results - show concise comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Extract metrics for comparison
    baseline_portfolio = PortfolioMetrics(baseline_results.get("portfolio_metric"))
    baseline_metric_df = baseline_portfolio.get_dataframe() if baseline_portfolio.has_data() else None
    baseline_perf = PerformanceMetrics(baseline_metric_df) if baseline_metric_df is not None and not baseline_metric_df.empty else None
    
    ml_exit_portfolio = PortfolioMetrics(ml_exit_results.get("portfolio_metric"))
    ml_exit_metric_df = ml_exit_portfolio.get_dataframe() if ml_exit_portfolio.has_data() else None
    ml_exit_perf = PerformanceMetrics(ml_exit_metric_df) if ml_exit_metric_df is not None and not ml_exit_metric_df.empty else None
    
    # Display concise comparison table
    print("\n📊 Key Metrics Comparison:")
    print(f"{'Metric':<25} {'Baseline':>15} {'ML Exit':>15} {'Difference':>15}")
    print("-" * 70)
    
    # Total Return
    baseline_total_return = baseline_perf.total_return if baseline_perf and baseline_perf.total_return is not None else None
    ml_exit_total_return = ml_exit_perf.total_return if ml_exit_perf and ml_exit_perf.total_return is not None else None
    if baseline_total_return is not None and ml_exit_total_return is not None:
        diff = ml_exit_total_return - baseline_total_return
        print(f"{'Total Return':<25} {baseline_total_return:>14.2%} {ml_exit_total_return:>14.2%} {diff:>+14.2%}")
    else:
        baseline_str = f"{baseline_total_return:.2%}" if baseline_total_return is not None else "N/A"
        ml_exit_str = f"{ml_exit_total_return:.2%}" if ml_exit_total_return is not None else "N/A"
        print(f"{'Total Return':<25} {baseline_str:>15} {ml_exit_str:>15} {'N/A':>15}")
    
    # Annual Return
    baseline_annual = baseline_perf.annualized_return if baseline_perf and baseline_perf.annualized_return is not None else None
    ml_exit_annual = ml_exit_perf.annualized_return if ml_exit_perf and ml_exit_perf.annualized_return is not None else None
    if baseline_annual is not None and ml_exit_annual is not None:
        diff = ml_exit_annual - baseline_annual
        print(f"{'Annualized Return':<25} {baseline_annual:>14.2%} {ml_exit_annual:>14.2%} {diff:>+14.2%}")
    else:
        baseline_str = f"{baseline_annual:.2%}" if baseline_annual is not None else "N/A"
        ml_exit_str = f"{ml_exit_annual:.2%}" if ml_exit_annual is not None else "N/A"
        print(f"{'Annualized Return':<25} {baseline_str:>15} {ml_exit_str:>15} {'N/A':>15}")
    
    # Sharpe Ratio
    baseline_sharpe = baseline_perf.sharpe_ratio if baseline_perf and baseline_perf.sharpe_ratio is not None else None
    ml_exit_sharpe = ml_exit_perf.sharpe_ratio if ml_exit_perf and ml_exit_perf.sharpe_ratio is not None else None
    if baseline_sharpe is not None and ml_exit_sharpe is not None:
        diff = ml_exit_sharpe - baseline_sharpe
        print(f"{'Sharpe Ratio':<25} {baseline_sharpe:>14.4f} {ml_exit_sharpe:>14.4f} {diff:>+14.4f}")
    else:
        baseline_str = f"{baseline_sharpe:.4f}" if baseline_sharpe is not None else "N/A"
        ml_exit_str = f"{ml_exit_sharpe:.4f}" if ml_exit_sharpe is not None else "N/A"
        print(f"{'Sharpe Ratio':<25} {baseline_str:>15} {ml_exit_str:>15} {'N/A':>15}")
    
    # Max Drawdown
    baseline_max_dd = baseline_perf.max_drawdown if baseline_perf and baseline_perf.max_drawdown is not None else None
    ml_exit_max_dd = ml_exit_perf.max_drawdown if ml_exit_perf and ml_exit_perf.max_drawdown is not None else None
    if baseline_max_dd is not None and ml_exit_max_dd is not None:
        diff = ml_exit_max_dd - baseline_max_dd
        print(f"{'Max Drawdown':<25} {baseline_max_dd:>14.2%} {ml_exit_max_dd:>14.2%} {diff:>+14.2%}")
    else:
        baseline_str = f"{baseline_max_dd:.2%}" if baseline_max_dd is not None else "N/A"
        ml_exit_str = f"{ml_exit_max_dd:.2%}" if ml_exit_max_dd is not None else "N/A"
        print(f"{'Max Drawdown':<25} {baseline_str:>15} {ml_exit_str:>15} {'N/A':>15}")
    
    # Win Rate
    baseline_position = baseline_portfolio.get_position_details()
    ml_exit_position = ml_exit_portfolio.get_position_details()
    if baseline_position and ml_exit_position:
        from examples.backtest_structure_expert import TradingStatistics
        baseline_stats = TradingStatistics(baseline_position)
        ml_exit_stats = TradingStatistics(ml_exit_position)
        if baseline_stats.total_trades > 0 and ml_exit_stats.total_trades > 0:
            diff = ml_exit_stats.win_rate - baseline_stats.win_rate
            print(f"{'Win Rate':<25} {baseline_stats.win_rate:>14.2f}% {ml_exit_stats.win_rate:>14.2f}% {diff:>+14.2f}%")
    
    print("-" * 70)
    
    # Summary
    if baseline_total_return is not None and ml_exit_total_return is not None:
        improvement = ml_exit_total_return - baseline_total_return
        print(f"\n📈 Overall Improvement: {improvement:+.2%}")
        if improvement > 0:
            print("   ✅ ML Exit Strategy performs better")
        elif improvement < 0:
            print("   ⚠️  Baseline Strategy performs better")
        else:
            print("   ➡️  Both strategies perform equally")
    else:
        improvement = 0.0
        print(f"\n📈 Overall Improvement: N/A (unable to calculate)")
    
    # Option to show full reports
    print("\n💡 Tip: To see full detailed reports, remove --compare flag")
    
    # Save to Eidos if enabled - use same format as backtest_structure_expert.py
    if enable_eidos and db_config and model and builder:
        try:
            from datetime import date as date_type
            from nq.analysis.backtest.qlib_types import QlibBacktestResult
            
            logger.info("\nSaving ML Exit Strategy results to Eidos...")
            writer = EidosBacktestWriter(db_config)
            
            # Generate experiment name if not provided
            exp_name = eidos_exp_name
            if exp_name is None:
                exp_name = (
                    f"ML Exit Strategy Comparison - "
                    f"TopK{top_k} - {start_date}_{end_date}"
                )
            
            # Create experiment
            exp_id = writer.create_experiment_from_backtest(
                name=exp_name,
                start_date=date_type.fromisoformat(start_date),
                end_date=date_type.fromisoformat(end_date),
                config={
                    "strategy": "MLExitStrategy",
                    "top_k": top_k,
                    "exit_model_path": exit_model_path,
                    "baseline_return": baseline_total_return if baseline_total_return is not None else 0.0,
                    "ml_exit_return": ml_exit_total_return if ml_exit_total_return is not None else 0.0,
                    "improvement": improvement,
                },
                model_type="StructureExpert_MLExit",
                strategy_name="MLExitStrategy",
            )
            
            logger.info(f"Created experiment: {exp_id}")
            
            # Convert results to QlibBacktestResult format
            portfolio_metric = ml_exit_results.get("portfolio_metric")
            indicator = ml_exit_results.get("indicator")
            
            if portfolio_metric and indicator:
                qlib_result = QlibBacktestResult.from_qlib_dict_output(
                    portfolio_metric_dict=portfolio_metric,
                    indicator_dict=indicator,
                )
                
                # Save results
                save_structure_expert_backtest_to_eidos(
                    exp_id=exp_id,
                    writer=writer,
                    qlib_result=qlib_result,
                    predictions=predictions,
                    initial_cash=initial_cash,
                    builder=builder,
                    model=model,
                    device=torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu"),
                    embeddings_dict=None,
                    strategy_instance=ml_exit_strategy,
                )
                
                # Finalize experiment - use same logic as backtest_structure_expert.py
                metrics_summary = {}
                
                # Use standard structure (reuse the one created above)
                metric_df = qlib_result.portfolio_metrics.metric_df
                
                if metric_df is not None and not metric_df.empty:
                    if "return" in metric_df.columns:
                        returns = metric_df["return"]
                        metrics_summary["return"] = float(returns.mean())
                        metrics_summary["sharpe"] = (
                            float(returns.mean() / returns.std() * (252 ** 0.5))
                            if returns.std() > 0
                            else 0.0
                        )
                        # Calculate max drawdown
                        cum_returns = (1 + returns).cumprod()
                        running_max = cum_returns.expanding().max()
                        drawdown = (cum_returns - running_max) / running_max
                        metrics_summary["max_drawdown"] = float(drawdown.min())
                
                # Add comparison metrics
                if baseline_total_return is not None:
                    metrics_summary["baseline_return"] = baseline_total_return
                if improvement is not None:
                    metrics_summary["improvement"] = improvement
                
                writer.finalize_experiment(exp_id, metrics_summary=metrics_summary)
                
                logger.info(f"✓ Eidos integration completed. Experiment ID: {exp_id}")
        except Exception as e:
            logger.error(f"Failed to save to Eidos: {e}", exc_info=True)
    
    return {
        "baseline": baseline_results,
        "ml_exit": ml_exit_results,
        "improvement": improvement,
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

    # Load model
    logger.info(f"Loading Structure Expert model from {args.model_path}")
    # Detect model type from file name or use defaults
    # For directional model, we need to check the actual model type
    model = load_model(
        model_path=args.model_path,
        n_feat=158,  # Alpha158 features
        n_hidden=128,
        n_heads=8,
        device=args.device,
    )
    
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

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_predictions(
        model=model,
        builder=builder,
        start_date=args.start_date,
        end_date=args.end_date,
        device=args.device,
        qlib_dir=qlib_dir,
    )

    if predictions.empty:
        logger.error("No predictions generated. Please check your model and data.")
        return 1

    logger.info(f"Generated predictions for {len(predictions)} stock-days")

    # Normalize benchmark
    benchmark = None
    if args.benchmark:
        benchmark = normalize_index_code(args.benchmark)

    # Compare strategies if requested
    if args.compare:
        compare_strategies(
            predictions=predictions,
            exit_model_path=args.exit_model_path,
            start_date=args.start_date,
            end_date=args.end_date,
            top_k=args.top_k,
            initial_cash=args.initial_cash,
            benchmark=benchmark,
            enable_eidos=args.enable_eidos,
            eidos_exp_name=args.eidos_exp_name,
            db_config=db_config,
            model=model,
            builder=builder,
            device=args.device,
        )
    else:
        # Run backtest with exit model only
        logger.info("Creating MLExitStrategy...")
        strategy = create_ml_exit_strategy(
            predictions=predictions,
            exit_model_path=args.exit_model_path,
            exit_scaler_path=args.exit_scaler_path,
            exit_threshold=args.exit_threshold,
            top_k=args.top_k,
            buffer_ratio=args.buffer_ratio,
            use_ml_exit=True,
        )

        logger.info("Running backtest...")
        results = run_backtest(
            strategy=strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.initial_cash,
            benchmark=benchmark,
        )

        # Display results - use same format as backtest_structure_expert.py
        print_results(results)

        # Save to Eidos if enabled
        if args.enable_eidos and db_config and model and builder:
            try:
                from datetime import date as date_type
                from nq.analysis.backtest.qlib_types import QlibBacktestResult
                
                logger.info("\nSaving results to Eidos...")
                writer = EidosBacktestWriter(db_config)
                
                # Generate experiment name if not provided
                exp_name = args.eidos_exp_name
                if exp_name is None:
                    exp_name = (
                        f"ML Exit Strategy - "
                        f"TopK{args.top_k} - {args.start_date}_{args.end_date}"
                    )
                
                # Create experiment
                exp_id = writer.create_experiment_from_backtest(
                    name=exp_name,
                    start_date=date_type.fromisoformat(args.start_date),
                    end_date=date_type.fromisoformat(args.end_date),
                    config={
                        "strategy": "MLExitStrategy",
                        "top_k": args.top_k,
                        "exit_model_path": args.exit_model_path,
                        "exit_threshold": args.exit_threshold,
                        "buffer_ratio": args.buffer_ratio,
                    },
                    model_type="StructureExpert_MLExit",
                    strategy_name="MLExitStrategy",
                )
                
                logger.info(f"Created experiment: {exp_id}")
                
                # Convert results to QlibBacktestResult format
                portfolio_metric = results.get("portfolio_metric")
                indicator = results.get("indicator")
                
                if portfolio_metric and indicator:
                    qlib_result = QlibBacktestResult.from_qlib_dict_output(
                        portfolio_metric_dict=portfolio_metric,
                        indicator_dict=indicator,
                    )
                    
                    # Save results
                    save_structure_expert_backtest_to_eidos(
                        exp_id=exp_id,
                        writer=writer,
                        qlib_result=qlib_result,
                        predictions=predictions,
                        initial_cash=args.initial_cash,
                        builder=builder,
                        model=model,
                        device=torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"),
                        embeddings_dict=None,
                        strategy_instance=strategy,
                    )
                    
                    # Finalize experiment - use same logic as backtest_structure_expert.py
                    metrics_summary = {}
                    
                    # Use standard structure (reuse the one created above)
                    metric_df = qlib_result.portfolio_metrics.metric_df
                    
                    if metric_df is not None and not metric_df.empty:
                        if "return" in metric_df.columns:
                            returns = metric_df["return"]
                            metrics_summary["return"] = float(returns.mean())
                            metrics_summary["sharpe"] = (
                                float(returns.mean() / returns.std() * (252 ** 0.5))
                                if returns.std() > 0
                                else 0.0
                            )
                            # Calculate max drawdown
                            cum_returns = (1 + returns).cumprod()
                            running_max = cum_returns.expanding().max()
                            drawdown = (cum_returns - running_max) / running_max
                            metrics_summary["max_drawdown"] = float(drawdown.min())
                    
                    writer.finalize_experiment(exp_id, metrics_summary=metrics_summary)
                    
                    logger.info(f"✓ Eidos integration completed. Experiment ID: {exp_id}")
            except Exception as e:
                logger.error(f"Failed to save to Eidos: {e}", exc_info=True)

    logger.info("\nBacktest completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
