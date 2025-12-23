"""
Common evaluation runner for strategy evaluation tools.

Provides reusable evaluation functionality for all strategy evaluation scripts.
"""

import logging
from typing import Any, Dict, Type

from atm.config import DatabaseConfig, load_config
from atm.trading.strategies.base import BaseStrategy
from .batch_evaluator import BatchStrategyEvaluator
from .common_args import validate_dates

logger = logging.getLogger(__name__)


def run_strategy_evaluation(
    strategy_class: Type[BaseStrategy],
    strategy_name: str,
    default_output: str,
    strategy_params: Dict[str, Any],
    args: Any,
) -> int:
    """
    Run strategy evaluation with common workflow.

    Args:
        strategy_class: Strategy class to evaluate.
        strategy_name: Strategy name for logging.
        default_output: Default output file name.
        strategy_params: Strategy-specific parameters dictionary.
        args: Parsed arguments namespace.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Parse dates
    try:
        start_date, end_date = validate_dates(args.start_date, args.end_date)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config from {args.config}: {e}")
        logger.info("Using default database configuration")
        db_config = DatabaseConfig()

    # Initialize evaluator
    evaluator = BatchStrategyEvaluator(
        db_config=db_config,
        schema=args.schema,
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
    )

    # Select stocks
    selected_stocks = evaluator.select_stocks(
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
        num_stocks=args.num_stocks,
        exchange=args.exchange,
        random_seed=args.random_seed,
        skip_market_cap_filter=args.skip_market_cap_filter,
    )

    if not selected_stocks:
        logger.error("No stocks selected. Please check your market cap criteria.")
        return 1

    logger.info(f"Selected {len(selected_stocks)} stocks for evaluation")

    # Determine output file
    output_file = args.output or default_output

    logger.info("=" * 80)
    logger.info(f"{strategy_name} Strategy Evaluation")
    logger.info("=" * 80)
    logger.info(f"Start Date: {start_date.date()}")
    logger.info(f"End Date: {end_date.date()}")
    logger.info(f"Number of Stocks: {len(selected_stocks)}")
    logger.info(f"Strategy Parameters: {strategy_params}")
    logger.info("=" * 80)

    # Run evaluation
    try:
        results = evaluator.evaluate_strategy(
            strategy_class=strategy_class,
            selected_stocks=selected_stocks,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params,
            add_analyzers=True,
            kline_type=args.kline_type,
        )

        # Generate report
        evaluator.generate_summary_report(
            results=results,
            output_file=output_file,
            sort_by=args.sort_by,
            ascending=args.ascending,
        )

        logger.info(f"Evaluation completed. Results saved to {output_file}")
        return 0

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

