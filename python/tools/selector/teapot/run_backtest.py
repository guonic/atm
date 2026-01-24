#!/usr/bin/env python3
"""
Backtest runner for Teapot pattern recognition strategy.

Executes backtests and generates reports.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.analysis.backtest import TeapotAnalyzer, TeapotBacktester
from nq.config import DatabaseConfig, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run backtest for Teapot strategy"
    )
    parser.add_argument(
        "--signals-file",
        type=str,
        required=True,
        help="Signals CSV file path",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        help="Strategy config file path (YAML)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1000000.0,
        help="Initial cash amount",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/backtest/backtest_results.csv",
        help="Output path for backtest results",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML report",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Config file path",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load strategy config if provided
    strategy_params = {
        "exit_strategy": "fixed_days",
        "holding_days": 20,
        "target_return": 0.15,
        "stop_loss": -0.10,
        "position_size": 0.1,
        "max_positions": 10,
        "use_ml_score": False,
        "ml_score_threshold": 0.7,
    }

    if args.strategy_config:
        import yaml

        with open(args.strategy_config, "r", encoding="utf-8") as f:
            strategy_config = yaml.safe_load(f)
            strategy_params.update(strategy_config)

    # Initialize backtester
    backtester = TeapotBacktester(
        signals_file=Path(args.signals_file),
        data_source="database",  # or "qlib"
        initial_cash=args.initial_cash,
    )

    # Run backtest
    logger.info(f"Running backtest: {args.start_date} to {args.end_date}")
    result = backtester.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_params=strategy_params,
    )

    # Analyze results
    analyzer = TeapotAnalyzer()
    analysis = analyzer.analyze_backtest(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save portfolio value
    result.portfolio_value.to_csv(
        output_path.parent / "portfolio_value.csv"
    )

    # Save trades if available
    if not result.trades.empty:
        result.trades.to_csv(output_path.parent / "trades.csv")

    # Save metrics
    metrics_path = output_path.parent / "performance_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    logger.info(f"Backtest results saved to {output_path.parent}")

    # Print summary
    print("\n" + "=" * 60)
    print("Backtest Summary")
    print("=" * 60)
    print(f"Total Return: {analysis.get('total_return', 0):.2%}")
    print(f"Annual Return: {analysis.get('annual_return', 0):.2%}")
    print(f"Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {analysis.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {analysis.get('win_rate', 0):.2%}")
    print(f"Total Trades: {analysis.get('total_trades', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
