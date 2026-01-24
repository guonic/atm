#!/usr/bin/env python3
"""
Signal evaluator for Teapot pattern recognition.

Evaluates signals and generates statistics and visualizations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import polars as pl

from nq.analysis.pattern.teapot import TeapotEvaluator, TeapotStatistics
from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from tools.visualization.teapot import BatchPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Teapot signals and generate statistics"
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
        help="Start date for market data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for market data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forward-horizons",
        type=int,
        nargs="+",
        default=[5, 20],
        help="Forward horizons for evaluation (e.g., 5 20)",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="outputs/teapot/visualizations",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="outputs/teapot/reports/evaluation_report.csv",
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use Parquet cache",
    )
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="Strict cache mode",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="storage/teapot_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    db_config = config.database

    # Load signals
    logger.info(f"Loading signals from {args.signals_file}")
    signals = pl.read_csv(args.signals_file)

    if signals.is_empty():
        logger.error("No signals found in file")
        sys.exit(1)

    logger.info(f"Loaded {len(signals)} signals")

    # Load market data
    logger.info(f"Loading market data: {args.start_date} to {args.end_date}")
    data_loader = TeapotDataLoader(
        db_config=db_config,
        schema=args.schema,
        use_cache=args.use_cache,
        cache_dir=Path(args.cache_dir),
        strict_cache=args.strict_cache,
    )

    # Get unique stock codes from signals
    symbols = signals["ts_code"].unique().to_list()

    market_data = data_loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
    )

    if market_data.is_empty():
        logger.error("No market data loaded")
        sys.exit(1)

    logger.info(f"Loaded {len(market_data)} market data records")

    # Evaluate signals
    logger.info("Evaluating signals...")
    evaluator = TeapotEvaluator(forward_horizons=args.forward_horizons)
    evaluation_results = evaluator.compute_forward_returns(
        signals, market_data
    )

    if evaluation_results.is_empty():
        logger.warning("No evaluation results generated")
        sys.exit(0)

    logger.info(f"Evaluated {len(evaluation_results)} signals")

    # Compute statistics
    logger.info("Computing statistics...")
    statistics = TeapotStatistics()
    basic_stats = statistics.compute_basic_stats(evaluation_results)
    stats_by_year = statistics.compute_by_period(
        evaluation_results, period="year"
    )
    stats_by_month = statistics.compute_by_period(
        evaluation_results, period="month"
    )

    # Save evaluation results
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results.write_csv(report_path)
    logger.info(f"Evaluation results saved to {report_path}")

    # Save statistics
    stats_path = report_path.parent / "statistics_summary.json"
    stats_summary = {
        "basic_stats": basic_stats,
        "by_year": stats_by_year.to_dicts() if not stats_by_year.is_empty() else [],
        "by_month": stats_by_month.to_dicts() if not stats_by_month.is_empty() else [],
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Statistics saved to {stats_path}")

    # Generate plots if requested
    if args.generate_plots:
        logger.info("Generating plots...")
        plotter = BatchPlotter(
            output_dir=Path(args.plot_output_dir),
            n_workers=4,
        )
        plot_stats = plotter.generate_all_plots(
            signals=signals,
            market_data=market_data,
            evaluation_results=evaluation_results,
        )
        logger.info(
            f"Generated {plot_stats['total_count']} plots "
            f"({plot_stats['success_count']} success, "
            f"{plot_stats['failure_count']} failure)"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total Signals: {basic_stats.get('total_signals', 0)}")
    for horizon in args.forward_horizons:
        win_rate = basic_stats.get(f"win_rate_t{horizon}", 0)
        avg_return = basic_stats.get(f"avg_return_t{horizon}", 0)
        sharpe = basic_stats.get(f"sharpe_ratio_t{horizon}", 0)
        print(f"\nT+{horizon} Days:")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Return: {avg_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
