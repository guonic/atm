"""
Batch strategy evaluator.

Provides functionality to evaluate strategies on multiple stocks and generate
comprehensive evaluation reports.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Type

import pandas as pd

from nq.config import DatabaseConfig
from nq.trading.selector.fundamental_selector import FundamentalSelector
from nq.trading.strategies.base import BaseStrategy
from nq.trading.strategies.strategy_runner import StrategyRunner

from .base import BacktestResult
from .evaluator import BacktestEvaluator
from .strategy_backtest import StrategyBacktester

logger = logging.getLogger(__name__)


class BatchStrategyEvaluator:
    """
    Batch evaluator for trading strategies.

    Evaluates strategies on multiple stocks and generates comprehensive reports.
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        schema: str = "quant",
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        """
        Initialize batch strategy evaluator.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            initial_cash: Initial cash amount for backtesting.
            commission: Commission rate (default: 0.001 = 0.1%).
            slippage: Slippage rate (default: 0.0).
        """
        self.db_config = db_config
        self.schema = schema
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.backtester = StrategyBacktester(
            db_config=db_config,
            schema=schema,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
        )

    def select_stocks(
        self,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        num_stocks: int = 100,
        exchange: Optional[str] = None,
        random_seed: Optional[int] = 42,
        skip_market_cap_filter: bool = False,
    ) -> List[str]:
        """
        Select stocks based on market capitalization.

        Args:
            min_market_cap: Minimum market cap (in 10K CNY). If None and skip_market_cap_filter=False, will use all stocks.
            max_market_cap: Maximum market cap (in 10K CNY). If None and skip_market_cap_filter=False, will use all stocks.
            num_stocks: Number of stocks to select (default: 100).
            exchange: Exchange code (SSE/SZSE/BSE). If None, select from all exchanges.
            random_seed: Random seed for reproducibility (default: 42).
            skip_market_cap_filter: If True, skip market cap filter and use all stocks.

        Returns:
            List of selected stock codes.
        """
        if skip_market_cap_filter:
            logger.info("Skipping market cap filter, selecting from all stocks")
            from nq.repo.stock_repo import StockBasicRepo
            stock_repo = StockBasicRepo(self.db_config, self.schema)
            if exchange:
                candidates = stock_repo.get_by_exchange(exchange=exchange, list_status="L")
            else:
                candidates = stock_repo.get_by_exchange(list_status="L")
            
            # Filter by listing status
            candidates = [s for s in candidates if s.is_listed]
            
            logger.info(f"Found {len(candidates)} candidate stocks")
            
            if len(candidates) < num_stocks:
                logger.warning(
                    f"Only found {len(candidates)} stocks, requested {num_stocks}"
                )
                return [s.ts_code for s in candidates]
            
            # Randomly select
            if random_seed is not None:
                random.seed(random_seed)
            selected = random.sample(candidates, num_stocks)
            return [s.ts_code for s in selected]
        
        # Use market cap filter; if parameters missing, fallback to skip filter
        if min_market_cap is None or max_market_cap is None:
            logger.warning(
                "min_market_cap/max_market_cap not provided, fallback to skip market cap filter"
            )
            return self.select_stocks(
                min_market_cap=None,
                max_market_cap=None,
                num_stocks=num_stocks,
                exchange=exchange,
                random_seed=random_seed,
                skip_market_cap_filter=True,
            )
        
        # Convert to billions for display
        # total_mv unit is 10K CNY (万元)
        # 1亿 = 10000万, so to convert 万元 to 亿元: divide by 10000
        min_market_cap_b = min_market_cap / 10000  # Convert 万元 to 亿元
        max_market_cap_b = max_market_cap / 10000  # Convert 万元 to 亿元

        logger.info(
            f"Selecting stocks with market cap between {min_market_cap_b:.0f}B and "
            f"{max_market_cap_b:.0f}B CNY (in 10K CNY: {min_market_cap:.0f} to {max_market_cap:.0f})"
        )

        selector = FundamentalSelector(
            db_config=self.db_config,
            schema=self.schema,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
        )

        result = selector.select(exchange=exchange)
        selected_stocks = result.selected_stocks

        # Debug: Check if we have finance data
        if len(selected_stocks) == 0:
            logger.error(
                "=" * 80
            )
            logger.error(
                "No stocks found matching criteria. Possible reasons:"
            )
            logger.error(
                "1) No finance data in database (stock_finance_basic table is empty)"
            )
            logger.error(
                "2) Market cap values in database are outside the specified range"
            )
            logger.error(
                "=" * 80
            )
            
            # Try to get some sample finance data to debug
            from nq.repo.stock_repo import StockFinanceBasicRepo
            from sqlalchemy import text
            finance_repo = StockFinanceBasicRepo(self.db_config, self.schema)
            engine = finance_repo._get_engine()
            table_name = finance_repo._get_full_table_name()
            
            try:
                with engine.connect() as conn:
                    # Check if table exists and has data
                    count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                    count_result = conn.execute(count_query)
                    total_count = count_result.fetchone()[0]
                    
                    if total_count == 0:
                        logger.error(
                            f"\n❌ Table {table_name} is empty. "
                            f"You need to sync finance data first."
                        )
                        logger.error(
                            "\nTo sync finance data, you can use Tushare API:"
                        )
                        logger.error(
                            "  - Use Tushare's 'daily_basic' or 'fina_indicator' API"
                        )
                        logger.error(
                            "  - Or use 'stock_company' API for company basic info"
                        )
                    else:
                        logger.info(f"\nFound {total_count} records in {table_name}")
                        
                        # Get sample finance data with market cap
                        query = text(
                            f"SELECT ts_code, total_mv FROM {table_name} "
                            f"WHERE total_mv IS NOT NULL ORDER BY total_mv DESC LIMIT 10"
                        )
                        result = conn.execute(query)
                        sample_rows = result.fetchall()
                        
                        if sample_rows:
                            logger.info("Sample finance data (top 10 by market cap):")
                            for row in sample_rows:
                                ts_code, total_mv = row
                                total_mv_b = float(total_mv) / 10000 if total_mv else None  # Convert 万元 to 亿元
                                logger.info(
                                    f"  {ts_code}: total_mv={total_mv} (10K CNY) "
                                    f"= {total_mv_b:.2f}B CNY" if total_mv_b else f"(None)"
                                )
                            
                            # Check range
                            min_mv = min(float(row[1]) for row in sample_rows if row[1])
                            max_mv = max(float(row[1]) for row in sample_rows if row[1])
                            logger.info(
                                f"\nMarket cap range in database: "
                                f"{min_mv/10000:.2f}B to {max_mv/10000:.2f}B CNY"
                            )
                            logger.info(
                                f"Requested range: {min_market_cap/10000:.2f}B to {max_market_cap/10000:.2f}B CNY"
                            )
                        else:
                            logger.warning(
                                f"Table {table_name} has {total_count} records but none have total_mv data."
                            )
            except Exception as e:
                logger.error(f"Failed to query finance data: {e}", exc_info=True)

        if len(selected_stocks) < num_stocks:
            logger.warning(
                f"Only found {len(selected_stocks)} stocks matching criteria, "
                f"requested {num_stocks}"
            )
            return selected_stocks

        # Randomly select num_stocks from the pool
        if random_seed is not None:
            random.seed(random_seed)
        selected = random.sample(selected_stocks, num_stocks)
        logger.info(
            f"Randomly selected {len(selected)} stocks from {len(selected_stocks)} candidates"
        )

        return selected

    def evaluate_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        selected_stocks: List[str],
        start_date: datetime,
        end_date: datetime,
        strategy_params: Optional[Dict] = None,
        add_analyzers: bool = True,
        kline_type: str = "day",
        run_id: Optional[str] = None,
    ) -> List[BacktestResult]:
        """
        Evaluate strategy on multiple stocks.

        Args:
            strategy_class: Strategy class to evaluate.
            selected_stocks: List of stock codes to evaluate.
            start_date: Backtest start date.
            end_date: Backtest end date.
            strategy_params: Strategy parameters dictionary.
            add_analyzers: Whether to add analyzers (default: True).
            kline_type: K-line type (default: 'day').
            run_id: Backtest run ID for signal tracking (default: None).

        Returns:
            List of BacktestResult objects.
        """
        logger.info(f"Evaluating {strategy_class.__name__} on {len(selected_stocks)} stocks")
        logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")

        results = []

        for idx, ts_code in enumerate(selected_stocks, 1):
            logger.info(f"[{idx}/{len(selected_stocks)}] Running backtest for {ts_code}")

            try:
                result = self.backtester.run(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_class=strategy_class,
                    strategy_params=strategy_params,
                    add_analyzers=add_analyzers,
                    kline_type=kline_type,
                    run_id=run_id,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error running backtest for {ts_code}: {e}", exc_info=True)
                # Create a failed result
                failed_result = BacktestResult(
                    ts_code=ts_code,
                    backtest_date=datetime.now(),
                    start_date=start_date,
                    end_date=end_date,
                    predictions=pd.DataFrame(),
                    actuals=pd.DataFrame(),
                    metrics={},
                    metadata={"error": str(e)},
                )
                results.append(failed_result)

            # Log progress every 10 stocks
            if idx % 10 == 0:
                successful = sum(1 for r in results if "error" not in r.metadata)
                logger.info(f"Progress: {idx}/{len(selected_stocks)} ({successful} successful)")

        return results

    def generate_summary_report(
        self,
        results: List[BacktestResult],
        output_file: Optional[str] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Generate summary report from evaluation results.

        Args:
            results: List of BacktestResult objects.
            output_file: Optional output file path for saving report.
            sort_by: Column name to sort by (e.g., 'total_return', 'sharpe_ratio', 'max_drawdown').
            ascending: Whether to sort in ascending order (default: False, descending).

        Returns:
            DataFrame containing summary statistics.
        """
        logger.info("=" * 80)
        logger.info("Strategy Evaluation Report")
        logger.info("=" * 80)

        # Filter out errors
        valid_results = [r for r in results if "error" not in r.metadata]
        error_count = len(results) - len(valid_results)

        logger.info(f"\nTotal Stocks Evaluated: {len(results)}")
        logger.info(f"Successful Backtests: {len(valid_results)}")
        logger.info(f"Failed Backtests: {error_count}")

        if len(valid_results) == 0:
            logger.warning("No valid results to analyze")
            return pd.DataFrame()

        # Helper function to round float values to 2 decimal places
        def round_float(value, decimals: int = 2):
            """Round float value to specified decimal places."""
            if value is None:
                return None
            try:
                return round(float(value), decimals)
            except (ValueError, TypeError):
                return value

        # Generate summary DataFrame
        summary_df = pd.DataFrame()
        for result in valid_results:
            row = {
                "ts_code": result.ts_code,
                "start_date": result.start_date.date(),
                "end_date": result.end_date.date(),
            }

            # Extract metrics from result.metrics (which contains analyzer results)
            if result.metrics:
                # Extract basic results (rounded to 2 decimals)
                if "initial_value" in result.metrics:
                    row["initial_value"] = round_float(result.metrics["initial_value"], decimals=2)
                if "final_value" in result.metrics:
                    row["final_value"] = round_float(result.metrics["final_value"], decimals=2)
                if "total_return" in result.metrics:
                    row["total_return"] = round_float(result.metrics["total_return"], decimals=2)

                # Extract trade statistics (integers, no rounding needed)
                if "total_trades" in result.metrics:
                    row["total_trades"] = result.metrics["total_trades"]
                if "won_trades" in result.metrics:
                    row["won_trades"] = result.metrics["won_trades"]
                if "lost_trades" in result.metrics:
                    row["lost_trades"] = result.metrics["lost_trades"]

                # Extract Sharpe ratio (rounded to 2 decimals)
                if "sharpe_ratio" in result.metrics:
                    row["sharpe_ratio"] = round_float(result.metrics["sharpe_ratio"], decimals=2)

                # Extract drawdown (rounded to 2 decimals)
                if "max_drawdown" in result.metrics:
                    row["max_drawdown"] = round_float(result.metrics["max_drawdown"], decimals=2)
                if "max_drawdown_len" in result.metrics:
                    row["max_drawdown_len"] = round_float(result.metrics["max_drawdown_len"], decimals=0)

            summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

        # Sort DataFrame if sort_by is specified
        if sort_by and sort_by in summary_df.columns:
            summary_df = summary_df.sort_values(by=sort_by, ascending=ascending, na_position="last")
            logger.info(f"\nResults sorted by '{sort_by}' ({'ascending' if ascending else 'descending'})")
        elif sort_by:
            logger.warning(f"Sort column '{sort_by}' not found in results. Available columns: {list(summary_df.columns)}")

        # Print statistics
        self._print_statistics(summary_df, sort_by=sort_by)

        # Save to file if requested
        if output_file:
            summary_df.to_csv(output_file, index=False)
            logger.info(f"\nResults saved to {output_file}")

        logger.info("\n" + "=" * 80)

        return summary_df

    def _print_statistics(self, summary_df: pd.DataFrame, sort_by: Optional[str] = None) -> None:
        """
        Print evaluation statistics.

        Args:
            summary_df: DataFrame containing evaluation results.
            sort_by: Column name used for sorting (for display purposes).
        """
        # Return statistics
        if "total_return" in summary_df.columns:
            returns = summary_df["total_return"].dropna()
            if len(returns) > 0:
                logger.info("\n" + "=" * 80)
                logger.info("Return Statistics")
                logger.info("=" * 80)
                logger.info(f"Mean Return: {returns.mean():.2f}%")
                logger.info(f"Median Return: {returns.median():.2f}%")
                logger.info(f"Std Return: {returns.std():.2f}%")
                logger.info(f"Min Return: {returns.min():.2f}%")
                logger.info(f"Max Return: {returns.max():.2f}%")

                # Win rate
                positive_returns = (returns > 0).sum()
                win_rate = positive_returns / len(returns) * 100
                logger.info(f"Win Rate: {win_rate:.2f}% ({positive_returns}/{len(returns)})")

        # Sharpe ratio statistics
        if "sharpe_ratio" in summary_df.columns:
            sharpe_ratios = summary_df["sharpe_ratio"].dropna()
            if len(sharpe_ratios) > 0:
                logger.info("\n" + "=" * 80)
                logger.info("Sharpe Ratio Statistics")
                logger.info("=" * 80)
                logger.info(f"Mean Sharpe Ratio: {sharpe_ratios.mean():.2f}")
                logger.info(f"Median Sharpe Ratio: {sharpe_ratios.median():.2f}")
                logger.info(f"Std Sharpe Ratio: {sharpe_ratios.std():.2f}")

        # Drawdown statistics
        if "max_drawdown" in summary_df.columns:
            max_drawdowns = summary_df["max_drawdown"].dropna()
            if len(max_drawdowns) > 0:
                logger.info("\n" + "=" * 80)
                logger.info("Drawdown Statistics")
                logger.info("=" * 80)
                logger.info(f"Mean Max Drawdown: {max_drawdowns.mean():.2f}%")
                logger.info(f"Median Max Drawdown: {max_drawdowns.median():.2f}%")
                logger.info(f"Worst Drawdown: {max_drawdowns.min():.2f}%")

        # Trade statistics
        if "total_trades" in summary_df.columns:
            total_trades = summary_df["total_trades"].dropna()
            if len(total_trades) > 0:
                logger.info("\n" + "=" * 80)
                logger.info("Trade Statistics")
                logger.info("=" * 80)
                logger.info(f"Mean Total Trades: {total_trades.mean():.1f}")
                if "won_trades" in summary_df.columns:
                    won_trades = summary_df["won_trades"].dropna()
                    logger.info(f"Mean Won Trades: {won_trades.mean():.1f}")
                if "lost_trades" in summary_df.columns:
                    lost_trades = summary_df["lost_trades"].dropna()
                    logger.info(f"Mean Lost Trades: {lost_trades.mean():.1f}")

        # Top and bottom performers
        # Use sort_by if specified, otherwise use total_return
        sort_column = sort_by if sort_by and sort_by in summary_df.columns else "total_return"
        
        if sort_column in summary_df.columns:
            logger.info("\n" + "=" * 80)
            logger.info(f"Top 10 Performers (by {sort_column.replace('_', ' ').title()})")
            logger.info("=" * 80)

            top_performers = summary_df.nlargest(10, sort_column)
            display_cols = ["ts_code", sort_column]
            if "final_value" in summary_df.columns and "final_value" not in display_cols:
                display_cols.append("final_value")
            if "total_return" in summary_df.columns and "total_return" not in display_cols:
                display_cols.append("total_return")
            
            for _, row in top_performers[display_cols].iterrows():
                col_value = row[sort_column]
                if isinstance(col_value, (int, float)):
                    col_str = f"{col_value:.2f}"
                else:
                    col_str = str(col_value)
                
                info_parts = [f"{row['ts_code']}: {sort_column}={col_str}"]
                if "final_value" in display_cols:
                    info_parts.append(f"Final Value: {row['final_value']:.2f}")
                if "total_return" in display_cols and sort_column != "total_return":
                    info_parts.append(f"Return: {row['total_return']:.2f}%")
                
                logger.info(" | ".join(info_parts))

            logger.info("\n" + "=" * 80)
            logger.info(f"Bottom 10 Performers (by {sort_column.replace('_', ' ').title()})")
            logger.info("=" * 80)

            bottom_performers = summary_df.nsmallest(10, sort_column)
            for _, row in bottom_performers[display_cols].iterrows():
                col_value = row[sort_column]
                if isinstance(col_value, (int, float)):
                    col_str = f"{col_value:.2f}"
                else:
                    col_str = str(col_value)
                
                info_parts = [f"{row['ts_code']}: {sort_column}={col_str}"]
                if "final_value" in display_cols:
                    info_parts.append(f"Final Value: {row['final_value']:.2f}")
                if "total_return" in display_cols and sort_column != "total_return":
                    info_parts.append(f"Return: {row['total_return']:.2f}%")
                
                logger.info(" | ".join(info_parts))

