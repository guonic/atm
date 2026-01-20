"""
Correlation Optimizer for correlation algorithm evaluation.

This module implements the second stage of the correlation test framework:
evaluating correlation algorithm impact on StructureExpert model performance
by applying correlation filters and comparing results.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from qlib.data import D
except ImportError:
    D = None

from .correlation import (
    ICorrelationCalculator,
    DynamicCrossSectionalCorrelation,
    CrossLaggedCorrelation,
    VolatilitySync,
    GrangerCausality,
    TransferEntropy,
)

logger = logging.getLogger(__name__)


class CorrelationOptimizer:
    """
    Evaluate correlation algorithm impact on StructureExpert model performance.
    
    This class implements the second stage of the correlation test framework:
    1. Traverse 5 correlation algorithms with different window parameters
    2. Apply correlation threshold filtering on Top K stocks
    3. Align filtered results with return matrix
    4. Calculate performance metrics (win rate, profit/loss ratio, max drawdown)
    """
    
    def __init__(
        self,
        window_params: List[int] = [3, 5, 8, 13],
        thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
        top_k: int = 20,
    ):
        """
        Initialize correlation optimizer.
        
        Args:
            window_params: List of window sizes for correlation calculation (default: [3, 5, 8, 13]).
            thresholds: List of correlation thresholds for filtering (default: [0.5, 0.6, 0.7, 0.8, 0.9]).
            top_k: Number of top stocks from StructureExpert rankings (default: 20).
        """
        self.window_params = window_params
        self.thresholds = sorted(thresholds)
        self.top_k = top_k
        
        # Initialize correlation calculators (will be configured with window params)
        self.calculators = {
            'DynamicCrossSectional': DynamicCrossSectionalCorrelation,
            'CrossLagged': CrossLaggedCorrelation,
            'VolatilitySync': VolatilitySync,
            'GrangerCausality': GrangerCausality,
            'TransferEntropy': TransferEntropy,
        }
        
        logger.info(
            f"CorrelationOptimizer initialized: "
            f"window_params={self.window_params}, "
            f"thresholds={self.thresholds}, top_k={self.top_k}"
        )
    
    def optimize(
        self,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
        return_matrix: pd.DataFrame,
        returns_data: pd.DataFrame,
        highs_data: Optional[pd.DataFrame] = None,
        lows_data: Optional[pd.DataFrame] = None,
        target_periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Run optimization across all algorithm and parameter combinations.
        
        Args:
            daily_ranks: Dictionary mapping dates to ranking DataFrames.
                        Each DataFrame should have columns: ['symbol', 'score', 'rank'].
            return_matrix: Return matrix from ReturnMatrixGenerator.generate().
            returns_data: DataFrame with returns data (columns: symbols, index: dates).
            highs_data: Optional DataFrame with high prices (for VolatilitySync).
            lows_data: Optional DataFrame with low prices (for VolatilitySync).
            target_periods: List of target holding periods to evaluate (default: all periods in return_matrix).
        
        Returns:
            DataFrame with optimization results:
            - algorithm: Algorithm name
            - window: Window parameter
            - threshold: Correlation threshold
            - target_period: Target holding period
            - baseline_win_rate: Baseline win rate (Top K without filtering)
            - filtered_win_rate: Win rate after correlation filtering
            - win_rate_lift: Win rate improvement (absolute)
            - baseline_avg_return: Baseline average return
            - filtered_avg_return: Filtered average return
            - profit_loss_ratio: Profit/Loss ratio
            - max_drawdown: Maximum drawdown
            - rank_halflife: Average ranking half-life (days)
        """
        if return_matrix.empty:
            logger.warning("Empty return matrix provided")
            return pd.DataFrame()
        
        # Determine target periods
        if target_periods is None:
            target_periods = self._extract_periods_from_matrix(return_matrix)
        
        logger.info(
            f"Starting optimization: {len(self.calculators)} algorithms, "
            f"{len(self.window_params)} window params, {len(self.thresholds)} thresholds, "
            f"{len(target_periods)} target periods"
        )
        
        results = []
        
        # Iterate over all combinations
        for algo_name, calculator_class in self.calculators.items():
            for window in self.window_params:
                # Create calculator instance with window parameter
                calculator = self._create_calculator(algo_name, calculator_class, window)
                
                if calculator is None:
                    continue
                
                # Calculate correlation matrix for all dates
                correlation_matrices = self._calculate_correlation_matrices(
                    calculator,
                    daily_ranks,
                    returns_data,
                    highs_data,
                    lows_data,
                )
                
                # Skip if no correlation matrices were calculated
                if not correlation_matrices:
                    logger.warning(
                        f"No correlation matrices calculated for {algo_name} "
                        f"with window={window}. Skipping this combination."
                    )
                    continue
                
                logger.debug(
                    f"Calculated {len(correlation_matrices)} correlation matrices "
                    f"for {algo_name} with window={window}"
                )
                
                # Evaluate each threshold
                for threshold in self.thresholds:
                    # Apply filtering and calculate metrics
                    # Now also includes accuracy metrics
                    evaluation_results = self._evaluate_filtering_and_accuracy(
                        daily_ranks=daily_ranks,
                        return_matrix=return_matrix,
                        returns_data=returns_data,
                        correlation_matrices=correlation_matrices,
                        threshold=threshold,
                        target_periods=target_periods,
                    )
                    
                    # Add metadata
                    for period, metrics in evaluation_results.items():
                        result_row = {
                            'algorithm': algo_name,
                            'window': window,
                            'threshold': threshold,
                            'target_period': period,
                            **metrics,
                        }
                        results.append(result_row)
        
        result_df = pd.DataFrame(results)
        
        if result_df.empty:
            logger.warning(
                "Optimization completed but produced no results. "
                "This may indicate that all correlation calculations failed or "
                "no valid combinations were found."
            )
        else:
            logger.info(
                f"Optimization completed: {len(result_df)} combinations evaluated. "
                f"Results contain {len(result_df.columns)} columns: {list(result_df.columns)[:5]}..."
            )
        
        return result_df
    
    def _create_calculator(
        self,
        algo_name: str,
        calculator_class: type,
        window: int,
    ) -> Optional[ICorrelationCalculator]:
        """
        Create calculator instance with appropriate parameters.
        
        Args:
            algo_name: Algorithm name.
            calculator_class: Calculator class.
            window: Window parameter.
        
        Returns:
            Calculator instance or None if creation fails.
        """
        try:
            if algo_name == 'DynamicCrossSectional':
                return calculator_class(window=window)
            elif algo_name == 'CrossLagged':
                return calculator_class(lag=window)
            elif algo_name == 'VolatilitySync':
                return calculator_class(bandwidth=window / 100.0)  # Convert to bandwidth
            elif algo_name == 'GrangerCausality':
                return calculator_class(maxlag=window)
            elif algo_name == 'TransferEntropy':
                return calculator_class(n_bins=window)  # Use window as n_bins
            else:
                logger.warning(f"Unknown algorithm: {algo_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to create calculator {algo_name} with window {window}: {e}")
            return None
    
    def _calculate_correlation_matrices(
        self,
        calculator: ICorrelationCalculator,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
        returns_data: pd.DataFrame,
        highs_data: Optional[pd.DataFrame] = None,
        lows_data: Optional[pd.DataFrame] = None,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Calculate correlation matrices for all dates.
        
        Args:
            calculator: Correlation calculator instance.
            daily_ranks: Dictionary of daily rankings.
            returns_data: Returns DataFrame.
            highs_data: Optional highs DataFrame.
            lows_data: Optional lows DataFrame.
        
        Returns:
            Dictionary mapping dates to correlation matrices.
        """
        correlation_matrices = {}
        success_count = 0
        fail_count = 0
        
        for date in sorted(daily_ranks.keys()):
            # Get top K symbols for this date
            ranks_df = daily_ranks[date]
            top_k_symbols = ranks_df.head(self.top_k)['symbol'].tolist()
            
            if len(top_k_symbols) < 2:
                fail_count += 1
                continue
            
            # Get historical returns for these symbols
            # Use lookback window before this date
            date_str = date.strftime("%Y-%m-%d")
            
            try:
                # Calculate correlation matrix
                if isinstance(calculator, VolatilitySync):
                    if highs_data is None or lows_data is None:
                        fail_count += 1
                        continue
                    
                    # Filter data to only include dates up to current date
                    date_mask = highs_data.index <= date
                    highs_for_date = highs_data[date_mask]
                    lows_for_date = lows_data[date_mask]
                    
                    if highs_for_date.empty or lows_for_date.empty or len(highs_for_date) < 2:
                        fail_count += 1
                        logger.debug(
                            f"Insufficient highs/lows data for {date_str}: "
                            f"{len(highs_for_date)} days available"
                        )
                        continue
                    
                    # For VolatilitySync, need highs and lows
                    corr_matrix = calculator.calculate_matrix(
                        returns_data,  # Required by interface but not used
                        symbols=top_k_symbols,
                        highs=highs_for_date,
                        lows=lows_for_date,
                    )
                else:
                    # Filter returns_data to only include dates up to current date
                    # This ensures we don't use future data for correlation calculation
                    date_mask = returns_data.index <= date
                    returns_for_date = returns_data[date_mask]
                    
                    if returns_for_date.empty or len(returns_for_date) < 2:
                        fail_count += 1
                        logger.debug(
                            f"Insufficient returns data for {date_str}: "
                            f"{len(returns_for_date)} days available"
                        )
                        continue
                    
                    corr_matrix = calculator.calculate_matrix(
                        returns_for_date,
                        symbols=top_k_symbols,
                    )
                
                # Handle tuple return (for directed correlations)
                if isinstance(corr_matrix, tuple):
                    corr_matrix = corr_matrix[0]  # Use correlation matrix, ignore direction
                
                # Validate correlation matrix
                if corr_matrix.empty:
                    fail_count += 1
                    logger.debug(f"Empty correlation matrix for {date_str}")
                    continue
                
                correlation_matrices[date] = corr_matrix
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                logger.debug(f"Failed to calculate correlation for {date_str}: {e}")
                continue
        
        logger.info(
            f"Correlation matrix calculation: {success_count} succeeded, "
            f"{fail_count} failed out of {len(daily_ranks)} dates"
        )
        
        return correlation_matrices
    
    def _evaluate_filtering_and_accuracy(
        self,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
        return_matrix: pd.DataFrame,
        returns_data: pd.DataFrame,
        correlation_matrices: Dict[pd.Timestamp, pd.DataFrame],
        threshold: float,
        target_periods: List[int],
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate filtering performance and correlation accuracy.
        
        Args:
            daily_ranks: Dictionary of daily rankings.
            return_matrix: Return matrix DataFrame.
            returns_data: Returns DataFrame (must include future data).
            correlation_matrices: Dictionary of correlation matrices by date.
            threshold: Correlation threshold.
            target_periods: List of target holding periods.
        
        Returns:
            Dictionary mapping period to metrics (performance + accuracy).
        """
        metrics = {}
        
        if not correlation_matrices or return_matrix.empty:
            return metrics
        
        # Get all relevant dates and symbols
        daily_ranks_dates = set(daily_ranks.keys())
        correlation_dates = set(correlation_matrices.keys())
        return_matrix_dates = set(return_matrix.index.get_level_values(0))
        return_matrix_symbols = set(return_matrix.index.get_level_values(1))
        
        overlap_dates = sorted(list(daily_ranks_dates & correlation_dates & return_matrix_dates))
        
        if not overlap_dates:
            return metrics

        for period in target_periods:
            return_col = f'return_T+{period}'
            if return_col not in return_matrix.columns:
                continue
            
            baseline_returns = []
            filtered_returns = []
            accuracy_scores = []
            
            filtered_count = 0
            total_count = 0
            
            for date in overlap_dates:
                # 1. Prediction/Filtering Logic
                ranks_df = daily_ranks[date]
                top_k_stocks = [s for s in ranks_df.head(self.top_k)['symbol'].tolist() if s in return_matrix_symbols]
                
                if not top_k_stocks:
                    continue
                
                corr_matrix = correlation_matrices[date]
                filtered_stocks = []
                
                # Check accuracy for this date/period combination
                # Accuracy is the correlation between Predicted Matrix and Realized Matrix
                if len(top_k_stocks) >= 2:
                    realized_accuracy = self._calculate_accuracy_for_date(
                        date, period, top_k_stocks, corr_matrix, returns_data
                    )
                    if realized_accuracy is not None:
                        accuracy_scores.append(realized_accuracy)

                for symbol in top_k_stocks:
                    if symbol not in corr_matrix.index:
                        continue
                    
                    other_stocks = [s for s in top_k_stocks if s != symbol and s in corr_matrix.columns]
                    if not other_stocks:
                        continue
                    
                    avg_corr = corr_matrix.loc[symbol, other_stocks].mean()
                    
                    if not pd.isna(avg_corr) and avg_corr >= threshold:
                        filtered_stocks.append(symbol)
                
                # 2. Performance Tracking
                for symbol in top_k_stocks:
                    try:
                        if (date, symbol) not in return_matrix.index:
                            continue
                            
                        return_value = return_matrix.loc[(date, symbol), return_col]
                        if pd.isna(return_value):
                            continue
                        
                        total_count += 1
                        baseline_returns.append(return_value)
                        
                        if symbol in filtered_stocks:
                            filtered_count += 1
                            filtered_returns.append(return_value)
                    except:
                        continue
            
            # 3. Calculate Final Metrics for Period
            if not baseline_returns:
                continue
            
            # Baseline Stats
            baseline_win_rate = sum(1 for r in baseline_returns if r > 0) / len(baseline_returns)
            baseline_avg_return = np.mean(baseline_returns)
            
            # Filtered Stats
            if filtered_returns:
                filtered_win_rate = sum(1 for r in filtered_returns if r > 0) / len(filtered_returns)
                filtered_avg_return = np.mean(filtered_returns)
                win_rate_lift = filtered_win_rate - baseline_win_rate
                
                profits = [r for r in filtered_returns if r > 0]
                losses = [r for r in filtered_returns if r < 0]
                profit_loss_ratio = np.mean(profits) / abs(np.mean(losses)) if losses else float('inf')
                
                cumulative = np.cumprod(1 + np.array(filtered_returns))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            else:
                filtered_win_rate = 0.0
                filtered_avg_return = 0.0
                win_rate_lift = -baseline_win_rate
                profit_loss_ratio = 0.0
                max_drawdown = 0.0
            
            # Accuracy Metric (Mean of Meta-Correlations)
            mean_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            metrics[period] = {
                'baseline_win_rate': baseline_win_rate,
                'filtered_win_rate': filtered_win_rate,
                'win_rate_lift': win_rate_lift,
                'baseline_avg_return': baseline_avg_return,
                'filtered_avg_return': filtered_avg_return,
                'profit_loss_ratio': profit_loss_ratio,
                'max_drawdown': max_drawdown,
                'accuracy': mean_accuracy,
                'filtered_count': filtered_count,
                'total_count': total_count,
            }
        
        return metrics

    def _calculate_accuracy_for_date(
        self,
        date: pd.Timestamp,
        period: int,
        symbols: List[str],
        predicted_matrix: pd.DataFrame,
        returns_data: pd.DataFrame,
    ) -> Optional[float]:
        """
        Calculate Correlation Accuracy (Meta-Correlation) for a specific date and period.
        
        Accuracy = Correlation(vectorized(Predicted Matrix), vectorized(Realized Matrix))
        """
        try:
            # 1. Get realized daily returns for the future window [T+1, T+period]
            future_dates = returns_data.index[returns_data.index > date]
            if len(future_dates) < period:
                return None
            
            realized_window_dates = future_dates[:period]
            realized_returns = returns_data.loc[realized_window_dates, symbols]
            
            if realized_returns.empty or realized_returns.isna().all().all():
                return None
            
            # 2. Calculate Realized Correlation Matrix
            # Use Pearson for realized ground truth
            realized_matrix = realized_returns.corr(method='pearson')
            
            # 3. Compare Predicted vs Realized
            # Ensure both have the same symbols in the same order
            common_symbols = [s for s in symbols if s in realized_matrix.index and s in predicted_matrix.index]
            if len(common_symbols) < 2:
                return None
            
            pred = predicted_matrix.loc[common_symbols, common_symbols]
            real = realized_matrix.loc[common_symbols, common_symbols]
            
            # Extract upper triangle (excluding diagonal)
            pred_values = pred.values[np.triu_indices(len(common_symbols), k=1)]
            real_values = real.values[np.triu_indices(len(common_symbols), k=1)]
            
            if len(pred_values) < 1:
                return None
                
            # Filter NaNs
            mask = ~np.isnan(pred_values) & ~np.isnan(real_values)
            if mask.sum() < 2:
                return None
                
            # Correlation of Correlations (Meta-Correlation)
            accuracy = np.corrcoef(pred_values[mask], real_values[mask])[0, 1]
            return float(accuracy) if not np.isnan(accuracy) else None
            
        except Exception as e:
            logger.debug(f"Failed to calculate accuracy for {date} period {period}: {e}")
            return None
    
    def _extract_periods_from_matrix(self, return_matrix: pd.DataFrame) -> List[int]:
        """
        Extract holding periods from return matrix columns.
        
        Args:
            return_matrix: Return matrix DataFrame.
        
        Returns:
            List of holding periods.
        """
        periods = []
        for col in return_matrix.columns:
            if col.startswith('return_T+'):
                period_str = col.replace('return_T+', '')
                try:
                    periods.append(int(period_str))
                except ValueError:
                    continue
        return sorted(periods)
