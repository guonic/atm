"""
Correlation Filter for stock selection.

This module implements a correlation-based filter that can be integrated
into trading strategies to filter stocks based on correlation metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .correlation import (
    ICorrelationCalculator,
    DynamicCrossSectionalCorrelation,
    CrossLaggedCorrelation,
    VolatilitySync,
    GrangerCausality,
    TransferEntropy,
)

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """
    Correlation-based filter for stock selection.
    
    This filter applies correlation algorithms to filter stocks from a ranking list.
    It can be used as a secondary filter after StructureExpert model ranking.
    
    Usage:
        filter = CorrelationFilter(
            algorithm='DynamicCrossSectional',
            window=5,
            threshold=0.7
        )
        filtered_stocks = filter.filter_stocks(
            ranks_df=ranks_df,
            returns_data=returns_data,
            date=date
        )
    """
    
    def __init__(
        self,
        algorithm: str = 'DynamicCrossSectional',
        window: int = 5,
        threshold: float = 0.7,
        use_average_correlation: bool = True,
        min_stocks: int = 2,
    ):
        """
        Initialize correlation filter.
        
        Args:
            algorithm: Correlation algorithm name. Options:
                - 'DynamicCrossSectional': Dynamic cross-sectional correlation
                - 'CrossLagged': Cross-lagged correlation
                - 'VolatilitySync': Volatility synchronization
                - 'GrangerCausality': Granger causality
                - 'TransferEntropy': Transfer entropy
            window: Window parameter for correlation calculation.
            threshold: Correlation threshold for filtering (stocks with correlation >= threshold are kept).
            use_average_correlation: If True, use average correlation with other stocks.
                                     If False, use minimum correlation (stricter).
            min_stocks: Minimum number of stocks required for correlation calculation.
        """
        self.algorithm = algorithm
        self.window = window
        self.threshold = threshold
        self.use_average_correlation = use_average_correlation
        self.min_stocks = min_stocks
        
        # Create calculator instance
        self.calculator = self._create_calculator()
        
        logger.info(
            f"CorrelationFilter initialized: algorithm={algorithm}, "
            f"window={window}, threshold={threshold}, "
            f"use_average={use_average_correlation}"
        )
    
    def _create_calculator(self) -> Optional[ICorrelationCalculator]:
        """
        Create correlation calculator instance.
        
        Returns:
            Calculator instance or None if creation fails.
        """
        try:
            if self.algorithm == 'DynamicCrossSectional':
                return DynamicCrossSectionalCorrelation(window=self.window)
            elif self.algorithm == 'CrossLagged':
                return CrossLaggedCorrelation(lag=self.window)
            elif self.algorithm == 'VolatilitySync':
                return VolatilitySync(bandwidth=self.window / 100.0)
            elif self.algorithm == 'GrangerCausality':
                return GrangerCausality(maxlag=self.window)
            elif self.algorithm == 'TransferEntropy':
                return TransferEntropy(n_bins=self.window)
            else:
                logger.error(f"Unknown algorithm: {self.algorithm}")
                return None
        except Exception as e:
            logger.error(f"Failed to create calculator: {e}", exc_info=True)
            return None
    
    def filter_stocks(
        self,
        ranks_df: pd.DataFrame,
        returns_data: pd.DataFrame,
        date: pd.Timestamp,
        highs_data: Optional[pd.DataFrame] = None,
        lows_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter stocks based on correlation threshold.
        
        Args:
            ranks_df: DataFrame with columns ['symbol', 'score', 'rank'].
                    Should be sorted by score (descending).
            returns_data: DataFrame with returns data (columns: symbols, index: dates).
            date: Current date for filtering.
            highs_data: Optional DataFrame with high prices (for VolatilitySync).
            lows_data: Optional DataFrame with low prices (for VolatilitySync).
        
        Returns:
            Filtered DataFrame with same columns as ranks_df, containing only
            stocks that pass the correlation filter.
        """
        if self.calculator is None:
            logger.warning("Calculator not available, returning original ranks")
            return ranks_df
        
        if ranks_df.empty:
            return ranks_df
        
        if len(ranks_df) < self.min_stocks:
            logger.debug(
                f"Insufficient stocks for correlation filter: {len(ranks_df)} < {self.min_stocks}"
            )
            return ranks_df
        
        try:
            # Get symbols from ranks
            symbols = ranks_df['symbol'].tolist()
            
            # Calculate correlation matrix
            if isinstance(self.calculator, VolatilitySync):
                if highs_data is None or lows_data is None:
                    logger.warning(
                        "VolatilitySync requires highs_data and lows_data, "
                        "returning original ranks"
                    )
                    return ranks_df
                corr_matrix = self.calculator.calculate_matrix(
                    returns_data,  # Required by interface but not used
                    symbols=symbols,
                    highs=highs_data,
                    lows=lows_data,
                )
            else:
                corr_matrix = self.calculator.calculate_matrix(
                    returns_data,
                    symbols=symbols,
                )
            
            # Handle tuple return (for directed correlations)
            if isinstance(corr_matrix, tuple):
                corr_matrix = corr_matrix[0]  # Use correlation matrix, ignore direction
            
            if corr_matrix.empty:
                logger.warning("Empty correlation matrix, returning original ranks")
                return ranks_df
            
            # Filter stocks based on correlation threshold
            filtered_symbols = self._apply_correlation_filter(
                symbols=symbols,
                corr_matrix=corr_matrix,
            )
            
            # Return filtered ranks
            filtered_ranks = ranks_df[ranks_df['symbol'].isin(filtered_symbols)].copy()
            
            logger.debug(
                f"Correlation filter: {len(ranks_df)} -> {len(filtered_ranks)} stocks "
                f"(threshold={self.threshold}, algorithm={self.algorithm})"
            )
            
            return filtered_ranks
        
        except Exception as e:
            logger.warning(f"Failed to apply correlation filter: {e}", exc_info=True)
            return ranks_df
    
    def _apply_correlation_filter(
        self,
        symbols: List[str],
        corr_matrix: pd.DataFrame,
    ) -> List[str]:
        """
        Apply correlation threshold filter to symbols.
        
        Args:
            symbols: List of stock symbols.
            corr_matrix: Correlation matrix DataFrame.
        
        Returns:
            List of symbols that pass the filter.
        """
        filtered_symbols = []
        
        for symbol in symbols:
            if symbol not in corr_matrix.index:
                continue
            
            # Get correlations with other symbols
            other_symbols = [s for s in symbols if s != symbol and s in corr_matrix.columns]
            
            if not other_symbols:
                # If no other symbols, keep this one
                filtered_symbols.append(symbol)
                continue
            
            # Calculate correlation metric
            if self.use_average_correlation:
                # Use average correlation with other stocks
                correlations = corr_matrix.loc[symbol, other_symbols]
                metric = correlations.mean()
            else:
                # Use minimum correlation (stricter)
                correlations = corr_matrix.loc[symbol, other_symbols]
                metric = correlations.min()
            
            # Apply threshold
            if not pd.isna(metric) and metric >= self.threshold:
                filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def get_filter_stats(
        self,
        ranks_df: pd.DataFrame,
        returns_data: pd.DataFrame,
        date: pd.Timestamp,
        highs_data: Optional[pd.DataFrame] = None,
        lows_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Get filter statistics without actually filtering.
        
        Args:
            ranks_df: DataFrame with columns ['symbol', 'score', 'rank'].
            returns_data: DataFrame with returns data.
            date: Current date.
            highs_data: Optional DataFrame with high prices.
            lows_data: Optional DataFrame with low prices.
        
        Returns:
            Dictionary with filter statistics:
            - 'total_stocks': Total number of stocks
            - 'filtered_stocks': Number of stocks after filtering
            - 'filter_ratio': Ratio of filtered stocks
            - 'avg_correlation': Average correlation among filtered stocks
        """
        if self.calculator is None or ranks_df.empty:
            return {
                'total_stocks': len(ranks_df),
                'filtered_stocks': len(ranks_df),
                'filter_ratio': 1.0,
                'avg_correlation': 0.0,
            }
        
        try:
            # Apply filter
            filtered_ranks = self.filter_stocks(
                ranks_df=ranks_df,
                returns_data=returns_data,
                date=date,
                highs_data=highs_data,
                lows_data=lows_data,
            )
            
            # Calculate average correlation for filtered stocks
            symbols = filtered_ranks['symbol'].tolist()
            avg_correlation = 0.0
            
            if len(symbols) >= 2:
                # Calculate correlation matrix for filtered stocks
                if isinstance(self.calculator, VolatilitySync):
                    if highs_data is None or lows_data is None:
                        avg_correlation = 0.0
                    else:
                        corr_matrix = self.calculator.calculate_matrix(
                            returns_data,
                            symbols=symbols,
                            highs=highs_data,
                            lows=lows_data,
                        )
                        if isinstance(corr_matrix, tuple):
                            corr_matrix = corr_matrix[0]
                        if not corr_matrix.empty:
                            # Calculate average correlation (excluding diagonal)
                            mask = ~pd.DataFrame(
                                index=corr_matrix.index,
                                columns=corr_matrix.columns
                            ).apply(lambda x: x.index == x.name, axis=1)
                            avg_correlation = corr_matrix.where(mask).mean().mean()
                else:
                    corr_matrix = self.calculator.calculate_matrix(
                        returns_data,
                        symbols=symbols,
                    )
                    if isinstance(corr_matrix, tuple):
                        corr_matrix = corr_matrix[0]
                    if not corr_matrix.empty:
                        # Calculate average correlation (excluding diagonal)
                        mask = ~pd.DataFrame(
                            index=corr_matrix.index,
                            columns=corr_matrix.columns
                        ).apply(lambda x: x.index == x.name, axis=1)
                        avg_correlation = corr_matrix.where(mask).mean().mean()
            
            return {
                'total_stocks': len(ranks_df),
                'filtered_stocks': len(filtered_ranks),
                'filter_ratio': len(filtered_ranks) / len(ranks_df) if len(ranks_df) > 0 else 0.0,
                'avg_correlation': avg_correlation if not pd.isna(avg_correlation) else 0.0,
            }
        
        except Exception as e:
            logger.warning(f"Failed to calculate filter stats: {e}", exc_info=True)
            return {
                'total_stocks': len(ranks_df),
                'filtered_stocks': len(ranks_df),
                'filter_ratio': 1.0,
                'avg_correlation': 0.0,
            }
