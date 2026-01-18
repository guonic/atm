"""
Return Matrix Generator for correlation algorithm evaluation.

This module implements the first stage of the correlation test framework:
generating multi-period forward return matrices and tracking ranking evolution.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from qlib.data import D
except ImportError:
    D = None
    logger.warning("Qlib not available. Some features may not work.")

logger = logging.getLogger(__name__)


class ReturnMatrixGenerator:
    """
    Generate multi-period forward return matrix and ranking evolution data.
    
    This class processes daily StructureExpert rankings and calculates:
    1. Forward returns for multiple holding periods (T+3, T+5, T+10, etc.)
    2. Ranking evolution (how rankings change over time after entry)
    
    Purpose: Provide data foundation for evaluating correlation algorithm impact
    on StructureExpert model performance.
    """
    
    def __init__(
        self,
        holding_periods: List[int] = [3, 5, 8, 10, 15, 20, 30, 60],
        top_k: int = 20,
    ):
        """
        Initialize return matrix generator.
        
        Args:
            holding_periods: List of holding periods in trading days (default: [3, 5, 8, 10, 15, 20, 30, 60]).
            top_k: Number of top stocks to track from daily rankings (default: 20).
        """
        self.holding_periods = sorted(holding_periods)
        self.top_k = top_k
        logger.info(
            f"ReturnMatrixGenerator initialized: "
            f"holding_periods={self.holding_periods}, top_k={self.top_k}"
        )
    
    def generate(
        self,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
        price_data: Optional[pd.DataFrame] = None,
        instruments: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate return matrix for all dates and holding periods.
        
        Args:
            daily_ranks: Dictionary mapping dates to ranking DataFrames.
                        Each DataFrame should have columns: ['symbol', 'score', 'rank'].
            price_data: Optional pre-loaded price data DataFrame.
                       If None, will load from Qlib.
                       Expected format: MultiIndex (instrument, datetime) with '$close' column.
            instruments: Optional list of instruments to process.
                       If None, uses all symbols from daily_ranks.
        
        Returns:
            DataFrame with MultiIndex (date, symbol) and columns:
            - 'entry_rank': Rank at entry date
            - 'entry_score': Score at entry date
            - 'return_T+3': Forward return for T+3 days
            - 'return_T+5': Forward return for T+5 days
            - ... (one column per holding period)
            - 'rank_T+1': Rank after 1 day
            - 'rank_T+3': Rank after 3 days
            - ... (one column per holding period for ranking evolution)
        """
        if not daily_ranks:
            logger.warning("No daily ranks provided")
            return pd.DataFrame()
        
        # Get all unique symbols
        all_symbols = set()
        for ranks_df in daily_ranks.values():
            if 'symbol' in ranks_df.columns:
                all_symbols.update(ranks_df['symbol'].tolist())
        
        if instruments is None:
            instruments = sorted(list(all_symbols))
        
        logger.info(
            f"Generating return matrix for {len(daily_ranks)} dates, "
            f"{len(instruments)} instruments, {len(self.holding_periods)} holding periods"
        )
        
        # Load price data if not provided
        if price_data is None:
            price_data = self._load_price_data(instruments, daily_ranks)
        
        if price_data.empty:
            logger.warning("No price data available")
            return pd.DataFrame()
        
        # Generate return matrix
        return_matrix = []
        
        for date, ranks_df in sorted(daily_ranks.items()):
            # Get top K stocks for this date
            top_k_stocks = ranks_df.head(self.top_k)['symbol'].tolist()
            
            for symbol in top_k_stocks:
                if symbol not in instruments:
                    continue
                
                # Get entry rank and score
                entry_row = ranks_df[ranks_df['symbol'] == symbol]
                if entry_row.empty:
                    continue
                
                entry_rank = entry_row['rank'].iloc[0]
                entry_score = entry_row['score'].iloc[0]
                
                # Calculate forward returns for each holding period
                row_data = {
                    'date': date,
                    'symbol': symbol,
                    'entry_rank': entry_rank,
                    'entry_score': entry_score,
                }
                
                # Calculate forward returns
                for period in self.holding_periods:
                    forward_return = self._calculate_forward_return(
                        symbol, date, period, price_data
                    )
                    row_data[f'return_T+{period}'] = forward_return
                
                # Track ranking evolution
                for period in self.holding_periods:
                    future_rank = self._get_future_rank(
                        symbol, date, period, daily_ranks
                    )
                    row_data[f'rank_T+{period}'] = future_rank
                
                return_matrix.append(row_data)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(return_matrix)
        
        if result_df.empty:
            logger.warning("No return matrix data generated")
            return pd.DataFrame()
        
        # Set MultiIndex
        result_df = result_df.set_index(['date', 'symbol'])
        
        logger.info(
            f"Generated return matrix: {len(result_df)} rows, "
            f"{len(result_df.columns)} columns"
        )
        
        return result_df
    
    def _load_price_data(
        self,
        instruments: List[str],
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Load price data from Qlib for all instruments and dates.
        
        Args:
            instruments: List of instrument symbols.
            daily_ranks: Dictionary of daily rankings to determine date range.
        
        Returns:
            DataFrame with MultiIndex (instrument, datetime) and '$close' column.
        """
        if not daily_ranks:
            return pd.DataFrame()
        
        # Get date range
        dates = sorted(daily_ranks.keys())
        start_date = dates[0]
        end_date = dates[-1]
        
        # Add buffer for forward returns (need future prices)
        max_period = max(self.holding_periods)
        end_date_buffer = end_date + pd.Timedelta(days=max_period * 2)
        
        logger.info(
            f"Loading price data: {len(instruments)} instruments, "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date_buffer.strftime('%Y-%m-%d')} "
            f"(buffer for {max_period}-day forward returns)"
        )
        
        try:
            # Load close prices
            price_data = D.features(
                instruments=instruments,
                fields=["$close"],
                start_time=start_date.strftime("%Y-%m-%d"),
                end_time=end_date_buffer.strftime("%Y-%m-%d"),
            )
            
            if price_data.empty:
                logger.warning("No price data loaded from Qlib")
                return pd.DataFrame()
            
            logger.info(f"Loaded price data: {len(price_data)} rows")
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _calculate_forward_return(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        holding_period: int,
        price_data: pd.DataFrame,
    ) -> Optional[float]:
        """
        Calculate forward return for a specific holding period.
        
        Args:
            symbol: Stock symbol.
            entry_date: Entry date.
            holding_period: Number of trading days to hold.
            price_data: Price data DataFrame.
        
        Returns:
            Forward return (as decimal, e.g., 0.05 for 5%) or None if unavailable.
        """
        try:
            # Get entry price
            if isinstance(price_data.index, pd.MultiIndex):
                entry_prices = price_data.loc[(symbol, entry_date), :]
            else:
                entry_prices = price_data.loc[entry_date, symbol]
            
            if isinstance(entry_prices, pd.Series):
                entry_price = entry_prices.get('$close', None)
            else:
                entry_price = entry_prices
            
            if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
                return None
            
            # Get future date (need to account for trading days)
            # For simplicity, use calendar days and then find actual trading day
            future_date = entry_date + pd.Timedelta(days=holding_period * 2)  # Buffer
            
            # Find actual trading day at or after target
            if isinstance(price_data.index, pd.MultiIndex):
                symbol_data = price_data.loc[symbol, :]
                if symbol_data.empty:
                    return None
                
                # Get dates for this symbol
                dates = symbol_data.index.get_level_values(0) if isinstance(symbol_data.index, pd.MultiIndex) else symbol_data.index
                future_dates = dates[dates >= entry_date]
                
                if len(future_dates) < holding_period + 1:
                    return None
                
                # Get price at target period
                target_date = future_dates.iloc[holding_period] if len(future_dates) > holding_period else future_dates.iloc[-1]
                future_prices = price_data.loc[(symbol, target_date), :]
                
                if isinstance(future_prices, pd.Series):
                    future_price = future_prices.get('$close', None)
                else:
                    future_price = future_prices
            else:
                # Simple index case
                dates = price_data.index
                future_dates = dates[dates >= entry_date]
                
                if len(future_dates) < holding_period + 1:
                    return None
                
                target_date = future_dates.iloc[holding_period] if len(future_dates) > holding_period else future_dates.iloc[-1]
                future_price = price_data.loc[target_date, symbol]
            
            if future_price is None or pd.isna(future_price) or future_price <= 0:
                return None
            
            # Calculate return
            forward_return = (future_price - entry_price) / entry_price
            return float(forward_return)
            
        except Exception as e:
            logger.debug(f"Failed to calculate forward return for {symbol} on {entry_date}: {e}")
            return None
    
    def _get_future_rank(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        days_after: int,
        daily_ranks: Dict[pd.Timestamp, pd.DataFrame],
    ) -> Optional[int]:
        """
        Get rank of symbol after specified number of days.
        
        Args:
            symbol: Stock symbol.
            entry_date: Entry date.
            days_after: Number of days after entry.
            daily_ranks: Dictionary of daily rankings.
        
        Returns:
            Rank at future date, or None if unavailable.
        """
        # Find future date (approximate using calendar days)
        future_date = entry_date + pd.Timedelta(days=days_after * 2)  # Buffer
        
        # Find closest date in daily_ranks
        available_dates = sorted([d for d in daily_ranks.keys() if d > entry_date])
        
        if not available_dates:
            return None
        
        # Find date closest to target
        target_date = min(available_dates, key=lambda d: abs((d - future_date).days))
        
        # Check if we're close enough (within reasonable range)
        days_diff = abs((target_date - future_date).days)
        if days_diff > days_after * 2:  # Too far off
            return None
        
        # Get rank at future date
        future_ranks = daily_ranks[target_date]
        symbol_row = future_ranks[future_ranks['symbol'] == symbol]
        
        if symbol_row.empty:
            return None
        
        return int(symbol_row['rank'].iloc[0])
    
    def calculate_rank_halflife(
        self,
        return_matrix: pd.DataFrame,
        top_percentile: float = 0.2,
    ) -> pd.DataFrame:
        """
        Calculate ranking half-life (days until rank drops below top percentile).
        
        Args:
            return_matrix: Return matrix from generate().
            top_percentile: Top percentile threshold (default: 0.2 for top 20%).
        
        Returns:
            DataFrame with columns: ['symbol', 'entry_date', 'halflife_days'].
        """
        if return_matrix.empty:
            return pd.DataFrame()
        
        halflife_data = []
        
        for (date, symbol), row in return_matrix.iterrows():
            entry_rank = row.get('entry_rank', None)
            if entry_rank is None:
                continue
            
            # Calculate total stocks (approximate from entry_rank)
            # Assuming rank is within reasonable range
            total_stocks_estimate = int(entry_rank / (1 - top_percentile))
            top_rank_threshold = int(total_stocks_estimate * top_percentile)
            
            # Find when rank drops below threshold
            halflife = None
            for period in self.holding_periods:
                rank_col = f'rank_T+{period}'
                if rank_col not in row.index:
                    continue
                
                future_rank = row[rank_col]
                if pd.isna(future_rank):
                    continue
                
                if future_rank > top_rank_threshold:
                    halflife = period
                    break
            
            halflife_data.append({
                'symbol': symbol,
                'entry_date': date,
                'halflife_days': halflife,
            })
        
        return pd.DataFrame(halflife_data)
