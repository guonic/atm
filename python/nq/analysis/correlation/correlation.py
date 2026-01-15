"""
Correlation analysis methods for stock structure features.

Implements five core correlation calculation methods:
1. Dynamic Cross-Sectional Correlation
2. Cross-Lagged Correlation
3. Volatility Sync
4. Granger Causality
5. Transfer Entropy

All correlation calculators implement the ICorrelationCalculator interface
for unified usage patterns.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


class ICorrelationCalculator(ABC):
    """
    Interface for correlation calculation methods.
    
    All correlation calculators should implement this interface to provide
    a unified API for calculating stock correlations and building graph structures.
    
    The interface supports:
    - Matrix calculation: Calculate correlation matrix for all stock pairs
    - Pairwise calculation: Calculate correlation for a single stock pair
    - Flexible return types: Single matrix or tuple of matrices (with direction/auxiliary data)
    - Optional additional inputs: Some calculators may require extra data (e.g., highs, lows)
    """
    
    @abstractmethod
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Calculate correlation matrix for all stock pairs.
        
        Args:
            returns: DataFrame with returns data.
                    Index: datetime
                    Columns: stock symbols
                    Values: returns
            symbols: Optional list of symbols to calculate (if None, uses all columns).
            **kwargs: Additional parameters (e.g., highs, lows for VolatilitySync).
        
        Returns:
            Single DataFrame (symmetric matrix) or Tuple of DataFrames:
            - For symmetric correlations: pd.DataFrame (correlation matrix)
            - For directed correlations: Tuple[pd.DataFrame, pd.DataFrame]
              (correlation matrix, direction/auxiliary matrix)
        """
        pass
    
    @abstractmethod
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
        **kwargs
    ) -> Union[float, Tuple[float, float], Tuple[float, float, Optional[str]], Tuple[bool, float, Optional[str]]]:
        """
        Calculate correlation between two stock series.
        
        Args:
            returns_i: Return series for stock i.
            returns_j: Return series for stock j.
            **kwargs: Additional parameters.
        
        Returns:
            Single value or tuple:
            - For symmetric correlations: float (correlation coefficient)
            - For directed correlations: Tuple[float, float, Optional[str]]
              (corr_i_to_j, corr_j_to_i, direction)
            - For causality tests: Tuple[bool, float, Optional[str]]
              (is_causal, p_value, direction)
        """
        pass


class DynamicCrossSectionalCorrelation(ICorrelationCalculator):
    """
    Dynamic Cross-Sectional Correlation calculator.
    
    Purpose: Basic topology construction. Identifies which stocks are currently
    in the same "sentiment cluster".
    
    Formula:
        R_{i,j} = Σ(r_{i,t} - r̄_i)(r_{j,t} - r̄_j) / √(Σ(r_{i,t} - r̄_i)² Σ(r_{j,t} - r̄_j)²)
    
    Engineering suggestion: Use KNN algorithm or threshold θ (e.g., R > 0.5)
    to sparsify the graph.
    """
    
    def __init__(self, window: int = 60, threshold: Optional[float] = None):
        """
        Initialize dynamic cross-sectional correlation calculator.
        
        Args:
            window: Lookback window size (T-n) for correlation calculation.
            threshold: Optional threshold for sparsification (e.g., 0.5).
                      If None, returns all correlations.
        """
        self.window = window
        self.threshold = 0.0 if threshold is None else threshold
    
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate dynamic cross-sectional correlation matrix.
        
        Args:
            returns: DataFrame with returns data.
                    Index: datetime
                    Columns: stock symbols
                    Values: returns
            symbols: Optional list of symbols to calculate (if None, uses all columns).
        
        Returns:
            Correlation matrix DataFrame (symmetric).
        """
        if symbols is None:
            symbols = returns.columns.tolist()
        
        # Select symbols
        returns_subset = returns[symbols]
        
        # Use rolling window if data is longer than window
        if len(returns_subset) > self.window:
            # Use last window days
            returns_subset = returns_subset.iloc[-self.window:]
        
        # Calculate spearman correlation, Notes: The Pearson algorithm is overly sensitive to extreme values.
        correlation_matrix = returns_subset.corr(method="spearman")
        
        # Apply threshold if specified
        correlation_matrix = correlation_matrix.where(
            abs(correlation_matrix) >= self.threshold, 0.0
        )
        
        return correlation_matrix
    
    def calculate(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Alias for calculate_matrix (backward compatibility).
        """
        return self.calculate_matrix(returns, symbols)
    
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
    ) -> float:
        """
        Calculate correlation between two return series.
        
        Args:
            returns_i: Return series for stock i.
            returns_j: Return series for stock j.
        
        Returns:
            Pearson correlation coefficient.
        """
        # Align indices
        aligned = pd.DataFrame({"i": returns_i, "j": returns_j}).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        # Use rolling window if needed
        if len(aligned) > self.window:
            aligned = aligned.iloc[-self.window:]
        
        return aligned["i"].corr(aligned["j"], method="spearman")


class CrossLaggedCorrelation(ICorrelationCalculator):
    """
    Cross-Lagged Correlation calculator.
    
    Purpose: Break symmetry, identify "leaders" and "followers", build directed graph.
    
    Formula:
        W_{A→B} = Corr(A_{t-k}, B_t)
    
    Logic: If Corr(A_{t-1}, B_t) > Corr(B_{t-1}, A_t), then establish A→B edge,
           indicating A is the current market driver.
    """
    
    def __init__(self, lag: int = 1):
        """
        Initialize cross-lagged correlation calculator.
        
        Args:
            lag: Lag period k (default: 1).
        """
        self.lag = lag
    
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
        **kwargs
    ) -> Tuple[float, float, Optional[str]]:
        """
        Calculate directed correlation between two stocks.
        
        Args:
            returns_i: Return series for stock i.
            returns_j: Return series for stock j.
        
        Returns:
            Tuple of (corr_i_to_j, corr_j_to_i, direction).
            direction: 'i->j', 'j->i', or None if no clear direction.
        """
        # Align indices
        aligned = pd.DataFrame({"i": returns_i, "j": returns_j}).dropna()
        
        if len(aligned) < self.lag + 1:
            return 0.0, 0.0, None
        
        # Calculate i -> j: Corr(i_{t-k}, j_t)
        i_lagged = aligned["i"].shift(self.lag)
        j_current = aligned["j"]
        pair_ij = pd.DataFrame({"i_lag": i_lagged, "j": j_current}).dropna()
        
        if len(pair_ij) < 2:
            corr_i_to_j = 0.0
        else:
            corr_i_to_j = pair_ij["i_lag"].corr(pair_ij["j"], method="spearman")
            if pd.isna(corr_i_to_j):
                corr_i_to_j = 0.0
        
        # Calculate j -> i: Corr(j_{t-k}, i_t)
        j_lagged = aligned["j"].shift(self.lag)
        i_current = aligned["i"]
        pair_ji = pd.DataFrame({"j_lag": j_lagged, "i": i_current}).dropna()
        
        if len(pair_ji) < 2:
            corr_j_to_i = 0.0
        else:
            corr_j_to_i = pair_ji["j_lag"].corr(pair_ji["i"], method="spearman")
            if pd.isna(corr_j_to_i):
                corr_j_to_i = 0.0
        
        # Determine direction
        if corr_i_to_j > corr_j_to_i:
            direction = "i->j"
        elif corr_j_to_i > corr_i_to_j:
            direction = "j->i"
        else:
            direction = None
        
        return corr_i_to_j, corr_j_to_i, direction
    
    def calculate_directed(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
    ) -> Tuple[float, float, Optional[str]]:
        """
        Alias for calculate_pairwise (backward compatibility).
        """
        return self.calculate_pairwise(returns_i, returns_j)
    
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate directed correlation matrix for all pairs.
        
        Args:
            returns: DataFrame with returns data.
            symbols: Optional list of symbols.
        
        Returns:
            Tuple of (correlation_matrix, direction_matrix).
            correlation_matrix: Maximum correlation value for each pair.
            direction_matrix: Direction string ('i->j', 'j->i', or None).
        """
        if symbols is None:
            symbols = returns.columns.tolist()
        
        n = len(symbols)
        corr_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)
        direction_matrix = pd.DataFrame(None, index=symbols, columns=symbols, dtype=object)
        
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    corr_matrix.loc[sym_i, sym_j] = 1.0
                    direction_matrix.loc[sym_i, sym_j] = None
                    continue
                
                corr_i_to_j, corr_j_to_i, direction = self.calculate_pairwise(
                    returns[sym_i], returns[sym_j]
                )
                
                # Store maximum correlation
                max_corr = max(corr_i_to_j, corr_j_to_i)
                corr_matrix.loc[sym_i, sym_j] = max_corr
                direction_matrix.loc[sym_i, sym_j] = direction
        
        return corr_matrix, direction_matrix


class VolatilitySync(ICorrelationCalculator):
    """
    Volatility Sync calculator.
    
    Purpose: Measure risk state consistency. Identify stocks controlled by the same
    type of capital (e.g., quantitative high-frequency arrays).
    
    Calculation logic (based on Range):
    1. Calculate intraday amplitude: Range_{i,t} = ln(High_{i,t} / Low_{i,t})
    2. Calculate sync rate: Sync_{i,j} = 1 - |Range_i - Range_j| / (Range_i + Range_j)
    
    Edge feature transformation: Use Gaussian kernel: e_{ij} = exp(-(σ_i - σ_j)² / (2h²))
    """
    
    def __init__(self, bandwidth: float = 0.1):
        """
        Initialize volatility sync calculator.
        
        Args:
            bandwidth: Bandwidth parameter h for Gaussian kernel (default: 0.1).
        """
        self.bandwidth = bandwidth
    
    def calculate_range(
        self,
        highs: pd.Series,
        lows: pd.Series,
    ) -> pd.Series:
        """
        Calculate intraday amplitude (Range) for each day.
        
        Args:
            highs: High prices series.
            lows: Low prices series.
        
        Returns:
            Range series: ln(High / Low)
        """
        aligned = pd.DataFrame({"high": highs, "low": lows}).dropna()
        aligned = aligned[aligned["high"] > 0]
        aligned = aligned[aligned["low"] > 0]
        
        range_values = np.log(aligned["high"] / aligned["low"])
        return range_values
    
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
        highs_i: Optional[pd.Series] = None,
        highs_j: Optional[pd.Series] = None,
        lows_i: Optional[pd.Series] = None,
        lows_j: Optional[pd.Series] = None,
        **kwargs
    ) -> float:
        """
        Calculate sync rate between two stocks.
        
        Args:
            returns_i: Return series for stock i (required by interface, but not used).
            returns_j: Return series for stock j (required by interface, but not used).
            highs_i: High prices series for stock i.
            highs_j: High prices series for stock j.
            lows_i: Low prices series for stock i.
            lows_j: Low prices series for stock j.
            **kwargs: Additional parameters.
        
        Returns:
            Sync rate (0-1, closer to 1 means more synchronized).
        """
        if highs_i is None or highs_j is None or lows_i is None or lows_j is None:
            raise ValueError("VolatilitySync.calculate_pairwise requires 'highs_i', 'highs_j', 'lows_i', 'lows_j'")
        
        range_i = self.calculate_range(highs_i, lows_i)
        range_j = self.calculate_range(highs_j, lows_j)
        
        # Align indices
        aligned = pd.DataFrame({"i": range_i, "j": range_j}).dropna()
        
        if len(aligned) == 0:
            return 0.0
        
        # Calculate average sync rate
        sync_rates = []
        for _, row in aligned.iterrows():
            range_i_val = row["i"]
            range_j_val = row["j"]
            
            if range_i_val + range_j_val == 0:
                sync_rate = 0.0
            else:
                sync_rate = 1 - abs(range_i_val - range_j_val) / (range_i_val + range_j_val)
            
            sync_rates.append(sync_rate)
        
        return np.mean(sync_rates)
    
    def calculate_gaussian_kernel(
        self,
        volatility_i: float,
        volatility_j: float,
    ) -> float:
        """
        Calculate Gaussian kernel edge feature.
        
        Args:
            volatility_i: Volatility (standard deviation) of stock i.
            volatility_j: Volatility (standard deviation) of stock j.
        
        Returns:
            Edge weight: exp(-(σ_i - σ_j)² / (2h²))
        """
        diff_squared = (volatility_i - volatility_j) ** 2
        exponent = -diff_squared / (2 * self.bandwidth ** 2)
        return np.exp(exponent)
    
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        highs: Optional[pd.DataFrame] = None,
        lows: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate volatility sync matrix for all pairs.
        
        Args:
            returns: DataFrame with returns data (required by interface, but not used).
            symbols: Optional list of symbols.
            highs: DataFrame with high prices (columns: symbols, index: dates).
            lows: DataFrame with low prices (columns: symbols, index: dates).
            **kwargs: Additional parameters.
        
        Returns:
            Sync matrix DataFrame.
        """
        if highs is None or lows is None:
            raise ValueError("VolatilitySync requires 'highs' and 'lows' DataFrames")
        
        if symbols is None:
            symbols = highs.columns.tolist()
        
        n = len(symbols)
        sync_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)
        
        # Calculate ranges for all stocks
        ranges = {}
        for sym in symbols:
            ranges[sym] = self.calculate_range(highs[sym], lows[sym])
        
        # Calculate sync rates for all pairs
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    sync_matrix.loc[sym_i, sym_j] = 1.0
                    continue
                
                sync_rate = self.calculate_sync_rate(ranges[sym_i], ranges[sym_j])
                sync_matrix.loc[sym_i, sym_j] = sync_rate
        
        return sync_matrix


class GrangerCausality(ICorrelationCalculator):
    """
    Granger Causality test calculator.
    
    Purpose: Statistically determine causal direction, filter out pure random "pseudo-correlations".
    
    Mathematical model:
        Compare two regression models:
        1. B_t = Σ α_i B_{t-i} + ε₁
        2. B_t = Σ α_i B_{t-i} + Σ β_j A_{t-j} + ε₂
    
    Calculation: F-test P-value.
    If adding A's history significantly improves prediction (P < 0.05),
    then A Granger-causes B.
    """
    
    def __init__(self, maxlag: int = 2, significance_level: float = 0.05):
        """
        Initialize Granger causality test calculator.
        
        Args:
            maxlag: Maximum lag to test (default: 2).
            significance_level: Significance level for causality test (default: 0.05).
        """
        self.maxlag = maxlag
        self.significance_level = significance_level
    
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
        **kwargs
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Test if returns_i Granger-causes returns_j.
        
        Args:
            returns_i: Return series for stock i.
            returns_j: Return series for stock j.
            **kwargs: Additional parameters.
        
        Returns:
            Tuple of (is_causal, p_value, direction).
            is_causal: True if returns_i causes returns_j (P < significance_level).
            p_value: Minimum P-value across all lags.
            direction: 'i->j', 'j->i', or None.
        """
        # Align indices
        aligned = pd.DataFrame({"A": returns_i, "B": returns_j}).dropna()
        
        if len(aligned) < self.maxlag + 10:  # Need enough data
            return False, 1.0, None
        
        try:
            # Suppress FutureWarning about verbose parameter
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*verbose.*")
                
                # Test A -> B
                test_ab = grangercausalitytests(
                    aligned[["B", "A"]],
                    maxlag=self.maxlag,
                )
                
                # Get minimum P-value across all lags
                p_values_ab = []
                for lag in range(1, self.maxlag + 1):
                    p_value = test_ab[lag][0]["ssr_ftest"][1]  # F-test P-value
                    p_values_ab.append(p_value)
                
                min_p_ab = min(p_values_ab)
                is_causal_ab = min_p_ab < self.significance_level
                
                # Test B -> A
                test_ba = grangercausalitytests(
                    aligned[["A", "B"]],
                    maxlag=self.maxlag,
                )
                
                p_values_ba = []
                for lag in range(1, self.maxlag + 1):
                    p_value = test_ba[lag][0]["ssr_ftest"][1]
                    p_values_ba.append(p_value)
                
                min_p_ba = min(p_values_ba)
                is_causal_ba = min_p_ba < self.significance_level
            
            # Determine direction (map to 'i->j' format for consistency)
            if is_causal_ab and is_causal_ba:
                # Both directions significant, choose stronger one
                if min_p_ab < min_p_ba:
                    direction = "i->j"
                    return True, min_p_ab, direction
                else:
                    direction = "j->i"
                    return True, min_p_ba, direction
            elif is_causal_ab:
                direction = "i->j"
                return True, min_p_ab, direction
            elif is_causal_ba:
                direction = "j->i"
                return True, min_p_ba, direction
            else:
                return False, max(min_p_ab, min_p_ba), None
                
        except Exception as e:
            logger.warning(f"Granger causality test failed: {e}")
            return False, 1.0, None
    
    def test(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Alias for calculate_pairwise (backward compatibility).
        
        Note: Returns direction in 'i->j'/'j->i' format, not 'A->B'/'B->A'.
        """
        return self.calculate_pairwise(series_a, series_b)
    
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate Granger causality matrix for all pairs.
        
        Args:
            returns: DataFrame with returns data.
            symbols: Optional list of symbols.
        
        Returns:
            Tuple of (causality_matrix, p_value_matrix).
            causality_matrix: Boolean matrix indicating causality.
            p_value_matrix: P-values for each test.
        """
        if symbols is None:
            symbols = returns.columns.tolist()
        
        n = len(symbols)
        causality_matrix = pd.DataFrame(False, index=symbols, columns=symbols)
        p_value_matrix = pd.DataFrame(1.0, index=symbols, columns=symbols)
        
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    continue
                
                is_causal, p_value, direction = self.calculate_pairwise(
                    returns[sym_i], returns[sym_j]
                )
                
                # Map direction from 'i->j'/'j->i' to symbol-based direction
                if direction == "i->j":
                    direction = f"{sym_i}->{sym_j}"
                elif direction == "j->i":
                    direction = f"{sym_j}->{sym_i}"
                
                if direction == f"{sym_i}->{sym_j}":
                    causality_matrix.loc[sym_i, sym_j] = True
                    p_value_matrix.loc[sym_i, sym_j] = p_value
        
        return causality_matrix, p_value_matrix


class TransferEntropy(ICorrelationCalculator):
    """
    Transfer Entropy calculator.
    
    Purpose: Capture nonlinear information flow. Measure the "certainty" of information
    transfer from A to B.
    
    Formula:
        TE_{A→B} = Σ p(B_{t+1}, B_t, A_t) log(p(B_{t+1} | B_t, A_t) / p(B_{t+1} | B_t))
    
    Calculation logic:
    1. Symbolization: Map returns to symbols (e.g.,大跌=-1, 震荡=0, 大涨=1)
    2. Entropy calculation: Calculate entropy reduction rate when knowing A's past
       to predict B's future.
    
    Logic: If ΔTE = TE_{A→B} - TE_{B→A} > 0, A is the information source.
    """
    
    def __init__(self, n_bins: int = 3, threshold_percentile: Tuple[float, float] = (33.3, 66.7)):
        """
        Initialize transfer entropy calculator.
        
        Args:
            n_bins: Number of bins for symbolization (default: 3, for -1, 0, 1).
            threshold_percentile: Percentiles for symbolization thresholds (default: (33.3, 66.7)).
        """
        self.n_bins = n_bins
        self.threshold_percentile = threshold_percentile
    
    def symbolize(self, series: pd.Series) -> pd.Series:
        """
        Symbolize return series.
        
        Args:
            series: Return series.
        
        Returns:
            Symbolized series (-1, 0, 1).
        """
        if len(series) == 0:
            return series
        
        # Calculate thresholds
        lower_threshold = np.percentile(series, self.threshold_percentile[0])
        upper_threshold = np.percentile(series, self.threshold_percentile[1])
        
        # Symbolize
        symbols = pd.Series(0, index=series.index)
        symbols[series < lower_threshold] = -1  # 大跌
        symbols[series > upper_threshold] = 1    # 大涨
        # 震荡 remains 0
        
        return symbols
    
    def calculate_entropy(self, symbols: pd.Series) -> float:
        """
        Calculate Shannon entropy of symbol series.
        
        Args:
            symbols: Symbol series.
        
        Returns:
            Entropy value.
        """
        if len(symbols) == 0:
            return 0.0
        
        # Count frequencies
        value_counts = symbols.value_counts()
        probabilities = value_counts / len(symbols)
        
        # Calculate entropy: H = -Σ p * log2(p)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def calculate_conditional_entropy(
        self,
        symbols_x: pd.Series,
        symbols_y: pd.Series,
    ) -> float:
        """
        Calculate conditional entropy H(Y|X).
        
        Args:
            symbols_x: Condition symbol series X.
            symbols_y: Target symbol series Y.
        
        Returns:
            Conditional entropy.
        """
        # Align indices
        aligned = pd.DataFrame({"X": symbols_x, "Y": symbols_y}).dropna()
        
        if len(aligned) == 0:
            return 0.0
        
        # Calculate conditional probabilities
        conditional_entropy = 0.0
        
        for x_val in aligned["X"].unique():
            x_mask = aligned["X"] == x_val
            y_given_x = aligned.loc[x_mask, "Y"]
            
            if len(y_given_x) == 0:
                continue
            
            # P(X=x)
            p_x = len(y_given_x) / len(aligned)
            
            # H(Y|X=x)
            h_y_given_x = self.calculate_entropy(y_given_x)
            
            conditional_entropy += p_x * h_y_given_x
        
        return conditional_entropy
    
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
        **kwargs
    ) -> Tuple[float, float, Optional[str]]:
        """
        Calculate transfer entropy from returns_i to returns_j and returns_j to returns_i.
        
        Args:
            returns_i: Return series for stock i.
            returns_j: Return series for stock j.
            **kwargs: Additional parameters.
        
        Returns:
            Tuple of (TE_i_to_j, TE_j_to_i, direction).
            direction: 'i->j', 'j->i', or None.
        """
        # Align indices
        aligned = pd.DataFrame({"A": returns_i, "B": returns_j}).dropna()
        
        if len(aligned) < 10:  # Need enough data
            return 0.0, 0.0, None
        
        # Symbolize
        symbols_a = self.symbolize(aligned["A"])
        symbols_b = self.symbolize(aligned["B"])
        
        # Calculate H(B_{t+1} | B_t)
        b_current = symbols_b.iloc[:-1]
        b_next = symbols_b.iloc[1:]
        h_b_next_given_b = self.calculate_conditional_entropy(b_current, b_next)
        
        # Calculate H(B_{t+1} | B_t, A_t)
        # Create combined condition (B_t, A_t)
        a_current = symbols_a.iloc[:-1]
        combined_condition = pd.Series(
            list(zip(b_current.values, a_current.values)),
            index=b_current.index
        )
        b_next_given_ab = self.calculate_conditional_entropy(combined_condition, b_next)
        
        # TE_{A→B} = H(B_{t+1} | B_t) - H(B_{t+1} | B_t, A_t)
        te_a_to_b = h_b_next_given_b - b_next_given_ab
        
        # Calculate H(A_{t+1} | A_t)
        a_current_for_ba = symbols_a.iloc[:-1]
        a_next = symbols_a.iloc[1:]
        h_a_next_given_a = self.calculate_conditional_entropy(a_current_for_ba, a_next)
        
        # Calculate H(A_{t+1} | A_t, B_t)
        # Create combined condition (A_t, B_t)
        b_current_for_ba = symbols_b.iloc[:-1]
        combined_condition_ba = pd.Series(
            list(zip(a_current_for_ba.values, b_current_for_ba.values)),
            index=a_current_for_ba.index
        )
        a_next_given_ab = self.calculate_conditional_entropy(combined_condition_ba, a_next)
        
        # TE_{B→A} = H(A_{t+1} | A_t) - H(A_{t+1} | A_t, B_t)
        te_b_to_a = h_a_next_given_a - a_next_given_ab
        
        # Determine direction
        delta_te = te_a_to_b - te_b_to_a
        
        if delta_te > 0:
            direction = "A->B"
        elif delta_te < 0:
            direction = "B->A"
        else:
            direction = None
        
        return te_a_to_b, te_b_to_a, direction
    
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate transfer entropy matrix for all pairs.
        
        Args:
            returns: DataFrame with returns data.
            symbols: Optional list of symbols.
        
        Returns:
            Tuple of (TE_matrix, direction_matrix).
            TE_matrix: Transfer entropy values.
            direction_matrix: Direction strings.
        """
        if symbols is None:
            symbols = returns.columns.tolist()
        
        n = len(symbols)
        te_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)
        direction_matrix = pd.DataFrame(None, index=symbols, columns=symbols, dtype=object)
        
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    continue
                
                te_i_to_j, te_j_to_i, direction = self.calculate_pairwise(
                    returns[sym_i], returns[sym_j]
                )
                
                # Map direction from 'i->j'/'j->i' to symbol-based direction
                if direction == "i->j":
                    direction = f"{sym_i}->{sym_j}"
                elif direction == "j->i":
                    direction = f"{sym_j}->{sym_i}"
                
                # Store maximum TE
                max_te = max(te_i_to_j, te_j_to_i)
                te_matrix.loc[sym_i, sym_j] = max_te
                direction_matrix.loc[sym_i, sym_j] = direction
        
        return te_matrix, direction_matrix


class IndustryCorrelation(ICorrelationCalculator):
    """
    Industry-level Correlation calculator.
    
    Purpose: Calculate correlation at industry level by aggregating stock returns
    within each industry and computing inter-industry correlations.
    
    Calculation logic:
    1. Group stocks by industry
    2. Calculate industry-level returns (e.g., average or weighted average)
    3. Calculate correlation matrix between industries
    
    This provides a higher-level view of market structure and can be used
    to identify industry clusters and sector relationships.
    """
    
    def __init__(
        self,
        industry_map: dict,
        aggregation_method: str = "mean",
        window: int = 60,
        threshold: Optional[float] = None,
    ):
        """
        Initialize industry correlation calculator.
        
        Args:
            industry_map: Dictionary mapping stock symbols to industry codes.
            aggregation_method: Method to aggregate stock returns within industry.
                              Options: 'mean' (default), 'median', 'weighted'.
            window: Lookback window size for correlation calculation (default: 60).
            threshold: Optional threshold for sparsification (e.g., 0.5).
        """
        self.industry_map = industry_map
        self.aggregation_method = aggregation_method
        self.window = window
        self.threshold = 0.0 if threshold is None else threshold
    
    def _aggregate_industry_returns(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate stock returns to industry-level returns.
        
        Args:
            returns: DataFrame with stock returns (columns: stock symbols, index: dates).
            symbols: Optional list of symbols to process.
        
        Returns:
            DataFrame with industry returns (columns: industry codes, index: dates).
        """
        if symbols is None:
            symbols = returns.columns.tolist()
        
        # Filter symbols that have industry mapping
        valid_symbols = [s for s in symbols if s in self.industry_map]
        if not valid_symbols:
            logger.warning("No valid symbols with industry mapping found")
            return pd.DataFrame()
        
        returns_subset = returns[valid_symbols]
        
        # Group by industry
        industry_returns = {}
        for symbol in valid_symbols:
            industry = self.industry_map[symbol]
            if industry not in industry_returns:
                industry_returns[industry] = []
            industry_returns[industry].append(symbol)
        
        # Aggregate returns for each industry
        industry_df = pd.DataFrame(index=returns_subset.index)
        
        for industry, stock_list in industry_returns.items():
            industry_stocks = [s for s in stock_list if s in returns_subset.columns]
            if not industry_stocks:
                continue
            
            industry_data = returns_subset[industry_stocks]
            
            if self.aggregation_method == "mean":
                industry_df[industry] = industry_data.mean(axis=1)
            elif self.aggregation_method == "median":
                industry_df[industry] = industry_data.median(axis=1)
            elif self.aggregation_method == "weighted":
                # Equal weight for now (could be extended to market cap weighted)
                industry_df[industry] = industry_data.mean(axis=1)
            else:
                logger.warning(f"Unknown aggregation method: {self.aggregation_method}, using mean")
                industry_df[industry] = industry_data.mean(axis=1)
        
        return industry_df
    
    def calculate_matrix(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate industry-level correlation matrix.
        
        Args:
            returns: DataFrame with stock returns (columns: stock symbols, index: dates).
            symbols: Optional list of symbols to calculate (if None, uses all columns).
            **kwargs: Additional parameters.
        
        Returns:
            Industry correlation matrix DataFrame (symmetric).
        """
        # Aggregate to industry level
        industry_returns = self._aggregate_industry_returns(returns, symbols)
        
        if industry_returns.empty:
            logger.warning("No industry returns calculated")
            return pd.DataFrame()
        
        # Use rolling window if data is longer than window
        if len(industry_returns) > self.window:
            industry_returns = industry_returns.iloc[-self.window:]
        
        # Calculate correlation matrix
        correlation_matrix = industry_returns.corr(method="spearman")
        
        # Apply threshold if specified
        correlation_matrix = correlation_matrix.where(
            abs(correlation_matrix) >= self.threshold, 0.0
        )
        
        return correlation_matrix
    
    def calculate_pairwise(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
        **kwargs
    ) -> float:
        """
        Calculate correlation between two industry return series.
        
        Note: This method expects industry-level returns, not stock-level returns.
        For stock-level inputs, use calculate_matrix() instead.
        
        Args:
            returns_i: Return series for industry i (or stock i if industry mapping exists).
            returns_j: Return series for industry j (or stock j if industry mapping exists).
            **kwargs: Additional parameters.
        
        Returns:
            Correlation coefficient.
        """
        # Align indices
        aligned = pd.DataFrame({"i": returns_i, "j": returns_j}).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        # Use rolling window if needed
        if len(aligned) > self.window:
            aligned = aligned.iloc[-self.window:]
        
        return aligned["i"].corr(aligned["j"], method="spearman")

