"""
Correlation analysis methods for stock structure features.

Implements five core correlation calculation methods:
1. Dynamic Cross-Sectional Correlation
2. Cross-Lagged Correlation
3. Volatility Sync
4. Granger Causality
5. Transfer Entropy
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


class DynamicCrossSectionalCorrelation:
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
    
    def calculate(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
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


class CrossLaggedCorrelation:
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
    
    def calculate_directed(
        self,
        returns_i: pd.Series,
        returns_j: pd.Series,
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
                
                corr_i_to_j, corr_j_to_i, direction = self.calculate_directed(
                    returns[sym_i], returns[sym_j]
                )
                
                # Store maximum correlation
                max_corr = max(corr_i_to_j, corr_j_to_i)
                corr_matrix.loc[sym_i, sym_j] = max_corr
                direction_matrix.loc[sym_i, sym_j] = direction
        
        return corr_matrix, direction_matrix


class VolatilitySync:
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
    
    def calculate_sync_rate(
        self,
        range_i: pd.Series,
        range_j: pd.Series,
    ) -> float:
        """
        Calculate sync rate between two stocks.
        
        Args:
            range_i: Range series for stock i.
            range_j: Range series for stock j.
        
        Returns:
            Sync rate (0-1, closer to 1 means more synchronized).
        """
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
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate volatility sync matrix for all pairs.
        
        Args:
            highs: DataFrame with high prices (columns: symbols, index: dates).
            lows: DataFrame with low prices (columns: symbols, index: dates).
            symbols: Optional list of symbols.
        
        Returns:
            Sync matrix DataFrame.
        """
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


class GrangerCausality:
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
    
    def test(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Test if A Granger-causes B.
        
        Args:
            series_a: Time series A.
            series_b: Time series B.
        
        Returns:
            Tuple of (is_causal, p_value, direction).
            is_causal: True if A causes B (P < significance_level).
            p_value: Minimum P-value across all lags.
            direction: 'A->B', 'B->A', or None.
        """
        # Align indices
        aligned = pd.DataFrame({"A": series_a, "B": series_b}).dropna()
        
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
            
            # Determine direction
            if is_causal_ab and is_causal_ba:
                # Both directions significant, choose stronger one
                if min_p_ab < min_p_ba:
                    direction = "A->B"
                    return True, min_p_ab, direction
                else:
                    direction = "B->A"
                    return True, min_p_ba, direction
            elif is_causal_ab:
                direction = "A->B"
                return True, min_p_ab, direction
            elif is_causal_ba:
                direction = "B->A"
                return True, min_p_ba, direction
            else:
                return False, max(min_p_ab, min_p_ba), None
                
        except Exception as e:
            logger.warning(f"Granger causality test failed: {e}")
            return False, 1.0, None
    
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
                
                is_causal, p_value, direction = self.test(
                    returns[sym_i], returns[sym_j]
                )
                
                if direction == f"{sym_i}->{sym_j}":
                    causality_matrix.loc[sym_i, sym_j] = True
                    p_value_matrix.loc[sym_i, sym_j] = p_value
        
        return causality_matrix, p_value_matrix


class TransferEntropy:
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
    
    def calculate(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> Tuple[float, float, Optional[str]]:
        """
        Calculate transfer entropy from A to B and B to A.
        
        Args:
            series_a: Time series A.
            series_b: Time series B.
        
        Returns:
            Tuple of (TE_A_to_B, TE_B_to_A, direction).
            direction: 'A->B', 'B->A', or None.
        """
        # Align indices
        aligned = pd.DataFrame({"A": series_a, "B": series_b}).dropna()
        
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
                
                te_i_to_j, te_j_to_i, direction = self.calculate(
                    returns[sym_i], returns[sym_j]
                )
                
                # Store maximum TE
                max_te = max(te_i_to_j, te_j_to_i)
                te_matrix.loc[sym_i, sym_j] = max_te
                direction_matrix.loc[sym_i, sym_j] = direction
        
        return te_matrix, direction_matrix

