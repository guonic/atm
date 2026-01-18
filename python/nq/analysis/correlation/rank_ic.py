"""
Rank IC (Information Coefficient) calculation for correlation evaluation.

Rank IC measures the correlation between factor values (correlation scores)
and future returns using rank correlation (Spearman).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def calculate_rank_ic(
    correlation_scores: pd.Series,
    future_returns: pd.Series,
) -> float:
    """
    Calculate Rank IC (Information Coefficient) between correlation scores and future returns.
    
    Rank IC uses Spearman rank correlation, which is more robust to outliers
    than Pearson correlation.
    
    Args:
        correlation_scores: Series of correlation scores (factor values).
        future_returns: Series of future returns (target values).
    
    Returns:
        Rank IC value (between -1 and 1).
        Higher positive values indicate better predictive power.
    """
    # Align indices
    aligned = pd.DataFrame({
        'correlation': correlation_scores,
        'return': future_returns,
    }).dropna()
    
    if len(aligned) < 2:
        logger.warning("Insufficient data for Rank IC calculation")
        return 0.0
    
    try:
        # Calculate Spearman rank correlation
        rank_ic, p_value = spearmanr(
            aligned['correlation'],
            aligned['return'],
        )
        
        if pd.isna(rank_ic):
            return 0.0
        
        return float(rank_ic)
        
    except Exception as e:
        logger.error(f"Failed to calculate Rank IC: {e}", exc_info=True)
        return 0.0


def calculate_rank_ic_series(
    correlation_scores: pd.Series,
    future_returns: pd.Series,
    window: Optional[int] = None,
) -> pd.Series:
    """
    Calculate rolling Rank IC series.
    
    Args:
        correlation_scores: Series of correlation scores indexed by date.
        future_returns: Series of future returns indexed by date.
        window: Rolling window size (if None, calculates single IC).
    
    Returns:
        Series of Rank IC values.
    """
    if window is None:
        # Single IC calculation
        ic_value = calculate_rank_ic(correlation_scores, future_returns)
        return pd.Series([ic_value], index=[correlation_scores.index[0] if len(correlation_scores) > 0 else pd.Timestamp.now()])
    
    # Rolling window calculation
    aligned = pd.DataFrame({
        'correlation': correlation_scores,
        'return': future_returns,
    }).dropna()
    
    if len(aligned) < window:
        logger.warning(f"Insufficient data for rolling Rank IC (need {window}, have {len(aligned)})")
        return pd.Series(dtype=float)
    
    ic_values = []
    ic_dates = []
    
    for i in range(window - 1, len(aligned)):
        window_data = aligned.iloc[i - window + 1:i + 1]
        
        ic_value = calculate_rank_ic(
            window_data['correlation'],
            window_data['return'],
        )
        
        ic_values.append(ic_value)
        ic_dates.append(window_data.index[-1])
    
    return pd.Series(ic_values, index=ic_dates)


def calculate_ic_statistics(
    rank_ic_series: pd.Series,
) -> dict:
    """
    Calculate IC statistics from Rank IC series.
    
    Args:
        rank_ic_series: Series of Rank IC values.
    
    Returns:
        Dictionary with IC statistics:
        - mean_ic: Mean Rank IC
        - std_ic: Standard deviation of Rank IC
        - ic_ir: IC Information Ratio (mean / std)
        - positive_ic_ratio: Ratio of positive IC values
        - max_ic: Maximum IC value
        - min_ic: Minimum IC value
    """
    if len(rank_ic_series) == 0:
        return {
            'mean_ic': 0.0,
            'std_ic': 0.0,
            'ic_ir': 0.0,
            'positive_ic_ratio': 0.0,
            'max_ic': 0.0,
            'min_ic': 0.0,
        }
    
    mean_ic = float(rank_ic_series.mean())
    std_ic = float(rank_ic_series.std())
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    positive_ic_ratio = float((rank_ic_series > 0).sum() / len(rank_ic_series))
    max_ic = float(rank_ic_series.max())
    min_ic = float(rank_ic_series.min())
    
    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ic_ir': ic_ir,
        'positive_ic_ratio': positive_ic_ratio,
        'max_ic': max_ic,
        'min_ic': min_ic,
    }
