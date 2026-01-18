"""
Sensitivity Analysis and Report Generator for correlation algorithm evaluation.

This module implements the third stage of the correlation test framework:
generating analysis reports with optimal parameter combinations, ranking drift
visualizations, and correlation threshold heatmaps.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not available. Plotting functions will be disabled.")


class SensitivityAnalyzer:
    """
    Generate sensitivity analysis reports for correlation algorithm evaluation.
    
    This class produces:
    1. Optimal combination matrix: Best algorithm/window combinations for each target period
    2. Ranking drift plots: Visualization of ranking half-life
    3. Heatmaps: Correlation threshold vs future return correlation
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize sensitivity analyzer.
        
        Args:
            output_dir: Output directory for saving plots (default: current directory).
        """
        self.output_dir = output_dir or "."
        logger.info(f"SensitivityAnalyzer initialized: output_dir={self.output_dir}")
    
    def generate_optimal_combination_matrix(
        self,
        optimization_results: pd.DataFrame,
        target_periods: Optional[List[int]] = None,
        metric: str = 'win_rate_lift',
    ) -> pd.DataFrame:
        """
        Generate optimal combination matrix showing the best algorithm/window for each period.
        
        Args:
            optimization_results: Results DataFrame from CorrelationOptimizer.optimize().
            target_periods: List of target periods to analyze (default: all periods).
            metric: Metric to optimize (default: 'win_rate_lift').
        
        Returns:
            DataFrame with optimal combinations:
            - target_period: Target holding period
            - algorithm: Best algorithm
            - window: Best window parameter
            - threshold: Best threshold
            - metric_value: Best metric value
        """
        if optimization_results.empty:
            logger.warning("Empty optimization results provided")
            return pd.DataFrame()
        
        if target_periods is None:
            target_periods = sorted(optimization_results['target_period'].unique().tolist())
        
        optimal_combinations = []
        
        for period in target_periods:
            period_data = optimization_results[
                optimization_results['target_period'] == period
            ]
            
            if period_data.empty:
                continue
            
            # Find best combination for this period
            best_idx = period_data[metric].idxmax()
            best_row = period_data.loc[best_idx]
            
            optimal_combinations.append({
                'target_period': period,
                'algorithm': best_row['algorithm'],
                'window': best_row['window'],
                'threshold': best_row['threshold'],
                'metric_value': best_row[metric],
                'baseline_win_rate': best_row.get('baseline_win_rate', 0.0),
                'filtered_win_rate': best_row.get('filtered_win_rate', 0.0),
                'win_rate_lift': best_row.get('win_rate_lift', 0.0),
            })
        
        result_df = pd.DataFrame(optimal_combinations)
        
        logger.info(
            f"Generated optimal combination matrix: {len(result_df)} periods analyzed"
        )
        
        return result_df
    
    def plot_ranking_drift(
        self,
        return_matrix: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot ranking drift visualization showing ranking half-life.
        
        Args:
            return_matrix: Return matrix from ReturnMatrixGenerator.generate().
            output_path: Output path for plot (default: auto-generated).
        """
        if not HAS_PLOTTING:
            logger.warning("Plotting libraries not available. Skipping plot_ranking_drift.")
            return
        
        if return_matrix.empty:
            logger.warning("Empty return matrix provided")
            return
        
        # Extract ranking evolution data
        rank_cols = [col for col in return_matrix.columns if col.startswith('rank_T+')]
        
        if not rank_cols:
            logger.warning("No ranking evolution columns found")
            return
        
        # Calculate ranking changes
        drift_data = []
        
        for (date, symbol), row in return_matrix.iterrows():
            entry_rank = row.get('entry_rank', None)
            if pd.isna(entry_rank):
                continue
            
            for rank_col in rank_cols:
                period = int(rank_col.replace('rank_T+', ''))
                future_rank = row[rank_col]
                
                if pd.isna(future_rank):
                    continue
                
                rank_change = future_rank - entry_rank
                rank_change_pct = rank_change / entry_rank if entry_rank > 0 else 0.0
                
                drift_data.append({
                    'period': period,
                    'rank_change': rank_change,
                    'rank_change_pct': rank_change_pct,
                    'entry_rank': entry_rank,
                    'future_rank': future_rank,
                })
        
        if not drift_data:
            logger.warning("No drift data to plot")
            return
        
        drift_df = pd.DataFrame(drift_data)
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Average rank change by period
        avg_drift = drift_df.groupby('period')['rank_change'].mean()
        axes[0].plot(avg_drift.index, avg_drift.values, marker='o', linewidth=2)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Days After Entry')
        axes[0].set_ylabel('Average Rank Change')
        axes[0].set_title('Ranking Drift: Average Rank Change Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Ranking half-life distribution
        # Calculate half-life (days until rank doubles)
        halflife_data = []
        for (date, symbol), row in return_matrix.iterrows():
            entry_rank = row.get('entry_rank', None)
            if pd.isna(entry_rank):
                continue
            
            threshold_rank = entry_rank * 2  # Double the rank
            
            for rank_col in rank_cols:
                period = int(rank_col.replace('rank_T+', ''))
                future_rank = row[rank_col]
                
                if pd.isna(future_rank):
                    continue
                
                if future_rank >= threshold_rank:
                    halflife_data.append(period)
                    break
        
        if halflife_data:
            axes[1].hist(halflife_data, bins=20, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Ranking Half-Life (Days)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Ranking Half-Life')
            axes[1].axvline(
                x=np.mean(halflife_data),
                color='r',
                linestyle='--',
                label=f'Mean: {np.mean(halflife_data):.1f} days'
            )
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = f"{self.output_dir}/ranking_drift.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ranking drift plot saved to {output_path}")
        plt.close()
    
    def plot_correlation_threshold_heatmap(
        self,
        optimization_results: pd.DataFrame,
        target_period: int = 5,
        metric: str = 'win_rate_lift',
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot heatmap showing correlation threshold vs future return correlation.
        
        Args:
            optimization_results: Results DataFrame from CorrelationOptimizer.optimize().
            target_period: Target holding period to analyze (default: 5).
            metric: Metric to visualize (default: 'win_rate_lift').
            output_path: Output path for plot (default: auto-generated).
        """
        if not HAS_PLOTTING:
            logger.warning("Plotting libraries not available. Skipping plot_correlation_threshold_heatmap.")
            return
        
        if optimization_results.empty:
            logger.warning("Empty optimization results provided")
            return
        
        # Filter for target period
        period_data = optimization_results[
            optimization_results['target_period'] == target_period
        ]
        
        if period_data.empty:
            logger.warning(f"No data found for target period {target_period}")
            return
        
        # Create pivot table: algorithm x threshold -> metric value
        pivot_data = period_data.pivot_table(
            values=metric,
            index='algorithm',
            columns='threshold',
            aggfunc='mean',  # Average across window parameters
        )
        
        if pivot_data.empty:
            logger.warning("Empty pivot table")
            return
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': metric},
        )
        plt.title(f'Correlation Threshold Heatmap (Target Period: T+{target_period})')
        plt.xlabel('Correlation Threshold')
        plt.ylabel('Algorithm')
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = f"{self.output_dir}/correlation_threshold_heatmap_T+{target_period}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation threshold heatmap saved to {output_path}")
        plt.close()
    
    def generate_summary_report(
        self,
        optimization_results: pd.DataFrame,
        return_matrix: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive summary report.
        
        Args:
            optimization_results: Results DataFrame from CorrelationOptimizer.optimize().
            return_matrix: Return matrix from ReturnMatrixGenerator.generate().
            output_path: Output path for CSV report (default: auto-generated).
        
        Returns:
            Summary report DataFrame.
        """
        if optimization_results.empty:
            logger.warning("Empty optimization results provided")
            return pd.DataFrame()
        
        # Generate optimal combination matrix
        optimal_matrix = self.generate_optimal_combination_matrix(optimization_results)
        
        # Generate summary statistics
        summary_stats = {
            'total_combinations': len(optimization_results),
            'algorithms_tested': optimization_results['algorithm'].nunique(),
            'windows_tested': optimization_results['window'].nunique(),
            'thresholds_tested': optimization_results['threshold'].nunique(),
            'periods_tested': optimization_results['target_period'].nunique(),
            'best_win_rate_lift': optimization_results['win_rate_lift'].max(),
            'avg_win_rate_lift': optimization_results['win_rate_lift'].mean(),
            'best_algorithm': optimization_results.loc[
                optimization_results['win_rate_lift'].idxmax(), 'algorithm'
            ],
        }
        
        # Combine results
        report = {
            'optimal_combinations': optimal_matrix,
            'summary_statistics': pd.Series(summary_stats),
        }
        
        # Save to CSV if path provided
        if output_path is None:
            output_path = f"{self.output_dir}/correlation_optimization_report.csv"
        
        # Save optimal combinations
        optimal_matrix.to_csv(output_path, index=False)
        logger.info(f"Summary report saved to {output_path}")
        
        return optimal_matrix
