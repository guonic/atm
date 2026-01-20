"""
Compare correlation optimization results.

This script provides various comparison views of the optimization results:
1. Algorithm comparison: Compare performance across different algorithms
2. Parameter comparison: Compare different window/threshold combinations
3. Period comparison: Compare performance across different holding periods
4. Best vs Worst: Identify best and worst performing combinations
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_results(results_path: Path) -> pd.DataFrame:
    """Load optimization results from CSV."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    print(f"✅ Loaded {len(df)} results from {results_path}")
    return df


def compare_algorithms(df: pd.DataFrame, metric: str = 'win_rate_lift') -> pd.DataFrame:
    """
    Compare performance across different algorithms.
    
    Args:
        df: Optimization results DataFrame.
        metric: Metric to compare (default: 'win_rate_lift').
    
    Returns:
        Comparison DataFrame grouped by algorithm.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available: {list(df.columns)}")
    
    comparison = df.groupby('algorithm').agg({
        metric: ['mean', 'std', 'min', 'max', 'count'],
        'filtered_win_rate': 'mean',
        'baseline_win_rate': 'mean',
        'filtered_avg_return': 'mean',
        'baseline_avg_return': 'mean',
        'profit_loss_ratio': 'mean',
        'max_drawdown': 'mean',
    }).round(4)
    
    # Flatten column names
    comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in comparison.columns.values]
    
    # Sort by metric mean (descending)
    comparison = comparison.sort_values(f'{metric}_mean', ascending=False)
    
    return comparison


def compare_parameters(df: pd.DataFrame, metric: str = 'win_rate_lift') -> pd.DataFrame:
    """
    Compare performance across different parameter combinations.
    
    Args:
        df: Optimization results DataFrame.
        metric: Metric to compare.
    
    Returns:
        Comparison DataFrame grouped by window and threshold.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")
    
    comparison = df.groupby(['window', 'threshold']).agg({
        metric: ['mean', 'std', 'count'],
        'filtered_win_rate': 'mean',
        'filtered_avg_return': 'mean',
        'profit_loss_ratio': 'mean',
    }).round(4)
    
    comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in comparison.columns.values]
    comparison = comparison.sort_values(f'{metric}_mean', ascending=False)
    
    return comparison


def compare_periods(df: pd.DataFrame, metric: str = 'win_rate_lift') -> pd.DataFrame:
    """
    Compare performance across different holding periods.
    
    Args:
        df: Optimization results DataFrame.
        metric: Metric to compare.
    
    Returns:
        Comparison DataFrame grouped by target_period.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")
    
    comparison = df.groupby('target_period').agg({
        metric: ['mean', 'std', 'min', 'max', 'count'],
        'filtered_win_rate': 'mean',
        'baseline_win_rate': 'mean',
        'filtered_avg_return': 'mean',
        'baseline_avg_return': 'mean',
        'profit_loss_ratio': 'mean',
        'max_drawdown': 'mean',
    }).round(4)
    
    comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in comparison.columns.values]
    comparison = comparison.sort_index()
    
    return comparison


def find_best_combinations(df: pd.DataFrame, metric: str = 'win_rate_lift', top_n: int = 10) -> pd.DataFrame:
    """
    Find best performing combinations.
    
    Args:
        df: Optimization results DataFrame.
        metric: Metric to rank by.
        top_n: Number of top combinations to return.
    
    Returns:
        DataFrame with top N combinations.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")
    
    # Sort by metric (descending)
    best = df.nlargest(top_n, metric)[
        ['algorithm', 'window', 'threshold', 'target_period', 
         'win_rate_lift', 'filtered_win_rate', 'baseline_win_rate',
         'filtered_avg_return', 'baseline_avg_return', 
         'profit_loss_ratio', 'max_drawdown', 'filtered_count', 'total_count']
    ]
    
    return best


def find_worst_combinations(df: pd.DataFrame, metric: str = 'win_rate_lift', top_n: int = 10) -> pd.DataFrame:
    """
    Find worst performing combinations.
    
    Args:
        df: Optimization results DataFrame.
        metric: Metric to rank by.
        top_n: Number of worst combinations to return.
    
    Returns:
        DataFrame with worst N combinations.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")
    
    worst = df.nsmallest(top_n, metric)[
        ['algorithm', 'window', 'threshold', 'target_period',
         'win_rate_lift', 'filtered_win_rate', 'baseline_win_rate',
         'filtered_avg_return', 'baseline_avg_return',
         'profit_loss_ratio', 'max_drawdown', 'filtered_count', 'total_count']
    ]
    
    return worst


def generate_summary_report(df: pd.DataFrame, output_path: Optional[Path] = None) -> str:
    """
    Generate a comprehensive summary report.
    
    Args:
        df: Optimization results DataFrame.
        output_path: Optional path to save report.
    
    Returns:
        Report string.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CORRELATION OPTIMIZATION RESULTS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("📊 Overall Statistics:")
    report_lines.append(f"   Total combinations tested: {len(df)}")
    report_lines.append(f"   Algorithms: {df['algorithm'].nunique()} ({', '.join(df['algorithm'].unique())})")
    report_lines.append(f"   Windows: {df['window'].nunique()} ({', '.join(map(str, sorted(df['window'].unique())))})")
    report_lines.append(f"   Thresholds: {df['threshold'].nunique()} ({', '.join(map(str, sorted(df['threshold'].unique())))})")
    report_lines.append(f"   Holding periods: {df['target_period'].nunique()} ({', '.join(map(str, sorted(df['target_period'].unique())))})")
    report_lines.append("")
    
    # Win rate lift statistics
    report_lines.append("📈 Win Rate Lift Statistics:")
    report_lines.append(f"   Mean: {df['win_rate_lift'].mean():.4f}")
    report_lines.append(f"   Median: {df['win_rate_lift'].median():.4f}")
    report_lines.append(f"   Std: {df['win_rate_lift'].std():.4f}")
    report_lines.append(f"   Min: {df['win_rate_lift'].min():.4f}")
    report_lines.append(f"   Max: {df['win_rate_lift'].max():.4f}")
    report_lines.append(f"   Positive lifts: {len(df[df['win_rate_lift'] > 0])} ({len(df[df['win_rate_lift'] > 0]) / len(df) * 100:.1f}%)")
    report_lines.append("")
    
    # Best combinations
    report_lines.append("🏆 Top 5 Best Combinations (by Win Rate Lift):")
    best = find_best_combinations(df, metric='win_rate_lift', top_n=5)
    for idx, row in best.iterrows():
        report_lines.append(
            f"   {row['algorithm']} | window={row['window']} | threshold={row['threshold']} | "
            f"period={row['target_period']} | lift={row['win_rate_lift']:.4f} | "
            f"filtered_win_rate={row['filtered_win_rate']:.4f}"
        )
    report_lines.append("")
    
    # Worst combinations
    report_lines.append("⚠️  Top 5 Worst Combinations (by Win Rate Lift):")
    worst = find_worst_combinations(df, metric='win_rate_lift', top_n=5)
    for idx, row in worst.iterrows():
        report_lines.append(
            f"   {row['algorithm']} | window={row['window']} | threshold={row['threshold']} | "
            f"period={row['target_period']} | lift={row['win_rate_lift']:.4f} | "
            f"filtered_win_rate={row['filtered_win_rate']:.4f}"
        )
    report_lines.append("")
    
    # Algorithm comparison
    report_lines.append("🔬 Algorithm Comparison (Average Win Rate Lift):")
    algo_comp = compare_algorithms(df, metric='win_rate_lift')
    for algo in algo_comp.index:
        mean_lift = algo_comp.loc[algo, 'win_rate_lift_mean']
        count = algo_comp.loc[algo, 'win_rate_lift_count']
        report_lines.append(f"   {algo}: {mean_lift:.4f} (n={int(count)})")
    report_lines.append("")
    
    # Period comparison
    report_lines.append("📅 Period Comparison (Average Win Rate Lift):")
    period_comp = compare_periods(df, metric='win_rate_lift')
    for period in period_comp.index:
        mean_lift = period_comp.loc[period, 'win_rate_lift_mean']
        count = period_comp.loc[period, 'win_rate_lift_count']
        report_lines.append(f"   T+{period}: {mean_lift:.4f} (n={int(count)})")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    if output_path:
        output_path.write_text(report, encoding='utf-8')
        print(f"✅ Summary report saved to: {output_path}")
    
    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare correlation optimization results"
    )
    parser.add_argument(
        '--results',
        type=str,
        default='outputs/optimization_results.csv',
        help='Path to optimization results CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for comparison reports'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='win_rate_lift',
        choices=['win_rate_lift', 'filtered_win_rate', 'filtered_avg_return', 'profit_loss_ratio'],
        help='Metric to use for comparison'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top/worst combinations to show'
    )
    parser.add_argument(
        '--compare',
        type=str,
        choices=['algorithms', 'parameters', 'periods', 'all'],
        default='all',
        help='Type of comparison to perform'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results)
    df = load_results(results_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CORRELATION OPTIMIZATION RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # Generate comparisons
    if args.compare in ['algorithms', 'all']:
        print("🔬 Algorithm Comparison:")
        algo_comp = compare_algorithms(df, metric=args.metric)
        print(algo_comp)
        print()
        
        algo_path = output_dir / 'algorithm_comparison.csv'
        algo_comp.to_csv(algo_path)
        print(f"✅ Saved to: {algo_path}\n")
    
    if args.compare in ['parameters', 'all']:
        print("⚙️  Parameter Comparison:")
        param_comp = compare_parameters(df, metric=args.metric)
        print(param_comp.head(20))
        print()
        
        param_path = output_dir / 'parameter_comparison.csv'
        param_comp.to_csv(param_path)
        print(f"✅ Saved to: {param_path}\n")
    
    if args.compare in ['periods', 'all']:
        print("📅 Period Comparison:")
        period_comp = compare_periods(df, metric=args.metric)
        print(period_comp)
        print()
        
        period_path = output_dir / 'period_comparison.csv'
        period_comp.to_csv(period_path)
        print(f"✅ Saved to: {period_path}\n")
    
    # Best and worst combinations
    print("🏆 Top {} Best Combinations:".format(args.top_n))
    best = find_best_combinations(df, metric=args.metric, top_n=args.top_n)
    print(best)
    print()
    
    best_path = output_dir / 'best_combinations.csv'
    best.to_csv(best_path, index=False)
    print(f"✅ Saved to: {best_path}\n")
    
    print("⚠️  Top {} Worst Combinations:".format(args.top_n))
    worst = find_worst_combinations(df, metric=args.metric, top_n=args.top_n)
    print(worst)
    print()
    
    worst_path = output_dir / 'worst_combinations.csv'
    worst.to_csv(worst_path, index=False)
    print(f"✅ Saved to: {worst_path}\n")
    
    # Generate summary report
    print("📊 Generating Summary Report...")
    summary_path = output_dir / 'comparison_summary.txt'
    report = generate_summary_report(df, output_path=summary_path)
    print(report)
    
    print("\n" + "=" * 80)
    print("✅ All comparison reports saved to:", output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
