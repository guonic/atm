# -*- coding: utf-8 -*-
"""
teapot_box_quality_filter.py

Filter stocks by box quality metrics from evaluation results.
"""

import argparse
import polars as pl

def filter_quality_stocks(
    csv_file: str,
    min_box_ratio: float = 0.25,
    max_avg_box_width: float = 0.15,
    max_std_box_width: float = 0.10,
    max_max_box_width: float = 0.50,
) -> pl.DataFrame:
    """
    Filter stocks by box quality metrics.

    Args:
        csv_file: Path to per-stock evaluation CSV file.
        min_box_ratio: Minimum box ratio (default: 0.25).
        max_avg_box_width: Maximum average box width (default: 0.15).
        max_std_box_width: Maximum std box width (default: 0.10).
        max_max_box_width: Maximum max box width (default: 0.50).

    Returns:
        Filtered DataFrame with quality stocks.
    """
    df = pl.read_csv(csv_file)

    # Filter by quality criteria
    quality_stocks = df.filter(
        (pl.col("box_ratio") >= min_box_ratio) &
        (pl.col("avg_box_width") <= max_avg_box_width) &
        (pl.col("std_box_width") <= max_std_box_width) &
        (pl.col("max_box_width") <= max_max_box_width)
    ).sort("box_ratio", descending=True)

    return quality_stocks


def calculate_quality_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate quality score for each stock.

    Args:
        df: DataFrame with box metrics.

    Returns:
        DataFrame with added quality_score column.
    """
    # Normalize metrics to 0-1 range
    df = df.with_columns([
        # Box ratio (already 0-1, higher is better)
        pl.col("box_ratio").alias("score_box_ratio"),
        # Compactness: 1 - avg_box_width (normalized, assuming max 0.5)
        (1 - pl.col("avg_box_width") / 0.5).clip(0, 1).alias("score_compactness"),
        # Stability: 1 - std_box_width (normalized, assuming max 0.5)
        (1 - pl.col("std_box_width") / 0.5).clip(0, 1).alias("score_stability"),
        # No outliers: 1 - max_box_width (normalized, assuming max 1.0)
        (1 - pl.col("max_box_width").clip(0, 1)).alias("score_no_outliers"),
    ])

    # Calculate weighted quality score
    df = df.with_columns([
        (
            pl.col("score_box_ratio") * 0.3 +
            pl.col("score_compactness") * 0.3 +
            pl.col("score_stability") * 0.2 +
            pl.col("score_no_outliers") * 0.2
        ).alias("quality_score")
    ])

    return df.sort("quality_score", descending=True)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Filter stocks by box quality metrics"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to per-stock evaluation CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: print to stdout)",
    )
    parser.add_argument(
        "--min-box-ratio",
        type=float,
        default=0.25,
        help="Minimum box ratio (default: 0.25)",
    )
    parser.add_argument(
        "--max-avg-box-width",
        type=float,
        default=0.15,
        help="Maximum average box width (default: 0.15)",
    )
    parser.add_argument(
        "--max-std-box-width",
        type=float,
        default=0.10,
        help="Maximum std box width (default: 0.10)",
    )
    parser.add_argument(
        "--max-max-box-width",
        type=float,
        default=0.50,
        help="Maximum max box width (default: 0.50)",
    )
    parser.add_argument(
        "--with-score",
        action="store_true",
        help="Calculate and include quality score",
    )

    args = parser.parse_args()

    # Filter quality stocks
    quality_stocks = filter_quality_stocks(
        csv_file=args.csv_file,
        min_box_ratio=args.min_box_ratio,
        max_avg_box_width=args.max_avg_box_width,
        max_std_box_width=args.max_std_box_width,
        max_max_box_width=args.max_max_box_width,
    )

    # Calculate quality score if requested
    if args.with_score:
        quality_stocks = calculate_quality_score(quality_stocks)

    # Output results
    if args.output:
        quality_stocks.write_csv(args.output)
        print(f"Filtered {len(quality_stocks)} quality stocks, saved to {args.output}")
    else:
        print(f"\nFiltered {len(quality_stocks)} quality stocks:\n")
        print(quality_stocks.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
